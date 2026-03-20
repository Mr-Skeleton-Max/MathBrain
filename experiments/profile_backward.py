#!/usr/bin/env python3
"""Profile backward pass — 新版 (dense wp_grad + per-slot bilinear)"""
import sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, '/root/autodl-tmp')

from MathBrain.config import MathBrainConfig
from MathBrain.streaming_preprocessor import build_vocab
from MathBrain.gpu_preprocessor import gpu_preprocess, iter_batches_gpu
from MathBrain.gpu_phi_encoder import GPUPhiEncoder
from MathBrain.triton_kernels import triton_bilinear, triton_word_proj, build_wp_dense
from MathBrain.retina import HashRetina

cfg = MathBrainConfig(
    N=8, RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
    D_PHI=32, CP_RANK=32)

corpus = [l.strip() for l in open('datasets/tinystories_1000.txt') if l.strip()][:1000]
retina = HashRetina(cfg)
vocab = build_vocab(corpus, retina)
device = torch.device('cuda')
data = gpu_preprocess(corpus, vocab, cfg, device, verbose=False)
phi_gpu = GPUPhiEncoder(cfg, device='cuda')

S, V, D = vocab.S, vocab.V, cfg.D_PHI
dD = cfg.CP_RANK * D
E_src = torch.nn.Embedding(S, dD).to(device)
E_tgt = torch.nn.Parameter(torch.randn(S, dD, device=device) * 0.01)
wp_off = torch.from_numpy(vocab.wp_word_offsets).to(device)
wp_idx = torch.from_numpy(vocab.wp_slot_indices).to(device)
wp_wt = torch.from_numpy(vocab.wp_slot_weights).to(device)
wp_dense = build_wp_dense(wp_off, wp_idx, wp_wt, S, V, device)
optimizer = torch.optim.AdamW([E_src.weight, E_tgt], lr=0.01)

# 取第一个 batch
for b in iter_batches_gpu(data, 4096, device, shuffle=False):
    break

# Warmup
for _ in range(5):
    phi_n = phi_gpu.encode_normalized(b.flat_Q)
    Bi = b.counts.shape[0]
    optimizer.zero_grad(set_to_none=True)
    ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                          b.pos_lo, b.counts, Bi, dD, D, S)
    scores = ctx @ E_tgt.t()
    logits = triton_word_proj(scores, wp_off, wp_idx, wp_wt, Bi, S, V, wp_dense)
    loss = F.cross_entropy(logits, b.targets)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()

print(f"Batch: {Bi} positions, flat_slots: {b.flat_slots.shape}")

N_RUNS = 20
times = {}

for _ in range(N_RUNS):
    phi_n = phi_gpu.encode_normalized(b.flat_Q)
    Bi = b.counts.shape[0]
    optimizer.zero_grad(set_to_none=True)
    ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                          b.pos_lo, b.counts, Bi, dD, D, S)
    scores = ctx @ E_tgt.t()
    logits = triton_word_proj(scores, wp_off, wp_idx, wp_wt, Bi, S, V, wp_dense)
    loss = F.cross_entropy(logits, b.targets)

    torch.cuda.synchronize(); t0 = time.perf_counter()

    # 1. CE
    grad_logits = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.cuda.synchronize(); t1 = time.perf_counter()

    # 2. word_proj backward (dense matmul now!)
    grad_scores = torch.autograd.grad(logits, scores, grad_logits, retain_graph=True)[0]
    torch.cuda.synchronize(); t2 = time.perf_counter()

    # 3. E_tgt backward
    grad_E_tgt = torch.autograd.grad(scores, E_tgt, grad_scores, retain_graph=True)[0]
    torch.cuda.synchronize(); t3 = time.perf_counter()

    # 4. ctx backward
    grad_ctx = torch.autograd.grad(scores, ctx, grad_scores, retain_graph=True)[0]
    torch.cuda.synchronize(); t4 = time.perf_counter()

    # 5. E_src backward (per-slot Triton now!)
    grad_E_src = torch.autograd.grad(ctx, E_src.weight, grad_ctx)[0]
    torch.cuda.synchronize(); t5 = time.perf_counter()

    for k, v in [('CE_grad', t1-t0), ('wp_grad', t2-t1), ('Etgt_grad', t3-t2),
                 ('ctx_grad', t4-t3), ('Esrc_grad', t5-t4), ('total', t5-t0)]:
        times.setdefault(k, 0.0)
        times[k] += v

# Full backward
for _ in range(N_RUNS):
    phi_n = phi_gpu.encode_normalized(b.flat_Q)
    Bi = b.counts.shape[0]
    optimizer.zero_grad(set_to_none=True)
    ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                          b.pos_lo, b.counts, Bi, dD, D, S)
    scores = ctx @ E_tgt.t()
    logits = triton_word_proj(scores, wp_off, wp_idx, wp_wt, Bi, S, V, wp_dense)
    loss = F.cross_entropy(logits, b.targets)
    torch.cuda.synchronize(); tb0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(); tb1 = time.perf_counter()
    times.setdefault('full_backward', 0.0)
    times['full_backward'] += (tb1 - tb0)
    torch.cuda.synchronize(); ts0 = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(); ts1 = time.perf_counter()
    times.setdefault('opt_step', 0.0)
    times['opt_step'] += (ts1 - ts0)

print(f"\nBackward breakdown ({N_RUNS} runs avg):")
for k in ['CE_grad', 'wp_grad', 'Etgt_grad', 'ctx_grad', 'Esrc_grad', 'total',
          'full_backward', 'opt_step']:
    ms = times.get(k, 0) / N_RUNS * 1000
    print(f"  {k:15s}: {ms:.3f}ms")

# Compare: old=9.3ms, new=?
print(f"\nOld backward was 8.8ms, wp_grad was 5.3ms, Esrc_grad was 2.5ms")
