#!/usr/bin/env python3
"""详细 profile: E_word fusion 版"""
import sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, '/root/autodl-tmp')

from MathBrain.config import MathBrainConfig
from MathBrain.streaming_preprocessor import build_vocab
from MathBrain.gpu_preprocessor import gpu_preprocess, iter_batches_gpu, _batch_ema_and_filter
from MathBrain.gpu_phi_encoder import GPUPhiEncoder
from MathBrain.triton_kernels import triton_bilinear, build_wp_dense
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

batches = []
for b in iter_batches_gpu(data, 4096, device, shuffle=False):
    batches.append(b)
    if len(batches) >= 3:
        break

# Warmup
for _ in range(5):
    b = batches[0]
    phi_n = phi_gpu.encode_normalized(b.flat_Q)
    Bi = b.counts.shape[0]
    optimizer.zero_grad(set_to_none=True)
    ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                          b.pos_lo, b.counts, Bi, dD, D, S)
    E_word = wp_dense @ E_tgt
    logits = ctx @ E_word.t()
    loss = F.cross_entropy(logits, b.targets)
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()

N_RUNS = 20
steps = ['ema_filter', 'phi_encode', 'bilinear_fwd', 'E_word_build',
         'logits_matmul', 'CE_fwd', 'backward', 'opt_step', 'total']
times = {k: 0.0 for k in steps}

for run in range(N_RUNS):
    b = batches[run % len(batches)]

    torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    # EMA + filter
    torch.cuda.synchronize(); t0 = time.perf_counter()
    _ = _batch_ema_and_filter(data.sentences[:26], data.rho, data.eps_q, device)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times['ema_filter'] += t1 - t0

    # phi
    torch.cuda.synchronize(); t0 = time.perf_counter()
    phi_n = phi_gpu.encode_normalized(b.flat_Q)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times['phi_encode'] += t1 - t0

    Bi = b.counts.shape[0]
    optimizer.zero_grad(set_to_none=True)

    # bilinear
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                          b.pos_lo, b.counts, Bi, dD, D, S)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times['bilinear_fwd'] += t1 - t0

    # E_word build
    torch.cuda.synchronize(); t0 = time.perf_counter()
    E_word = wp_dense @ E_tgt
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times['E_word_build'] += t1 - t0

    # logits matmul
    torch.cuda.synchronize(); t0 = time.perf_counter()
    logits = ctx @ E_word.t()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times['logits_matmul'] += t1 - t0

    # CE
    torch.cuda.synchronize(); t0 = time.perf_counter()
    loss = F.cross_entropy(logits, b.targets)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times['CE_fwd'] += t1 - t0

    # backward
    torch.cuda.synchronize(); t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times['backward'] += t1 - t0

    # optimizer
    torch.cuda.synchronize(); t0 = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize(); t1 = time.perf_counter()
    times['opt_step'] += t1 - t0

    torch.cuda.synchronize()
    times['total'] += time.perf_counter() - t_total_start

# Print
b = batches[0]
print(f"═══ Batch Profile ({N_RUNS} runs avg) ═══")
print(f"Batch: {b.counts.shape[0]} pos, {b.flat_slots.shape[0]} active")
print(f"E_word: ({V}, {dD}), logits: ({b.counts.shape[0]}, {V})")
print()

total_ms = times['total'] / N_RUNS * 1000
print(f"{'Step':<20s} {'ms':>8s}  {'%':>5s}")
print("─" * 40)
for k in steps:
    ms = times[k] / N_RUNS * 1000
    pct = ms / total_ms * 100
    bar = '█' * int(pct / 2)
    print(f"  {k:<18s} {ms:8.3f}  {pct:5.1f}  {bar}")

fwd = sum(times[k] for k in steps[:6]) / N_RUNS * 1000
bwd = times['backward'] / N_RUNS * 1000
opt = times['opt_step'] / N_RUNS * 1000
print(f"\n  forward:  {fwd:.1f}ms")
print(f"  backward: {bwd:.1f}ms")
print(f"  opt:      {opt:.1f}ms")
print(f"  epoch est: {total_ms * 53:.0f}ms")
