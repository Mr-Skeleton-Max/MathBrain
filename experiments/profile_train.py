#!/usr/bin/env python3
"""精确测量训练循环每一步的时间: 找出 32ms/batch 的瓶颈"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MathBrain.config import MathBrainConfig
from MathBrain.retina import HashRetina
from MathBrain.streaming_preprocessor import build_vocab
from MathBrain.gpu_preprocessor import gpu_preprocess, iter_batches_gpu
from MathBrain.gpu_phi_encoder import GPUPhiEncoder
from MathBrain.triton_kernels import triton_bilinear, triton_word_proj


def main():
    device = torch.device('cuda')
    cfg = MathBrainConfig(
        N=8, RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
        D_PHI=32, CP_RANK=32)

    corpus = [l.strip() for l in open('datasets/tinystories_1000.txt') if l.strip()]
    retina = HashRetina(cfg)
    vocab = build_vocab(corpus, retina, verbose=False)
    data = gpu_preprocess(corpus, vocab, cfg, device, verbose=False)

    phi_gpu = GPUPhiEncoder(cfg, device='cuda')
    S, V, D = vocab.S, vocab.V, cfg.D_PHI
    dD = cfg.CP_RANK * D

    E_src = torch.nn.Embedding(S, dD).to(device)
    E_tgt = torch.nn.Parameter(torch.randn(S, dD, device=device) * 0.01)
    wp_off = torch.from_numpy(vocab.wp_word_offsets).to(device)
    wp_idx = torch.from_numpy(vocab.wp_slot_indices).to(device)
    wp_wt = torch.from_numpy(vocab.wp_slot_weights).to(device)
    optimizer = torch.optim.AdamW([E_src.weight, E_tgt], lr=0.01)

    # Warmup
    for b in iter_batches_gpu(data, 4096, device, shuffle=False):
        phi_n = phi_gpu.encode_normalized(b.flat_Q)
        Bi = b.counts.shape[0]
        optimizer.zero_grad(set_to_none=True)
        ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                              b.pos_lo, b.counts, Bi, dD, D, S)
        scores = ctx @ E_tgt.t()
        logits = triton_word_proj(scores, wp_off, wp_idx, wp_wt, Bi, S, V)
        loss = F.cross_entropy(logits, b.targets)
        loss.backward()
        optimizer.step()
        break
    torch.cuda.synchronize()

    # ── Profile: measure each step ──
    batch_size = 4096
    N_MEASURE = 3  # epochs to measure

    # Accumulators
    t_iter = []
    t_phi = []
    t_zero = []
    t_bilinear = []
    t_matmul = []
    t_wordproj = []
    t_loss = []
    t_backward = []
    t_step = []

    for epoch in range(N_MEASURE):
        for b in iter_batches_gpu(data, batch_size, device):
            torch.cuda.synchronize(); t0 = time.perf_counter()

            phi_n = phi_gpu.encode_normalized(b.flat_Q)
            torch.cuda.synchronize(); t1 = time.perf_counter()

            Bi = b.counts.shape[0]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t2 = time.perf_counter()

            ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                                  b.pos_lo, b.counts, Bi, dD, D, S)
            torch.cuda.synchronize(); t3 = time.perf_counter()

            scores = ctx @ E_tgt.t()
            torch.cuda.synchronize(); t4 = time.perf_counter()

            logits = triton_word_proj(scores, wp_off, wp_idx, wp_wt, Bi, S, V)
            torch.cuda.synchronize(); t5 = time.perf_counter()

            loss = F.cross_entropy(logits, b.targets)
            torch.cuda.synchronize(); t6 = time.perf_counter()

            loss.backward()
            torch.cuda.synchronize(); t7 = time.perf_counter()

            optimizer.step()
            torch.cuda.synchronize(); t8 = time.perf_counter()

            t_phi.append(t1 - t0)
            t_zero.append(t2 - t1)
            t_bilinear.append(t3 - t2)
            t_matmul.append(t4 - t3)
            t_wordproj.append(t5 - t4)
            t_loss.append(t6 - t5)
            t_backward.append(t7 - t6)
            t_step.append(t8 - t7)

    # Also measure iter_batches_gpu overhead separately
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in iter_batches_gpu(data, batch_size, device):
        pass
    torch.cuda.synchronize()
    t_iter_total = (time.perf_counter() - t0) * 1000

    n = len(t_phi)
    def ms(arr): return np.mean(arr) * 1000

    total = ms(t_phi) + ms(t_zero) + ms(t_bilinear) + ms(t_matmul) + ms(t_wordproj) + ms(t_loss) + ms(t_backward) + ms(t_step)

    print(f"\nPer-batch breakdown ({n} batches, {n//55} epochs):")
    print(f"  phi_encode:     {ms(t_phi):6.2f}ms")
    print(f"  zero_grad:      {ms(t_zero):6.2f}ms")
    print(f"  triton_bilinear:{ms(t_bilinear):6.2f}ms")
    print(f"  ctx @ E_tgt.t():{ms(t_matmul):6.2f}ms")
    print(f"  word_proj:      {ms(t_wordproj):6.2f}ms")
    print(f"  cross_entropy:  {ms(t_loss):6.2f}ms")
    print(f"  backward:       {ms(t_backward):6.2f}ms")
    print(f"  optimizer.step: {ms(t_step):6.2f}ms")
    print(f"  ─────────────────────────")
    print(f"  TOTAL compute:  {total:6.2f}ms/batch")
    print(f"  iter_batches:   {t_iter_total:6.1f}ms/epoch ({t_iter_total/55:.1f}ms/batch)")
    print(f"  Epoch target:   {total * 55:.0f}ms (compute) + {t_iter_total:.0f}ms (iter)")


if __name__ == '__main__':
    main()
