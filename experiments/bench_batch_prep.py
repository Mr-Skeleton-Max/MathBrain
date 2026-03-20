#!/usr/bin/env python3
"""微基准: 逐步测量 batch 准备的每个环节"""
import sys, time
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MathBrain.config import MathBrainConfig
from MathBrain.retina import HashRetina
from MathBrain.streaming_preprocessor import build_vocab, preprocess_once

def main():
    device = torch.device('cuda')
    cfg = MathBrainConfig(
        N=8, RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
        D_PHI=32, CP_RANK=32)

    corpus = [l.strip() for l in open('datasets/tinystories_1000.txt') if l.strip()]
    retina = HashRetina(cfg)
    vocab = build_vocab(corpus, retina)
    data = preprocess_once(corpus, vocab, cfg)

    batch_size = 4096
    perm = np.random.permutation(data.n_pos)
    bidx = perm[:batch_size]

    # 预热
    for _ in range(3):
        b_lo = data.pos_lo[bidx]
        b_cnt = data.pos_counts[bidx]

    N_RUNS = 20
    print(f"Batch: B={batch_size}, measuring {N_RUNS} runs each\n")

    # ─── [A] numpy index lookup ───
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        b_lo = data.pos_lo[bidx]
        b_cnt = data.pos_counts[bidx]
        b_tgt = data.targets[bidx]
    t_index = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[A] numpy index (pos_lo, cnt, tgt):     {t_index:.2f}ms")

    # ─── [B] 2D mask + fancy index ───
    b_lo = data.pos_lo[bidx]
    b_cnt = data.pos_counts[bidx]
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        max_cnt = int(b_cnt.max())
        rng = np.arange(max_cnt, dtype=np.int64)
        idx_2d = rng[np.newaxis, :] + b_lo[:, np.newaxis]
        mask = rng[np.newaxis, :] < b_cnt[:, np.newaxis]
        valid = idx_2d[mask]
    t_mask = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[B] 2D mask + idx_2d[mask]:              {t_mask:.2f}ms")

    valid = idx_2d[mask]
    print(f"    valid.shape={valid.shape}, max_cnt={max_cnt}")

    # ─── [C] flat_slots[valid] gather ───
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        b_slots = data.flat_slots[valid]
    t_gather_s = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[C] flat_slots[valid]:                   {t_gather_s:.2f}ms")

    # ─── [D] flat_Q[valid] gather ───
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        b_Q = data.flat_Q[valid]
    t_gather_q = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[D] flat_Q[valid]:                       {t_gather_q:.2f}ms")

    # ─── [E] .copy() ───
    b_slots = data.flat_slots[valid]
    b_Q = data.flat_Q[valid]
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        s_c = b_slots.copy()
        q_c = b_Q.copy()
    t_copy = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[E] .copy() (slots + Q):                 {t_copy:.2f}ms")

    # ─── [F] torch.from_numpy (zero-copy) ───
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        ts = torch.from_numpy(b_slots)
        tq = torch.from_numpy(b_Q)
    t_fromnp = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[F] torch.from_numpy:                    {t_fromnp:.2f}ms")

    # ─── [G] pin_memory() ───
    ts = torch.from_numpy(b_slots.copy())
    tq = torch.from_numpy(b_Q.copy())
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        ts_p = ts.pin_memory()
        tq_p = tq.pin_memory()
    t_pin = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[G] pin_memory() (slots + Q):            {t_pin:.2f}ms")

    # ─── [H] .to(device, non_blocking) from pinned ───
    ts_p = ts.pin_memory()
    tq_p = tq.pin_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        ts_g = ts_p.to(device, non_blocking=True)
        tq_g = tq_p.to(device, non_blocking=True)
        torch.cuda.synchronize()
    t_h2d_pin = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[H] .to(cuda) from pinned (sync):        {t_h2d_pin:.2f}ms")

    # ─── [I] .to(device) from pageable (no pin) ───
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        ts_g = ts.to(device)
        tq_g = tq.to(device)
        torch.cuda.synchronize()
    t_h2d_page = (time.perf_counter() - t0) / N_RUNS * 1000
    print(f"[I] .to(cuda) from pageable (sync):      {t_h2d_page:.2f}ms")

    # ─── Summary ───
    total_pin = t_index + t_mask + t_gather_s + t_gather_q + t_copy + t_fromnp + t_pin + t_h2d_pin
    total_nopin = t_index + t_mask + t_gather_s + t_gather_q + t_fromnp + t_h2d_page
    print(f"\n{'='*60}")
    print(f"Total with pin_memory:     {total_pin:.1f}ms")
    print(f"Total without pin_memory:  {total_nopin:.1f}ms")
    print(f"GPU train per batch:       ~24ms")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
