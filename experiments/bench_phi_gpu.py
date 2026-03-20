#!/usr/bin/env python3
"""Benchmark: GPU vs CPU phi encoding + CPU retina/EMA 开销

测量:
  [A] CPU retina.encode (hash lookup) — 每 word
  [B] CPU EMA Q 更新 — 每 position
  [C] CPU np.cos phi — 当前瓶颈
  [D] GPU torch.cos phi — 提议方案
  [E] GPU phi + Triton forward 完整链路

用法: python experiments/bench_phi_gpu.py --corpus datasets/tinystories_200.txt
"""
import sys, time, argparse
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MathBrain.config import MathBrainConfig
from MathBrain.retina import HashRetina
from MathBrain.phi_encoder import CosineChaosEncoder
from MathBrain.gpu_phi_encoder import GPUPhiEncoder
from MathBrain.data import tokenize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='datasets/tinystories_200.txt')
    args = parser.parse_args()

    device = torch.device('cuda')
    cfg = MathBrainConfig(
        N=8, RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
        D_PHI=32, CP_RANK=32)

    corpus = [l.strip() for l in open(args.corpus) if l.strip()]
    retina = HashRetina(cfg)
    phi_cpu = CosineChaosEncoder(cfg)
    phi_gpu = GPUPhiEncoder(cfg, device='cuda')
    rho = cfg.rho
    N = cfg.N
    eps_q = float(cfg.EPS_Q)

    # ── [A] Benchmark retina.encode ──
    all_words = []
    for s in corpus:
        all_words.extend(tokenize(s))
    n_words = len(all_words)

    t0 = time.time()
    for w in all_words:
        retina.encode(w)
    t_retina = time.time() - t0
    print(f"[A] retina.encode: {n_words} words in {t_retina*1000:.1f}ms "
          f"({n_words/t_retina:.0f} words/sec)")

    # ── [B] Benchmark EMA Q update (vectorized) ──
    # Simulate: process all sentences, collect Q_values
    all_Q = []
    all_slots_list = []
    all_targets_list = []
    n_positions = 0

    t0 = time.time()
    for s in corpus:
        words = tokenize(s)
        if len(words) < 2:
            continue
        encoded = [retina.encode(w) for w in words]
        all_slots_set = set()
        for enc in encoded[:-1]:
            all_slots_set.update(enc.keys())
        if not all_slots_set:
            continue
        slot_list = sorted(all_slots_set)
        slot_to_dense = {s: i for i, s in enumerate(slot_list)}
        n_slots = len(slot_list)
        T = len(words) - 1

        x = np.zeros((T, n_slots), dtype=np.float32)
        for t in range(T):
            for sid, cnt in encoded[t].items():
                x[t, slot_to_dense[sid]] = float(cnt)

        Q = np.zeros((T, n_slots, N), dtype=np.float32)
        Q[0] = x[0, :, np.newaxis]
        rho_row = rho[np.newaxis, :]
        for t in range(1, T):
            Q[t] = Q[t - 1] * rho_row + x[t, :, np.newaxis]

        alive_matrix = np.max(np.abs(Q), axis=2) >= eps_q
        slot_arr = np.array(slot_list, dtype=np.int32)

        for t in range(T):
            alive_t = alive_matrix[t]
            if not np.any(alive_t):
                continue
            idx = np.where(alive_t)[0]
            all_Q.append(Q[t, idx])
            all_slots_list.append(slot_arr[idx])
            all_targets_list.append(words[t + 1])
            n_positions += 1

    t_ema = time.time() - t0
    print(f"[B] retina+EMA (no phi): {n_positions} positions in {t_ema*1000:.1f}ms "
          f"({n_positions/t_ema:.0f} pos/sec)")

    # Concat all Q_values
    Q_all = np.concatenate(all_Q, axis=0)
    total_active = Q_all.shape[0]
    avg_active = total_active // max(n_positions, 1)
    print(f"    total_active={total_active:,}, avg={avg_active}")

    # ── [C] CPU phi ──
    t0 = time.time()
    phi_result_cpu = phi_cpu.encode_normalized(Q_all)
    t_phi_cpu = time.time() - t0
    print(f"[C] CPU phi (np.cos): {total_active} active in {t_phi_cpu*1000:.1f}ms "
          f"({total_active/t_phi_cpu:.0f} active/sec)")

    # ── [D] GPU phi ──
    Q_gpu = torch.from_numpy(Q_all).to(device)
    # warmup
    for _ in range(3):
        phi_gpu.encode_normalized(Q_gpu)
    torch.cuda.synchronize()

    t0 = time.time()
    N_ITER = 50
    for _ in range(N_ITER):
        phi_result_gpu = phi_gpu.encode_normalized(Q_gpu)
    torch.cuda.synchronize()
    t_phi_gpu = (time.time() - t0) / N_ITER
    print(f"[D] GPU phi (torch.cos): {total_active} active in {t_phi_gpu*1000:.2f}ms "
          f"({total_active/t_phi_gpu:.0f} active/sec)")

    # Correctness
    cpu_t = torch.from_numpy(phi_result_cpu).to(device)
    diff = (cpu_t - phi_result_gpu).abs().max().item()
    print(f"    CPU vs GPU diff: {diff:.6f}")

    # ── [E] Per-batch GPU phi timing ──
    batch_positions = 512  # positions per batch
    batch_active = batch_positions * avg_active
    Q_batch = Q_gpu[:batch_active]

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        phi_gpu.encode_normalized(Q_batch)
    torch.cuda.synchronize()
    t_batch = (time.time() - t0) / N_ITER * 1000
    print(f"\n[E] Per-batch GPU phi (B=512, ~{batch_active} active): {t_batch:.3f}ms")

    n_batches = max(1, (n_positions + batch_positions - 1) // batch_positions)
    print(f"    → {t_batch * n_batches:.1f}ms/epoch ({n_batches} batches)")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY (200 sentences, {n_positions} positions):")
    print(f"  CPU retina+EMA (no phi): {t_ema*1000:.0f}ms")
    print(f"  CPU phi (np.cos):        {t_phi_cpu*1000:.0f}ms")
    print(f"  GPU phi (torch.cos):     {t_phi_gpu*1000:.1f}ms")
    print(f"  Speedup (phi):           {t_phi_cpu/t_phi_gpu:.0f}x")
    print(f"  GPU phi per batch:       {t_batch:.2f}ms")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
