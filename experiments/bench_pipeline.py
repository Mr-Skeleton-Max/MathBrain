#!/usr/bin/env python3
"""MathBrain GPU + CPU Pipeline Benchmark

在服务器上跑: python experiments/bench_pipeline.py --corpus datasets/tinystories_2000.txt

测试内容:
  [1] GPU pure forward+backward (数据全在 GPU, 无 DataLoader)
  [2] GPU per-op breakdown (E_src, mul, sum, matmul, word_proj, CE)
  [3] CPU multi-core phi throughput (1,4,8,16,25 cores)
  [4] 流式供给估算: CPU phi 产能 vs GPU 消耗速度

需要: MathBrain 包 (MathBrain/ 目录)
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MathBrain.config import MathBrainConfig
from MathBrain.retina import HashRetina
from MathBrain.phi_encoder import CosineChaosEncoder
from MathBrain.data import tokenize, _process_sentence, preprocess_corpus, WordProjection
from MathBrain.trainer import BilinearPredictor


def device_sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def parse_rho(s):
    return tuple(float(x) for x in s.split(','))


# =====================================================================
# [1] GPU Pure Forward+Backward
# =====================================================================

def bench_gpu_pure(net, word_proj, device, B, avg_active, S, V, D, n_batches,
                   n_warmup=5, n_iter=50):
    """Data already on GPU, no DataLoader."""
    total_active = B * avg_active
    flat_active = torch.randint(0, S, (total_active,), device=device)
    flat_phi = torch.randn(total_active, D, device=device)
    batch_idx = torch.arange(B, device=device).repeat_interleave(avg_active)
    counts = torch.full((B,), avg_active, dtype=torch.long, device=device)
    targets = torch.randint(0, V, (B,), device=device)

    opt = torch.optim.SGD(net.parameters(), lr=0.01)

    # warmup
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        loss = net(flat_active, flat_phi, batch_idx, counts, targets)
        loss.backward()
        opt.step()
    device_sync(device)

    # benchmark
    t0 = time.time()
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        loss = net(flat_active, flat_phi, batch_idx, counts, targets)
        loss.backward()
        opt.step()
    device_sync(device)
    elapsed = time.time() - t0

    ms_per_batch = elapsed / n_iter * 1000
    s_per_epoch = ms_per_batch * n_batches / 1000
    return ms_per_batch, s_per_epoch


# =====================================================================
# [2] GPU Per-Op Breakdown
# =====================================================================

def bench_gpu_ops(net, word_proj, device, B, avg_active, S, V, D, n_iter=100):
    """Breakdown of each operation on GPU."""
    d = net.d_rank
    total_active = B * avg_active
    flat_active = torch.randint(0, S, (total_active,), device=device)
    flat_phi = torch.randn(total_active, D, device=device)
    batch_idx = torch.arange(B, device=device).repeat_interleave(avg_active)
    counts = torch.full((B,), avg_active, dtype=torch.long, device=device)
    targets = torch.randint(0, V, (B,), device=device)

    # warmup
    for _ in range(10):
        src = net.E_src(flat_active)
    device_sync(device)

    results = {}

    # 1. E_src lookup
    device_sync(device)
    t0 = time.time()
    for _ in range(n_iter):
        src = net.E_src(flat_active)
    device_sync(device)
    results['E_src_lookup'] = (time.time() - t0) / n_iter * 1000

    # 2. phi normalize + elementwise mul
    device_sync(device)
    t0 = time.time()
    for _ in range(n_iter):
        phi_n = flat_phi / flat_phi.norm(dim=1, keepdim=True).clamp(min=1e-8)
        src_3d = src.view(-1, d, D)
        w = (src_3d * phi_n.unsqueeze(1)).view(-1, d * D)
    device_sync(device)
    results['bilinear_mul'] = (time.time() - t0) / n_iter * 1000

    # 3. scatter_add (aggregation)
    device_sync(device)
    t0 = time.time()
    for _ in range(n_iter):
        ctx = torch.zeros(B, d * D, device=device)
        idx_exp = batch_idx.unsqueeze(1).expand(-1, d * D)
        ctx.scatter_add_(0, idx_exp, w)
        ctx /= counts.float().unsqueeze(1).clamp(min=1.0)
    device_sync(device)
    results['scatter_add'] = (time.time() - t0) / n_iter * 1000

    # 4. ctx @ E_tgt.t()
    device_sync(device)
    t0 = time.time()
    for _ in range(n_iter):
        scores = ctx @ net.E_tgt.t()
    device_sync(device)
    results['matmul_E_tgt'] = (time.time() - t0) / n_iter * 1000

    # 5. word_proj
    device_sync(device)
    t0 = time.time()
    for _ in range(n_iter):
        logits = word_proj.forward(scores)
    device_sync(device)
    results['word_proj'] = (time.time() - t0) / n_iter * 1000

    # 6. cross_entropy
    device_sync(device)
    t0 = time.time()
    for _ in range(n_iter):
        loss = F.cross_entropy(logits, targets)
    device_sync(device)
    results['cross_entropy'] = (time.time() - t0) / n_iter * 1000

    return results


# =====================================================================
# [3] CPU Multi-Core Phi Throughput
# =====================================================================

def _phi_worker(args):
    """Worker: 处理一组句子, 返回 position 数和耗时"""
    sentences, config_tuple = args
    N, RHO, D_PHI, CP_RANK, NGRAM_SCALES = config_tuple

    cfg = MathBrainConfig(N=N, RHO=RHO, D_PHI=D_PHI, CP_RANK=CP_RANK,
                          NGRAM_SCALES=NGRAM_SCALES)
    retina = HashRetina(cfg)
    phi_enc = CosineChaosEncoder(cfg)
    rho = cfg.rho
    eps_q = float(cfg.EPS_Q)

    t0 = time.time()
    total_pos = 0
    for s in sentences:
        words = tokenize(s)
        positions = _process_sentence(words, retina, phi_enc, rho, N, eps_q)
        total_pos += len(positions)

    return total_pos, time.time() - t0


def bench_cpu_phi(corpus, config, n_workers_list):
    """Test CPU phi throughput with different core counts."""
    config_tuple = (
        int(config.N), tuple(config.RHO), int(config.D_PHI),
        int(config.CP_RANK), config.NGRAM_SCALES,
    )

    results = []
    for nw in n_workers_list:
        chunk_size = max(1, (len(corpus) + nw - 1) // nw)
        chunks = [corpus[i:i + chunk_size]
                  for i in range(0, len(corpus), chunk_size)]
        worker_args = [(c, config_tuple) for c in chunks]

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=nw) as pool:
            outs = list(pool.map(_phi_worker, worker_args))
        elapsed = time.time() - t0

        total_pos = sum(o[0] for o in outs)
        pos_per_sec = total_pos / elapsed
        results.append((nw, total_pos, elapsed, pos_per_sec))

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='datasets/tinystories_2000.txt')
    parser.add_argument('--rho', type=parse_rho,
                        default='0.3,0.75,0.93,0.98,0.995,0.999,0.9995,0.9999')
    parser.add_argument('--D', type=int, default=32)
    parser.add_argument('--cp-rank', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    corpus = [l.strip() for l in open(args.corpus) if l.strip()]
    N = len(args.rho)
    cfg = MathBrainConfig(N=N, RHO=args.rho, D_PHI=args.D, CP_RANK=args.cp_rank)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Corpus: {args.corpus}, {len(corpus)} sentences")
    print(f"Config: N={N}, D={args.D}, CP_RANK={args.cp_rank}")
    print(f"Device: {device}")
    print()

    # ── Preprocess (get data stats) ──
    print("[0] Preprocess...")
    retina = HashRetina(cfg)
    phi_enc = CosineChaosEncoder(cfg)
    result = preprocess_corpus(corpus, retina, phi_enc, cfg)

    S, V, D = result.S, result.V, result.D
    n_pos = result.n_pos
    avg_active = int(result.offsets[-1]) // n_pos
    n_batches = max(1, (n_pos + args.batch_size - 1) // args.batch_size)

    print(f"  n_pos={n_pos:,}, avg_active={avg_active}, S={S}, V={V}")
    print(f"  n_batches={n_batches} (B={args.batch_size})")
    print()

    # ── Build model on GPU ──
    wp = WordProjection(result, device)
    net = BilinearPredictor(S, V, D, args.cp_rank, wp).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Model params: {n_params:,}")
    print()

    # ── [1] GPU Pure Forward+Backward ──
    print("[1] GPU pure forward+backward (data on device)...")
    ms, epoch_s = bench_gpu_pure(
        net, wp, device, args.batch_size, avg_active, S, V, D, n_batches)
    print(f"  {ms:.2f} ms/batch → {epoch_s:.2f}s/epoch")
    print(f"  {n_pos/epoch_s:,.0f} positions/sec")
    print()

    # ── [1b] GPU Compiled Forward+Backward ──
    if device.type == 'cuda':
        print("[1b] torch.compile forward+backward...")
        torch.set_float32_matmul_precision('high')
        try:
            net_compiled = torch.compile(net, mode='reduce-overhead',
                                         fullgraph=False)
            ms_c, epoch_s_c = bench_gpu_pure(
                net_compiled, wp, device, args.batch_size, avg_active,
                S, V, D, n_batches, n_warmup=10, n_iter=50)
            print(f"  {ms_c:.2f} ms/batch → {epoch_s_c:.2f}s/epoch")
            print(f"  {n_pos/epoch_s_c:,.0f} positions/sec")
            print(f"  speedup vs eager: {ms/ms_c:.2f}x")
        except Exception as e:
            print(f"  torch.compile failed: {e}")
        print()

    # ── [2] GPU Per-Op Breakdown ──
    print(f"[2] GPU per-op breakdown (B={args.batch_size}, "
          f"active={avg_active})...")
    ops = bench_gpu_ops(net, wp, device, args.batch_size, avg_active, S, V, D)
    total_ms = sum(ops.values())
    for name, ms in ops.items():
        pct = ms / total_ms * 100
        print(f"  {name:20s}: {ms:6.2f}ms ({pct:4.1f}%)")
    print(f"  {'TOTAL':20s}: {total_ms:6.2f}ms")
    print(f"  projected forward-only: {total_ms * n_batches / 1000:.2f}s/epoch")
    print()

    # ── [3] CPU Multi-Core Phi ──
    import multiprocessing as mp
    max_cores = min(mp.cpu_count(), 25)
    test_cores = sorted(set(
        [1, 4, 8, 16, max_cores] + [max_cores]))
    test_cores = [c for c in test_cores if c <= max_cores]

    print(f"[3] CPU phi throughput (max {max_cores} cores)...")
    cpu_results = bench_cpu_phi(corpus, cfg, test_cores)
    for nw, total_pos, elapsed, pps in cpu_results:
        print(f"  {nw:2d} cores: {pps:>10,.0f} pos/sec "
              f"({total_pos:,} pos in {elapsed:.1f}s)")
    print()

    # ── [4] Streaming estimate ──
    gpu_consume = n_pos / epoch_s  # pos/sec GPU can consume
    best_cpu = max(r[3] for r in cpu_results)
    best_cores = [r[0] for r in cpu_results if r[3] == best_cpu][0]

    print(f"[4] Streaming estimate:")
    print(f"  GPU consumption: {gpu_consume:>10,.0f} pos/sec")
    print(f"  CPU production:  {best_cpu:>10,.0f} pos/sec ({best_cores} cores)")
    if best_cpu >= gpu_consume:
        print(f"  ✅ CPU 供得上 GPU! 余量 {best_cpu/gpu_consume:.1f}x")
    else:
        needed = int(np.ceil(gpu_consume / (best_cpu / best_cores)))
        print(f"  ⚠️  GPU {gpu_consume/best_cpu:.1f}x faster than CPU")
        print(f"  需要 {needed} 核才能匹配")
    print()

    # ── Summary ──
    print("=" * 60)
    print("SUMMARY:")
    print(f"  GPU forward+backward:  {epoch_s:.2f}s / epoch")
    print(f"  CPU phi ({best_cores} cores):  "
          f"{n_pos/best_cpu:.2f}s / epoch equivalent")
    print(f"  理论最优 epoch time:   "
          f"{max(epoch_s, n_pos/best_cpu):.2f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
