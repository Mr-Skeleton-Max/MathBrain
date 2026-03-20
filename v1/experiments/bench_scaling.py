#!/usr/bin/env python3
"""CPU 多核扩展性测试: 1,2,4,8,16,32 核对比

用法:
  python experiments/bench_scaling.py \
      --corpus datasets/tinystories_2000.txt \
      --rho 0.3,0.75,0.93,0.98,0.995,0.999,0.9995,0.9999 \
      --D 32 --cp-rank 32
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.wake_dataset import WakeDataset
from mathbrain.data_cache import _process_one_sentence_vectorized


def parse_rho(s):
    return np.array([float(x) for x in s.split(',')], dtype=np.float64)


def _worker(args):
    sentences, config_tuple, worker_id = args
    N, RHO, D_PHI, CP_RANK, K, NGRAM_SCALES, phi_mode = config_tuple

    from mathbrain.config import MathBrainConfig
    from mathbrain.retina import HashRetinaV2
    from mathbrain.phi_encoder import create_phi_encoder

    cfg = MathBrainConfig(N=N, RHO=RHO, D_PHI=D_PHI, CP_RANK=CP_RANK,
                          K=K, NGRAM_SCALES=NGRAM_SCALES)
    retina = HashRetinaV2(cfg)
    phi_enc = create_phi_encoder(cfg)
    rho = cfg.rho.astype(np.float32)
    eps_q = float(cfg.EPS_Q)

    t0 = time.time()
    pid = os.getpid()
    count = 0
    for s in sentences:
        words = WakeDataset._tokenize(s)
        positions = _process_one_sentence_vectorized(
            words, retina, phi_enc, rho, N, eps_q, True, phi_mode)
        count += len(positions)
    elapsed = time.time() - t0
    return count, worker_id, pid, t0, time.time()


def bench_n_workers(corpus, config_tuple, n_workers):
    chunk_size = max(1, (len(corpus) + n_workers - 1) // n_workers)
    chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)]
    args = [(c, config_tuple, i) for i, c in enumerate(chunks)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_worker, args))
    elapsed = time.time() - t0

    total = sum(r[0] for r in results)
    return total / elapsed, total, elapsed, results


def main():
    parser = argparse.ArgumentParser(description='CPU 多核扩展性测试')
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--rho', type=parse_rho,
                        default='0.3,0.75,0.93,0.98')
    parser.add_argument('--D', type=int, default=8)
    parser.add_argument('--cp-rank', type=int, default=384)
    args = parser.parse_args()

    with open(args.corpus) as f:
        corpus = [l.strip() for l in f if l.strip()]

    rho = args.rho
    N = len(rho)
    cfg = MathBrainConfig(N=N, RHO=tuple(rho), D_PHI=args.D, CP_RANK=args.cp_rank)
    model = MathBrain(cfg)
    for s in corpus:
        for w in WakeDataset._tokenize(s):
            model._ensure_word(w)

    config_tuple = (
        int(cfg.N), tuple(cfg.RHO), int(cfg.D_PHI), int(cfg.CP_RANK),
        int(cfg.K), tuple(cfg.NGRAM_SCALES), 'chaos'
    )

    max_cores = mp.cpu_count()
    test_cores = [n for n in [1, 2, 4, 8, 16, 32] if n <= max_cores]

    print(f"Corpus: {args.corpus}, {len(corpus)} sentences")
    print(f"Available CPU cores: {max_cores}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}")
    print()

    results_table = []
    base_speed = None

    for n in test_cores:
        print(f"Testing {n:2d} cores...", end=" ", flush=True)
        speed, total, elapsed, worker_results = bench_n_workers(
            corpus, config_tuple, n)

        if base_speed is None:
            base_speed = speed
        speedup = speed / base_speed

        # Check overlap
        starts = [r[3] for r in worker_results]
        ends = [r[4] for r in worker_results]
        overlap = max(0, min(ends) - max(min(starts), min(starts)))
        concurrent = "✅" if (min(ends) - max(starts)) > 0.5 * (max(ends) - min(starts)) else "⚠️"

        results_table.append((n, speed, elapsed, speedup, concurrent))
        print(f"{speed:>8,.0f} pos/sec  {elapsed:5.1f}s  "
              f"speedup={speedup:.1f}x  {concurrent}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Cores':>6} {'pos/sec':>10} {'Time':>7} {'Speedup':>8} {'Status'}")
    print(f"{'-'*6:>6} {'-'*10:>10} {'-'*7:>7} {'-'*8:>8} {'-'*6}")
    for n, speed, elapsed, speedup, status in results_table:
        print(f"{n:>6} {speed:>10,.0f} {elapsed:>6.1f}s {speedup:>7.1f}x {status}")
    print(f"{'='*60}")

    if len(results_table) >= 2:
        last = results_table[-1]
        first = results_table[0]
        efficiency = (last[3] / last[0]) * 100  # speedup / cores * 100
        print(f"\n并行效率: {last[0]} 核 → {last[3]:.1f}x speedup = "
              f"{efficiency:.0f}% efficiency")


if __name__ == '__main__':
    main()
