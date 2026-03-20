#!/usr/bin/env python3
"""CPU 供给 vs GPU 消化 速度基准测试

测量:
  1. CPU 端: vectorized EMA + phi 编码 → positions/sec
  2. GPU 端: forward + backward → positions/sec
  3. DataLoader 流式管道端到端吞吐量

用法 (服务器):
  python experiments/bench_throughput.py \
      --corpus datasets/tinystories_2000.txt \
      --rho 0.3,0.75,0.93,0.98,0.995,0.999,0.9995,0.9999 \
      --D 32 --cp-rank 32
"""

import argparse
import os
import sys
import time

# 必须在 import numpy 之前设置！
# fork 的子进程继承已初始化的 BLAS 线程池，之后改 env 无效
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.wake_dataset import WakeDataset
from mathbrain.data_cache import _process_one_sentence_vectorized
from mathbrain.inference_fast import SparseMatrixDecoder


def parse_rho(s):
    return np.array([float(x) for x in s.split(',')], dtype=np.float64)


# ── 1. CPU throughput benchmark ──────────────────────────────────

def bench_cpu(corpus, model, phi_mode='chaos', n_repeat=3):
    """测量 CPU 单核 EMA+phi 吞吐量"""
    retina = model.retina
    phi_encoder = model.phi_encoder
    rho = model.config.rho.astype(np.float32)
    N = model.config.N
    eps_q = float(model.config.EPS_Q)

    # Warm up
    words = WakeDataset._tokenize(corpus[0])
    _process_one_sentence_vectorized(words, retina, phi_encoder, rho, N, eps_q, True, phi_mode)

    total_pos = 0
    t0 = time.time()
    for _ in range(n_repeat):
        for sentence in corpus:
            words = WakeDataset._tokenize(sentence)
            positions = _process_one_sentence_vectorized(
                words, retina, phi_encoder, rho, N, eps_q, True, phi_mode)
            total_pos += len(positions)
    elapsed = time.time() - t0

    pos_per_sec = total_pos / elapsed
    return pos_per_sec, total_pos, elapsed


# Top-level worker function — rebuild objects from config, no pickle of large objects
def _count_positions_chunk(args):
    """Worker: receives (sentences, config_tuple), rebuilds objects locally."""
    import os
    # 关键: 限制 BLAS 线程为 1，避免 N_workers × N_blas 线程争抢
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    sentences, config_tuple = args
    N, RHO, D_PHI, CP_RANK, K, NGRAM_SCALES, phi_mode = config_tuple

    # Rebuild all objects from scratch in this process (cheap)
    from mathbrain.config import MathBrainConfig
    from mathbrain.retina import HashRetinaV2
    from mathbrain.phi_encoder import create_phi_encoder

    cfg = MathBrainConfig(N=N, RHO=RHO, D_PHI=D_PHI, CP_RANK=CP_RANK,
                          K=K, NGRAM_SCALES=NGRAM_SCALES)
    retina = HashRetinaV2(cfg)
    phi_encoder = create_phi_encoder(cfg)
    rho = cfg.rho.astype(np.float32)
    eps_q = float(cfg.EPS_Q)

# Timed worker: returns count + timing info for diagnostic
def _count_positions_chunk_timed(args):
    """Worker with timing diagnostic."""
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    sentences, config_tuple, worker_id = args
    N, RHO, D_PHI, CP_RANK, K, NGRAM_SCALES, phi_mode = config_tuple

    from mathbrain.config import MathBrainConfig
    from mathbrain.retina import HashRetinaV2
    from mathbrain.phi_encoder import create_phi_encoder

    t_start = time.time()
    pid = os.getpid()

    cfg = MathBrainConfig(N=N, RHO=RHO, D_PHI=D_PHI, CP_RANK=CP_RANK,
                          K=K, NGRAM_SCALES=NGRAM_SCALES)
    retina = HashRetinaV2(cfg)
    phi_encoder = create_phi_encoder(cfg)
    rho = cfg.rho.astype(np.float32)
    eps_q = float(cfg.EPS_Q)

    count = 0
    for s in sentences:
        words = WakeDataset._tokenize(s)
        positions = _process_one_sentence_vectorized(
            words, retina, phi_encoder, rho, N, eps_q, True, phi_mode)
        count += len(positions)

    t_end = time.time()
    return count, worker_id, pid, t_start, t_end


def bench_cpu_parallel(corpus, model, n_workers, phi_mode='chaos'):
    """测量 CPU 多核 EMA+phi 吞吐量 — 带并行诊断"""
    from concurrent.futures import ProcessPoolExecutor

    cfg = model.config
    config_tuple = (
        int(cfg.N), tuple(cfg.RHO), int(cfg.D_PHI), int(cfg.CP_RANK),
        int(cfg.K), tuple(cfg.NGRAM_SCALES), phi_mode
    )

    chunk_size = max(1, (len(corpus) + n_workers - 1) // n_workers)
    chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)]

    worker_args = [(chunk, config_tuple, i) for i, chunk in enumerate(chunks)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(_count_positions_chunk_timed, worker_args))
    elapsed = time.time() - t0

    total_pos = sum(r[0] for r in results)
    pos_per_sec = total_pos / elapsed

    # 打印并行诊断
    print(f"  Worker timing diagnostic:")
    base_t = min(r[3] for r in results)
    for count, wid, pid, t_s, t_e in sorted(results, key=lambda r: r[3]):
        bar_start = int((t_s - base_t) * 2)
        bar_len = max(1, int((t_e - t_s) * 2))
        bar = ' ' * bar_start + '█' * bar_len
        print(f"    W{wid:02d} pid={pid:6d} [{t_s-base_t:5.1f}s .. {t_e-base_t:5.1f}s] |{bar}|")

    return pos_per_sec, total_pos, elapsed



# ── 2. GPU throughput benchmark ──────────────────────────────────

def bench_gpu(model, device, S, V, D, d_rank, word_proj_t,
              batch_size=128, max_active=100, n_batches=50):
    """测量 GPU forward+backward 吞吐量 (用随机数据)"""
    from mathbrain.trainer import _SleepModule

    P_init = model.phi_encoder.P
    net = _SleepModule(
        S, d_rank, D, model.config.N, word_proj_t,
        phi_mode='precomputed', learnable_P=False, P_init=P_init
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)

    M = max_active
    # Generate random data on GPU
    active = torch.randint(0, S, (batch_size, M), device=device)
    phi = torch.randn(batch_size, M, D, device=device)
    mask = torch.ones(batch_size, M, device=device)
    target = torch.randint(0, V, (batch_size,), device=device)

    # Warm up
    for _ in range(5):
        loss = net(active, phi, mask, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    total_pos = 0
    for _ in range(n_batches):
        loss = net(active, phi, mask, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_pos += batch_size

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    pos_per_sec = total_pos / elapsed
    return pos_per_sec, total_pos, elapsed


# ── 3. E2E streaming benchmark ──────────────────────────────────

class _BenchStreamDataset(IterableDataset):
    """IterableDataset 模拟: 实时 EMA+phi, yield 单个 position"""
    def __init__(self, corpus, retina, phi_encoder, rho, N, eps_q,
                 phi_mode, slot_to_compact, word_to_idx):
        self.corpus = corpus
        self.retina = retina
        self.phi_encoder = phi_encoder
        self.rho = rho
        self.N = N
        self.eps_q = eps_q
        self.phi_mode = phi_mode
        self.s2c = slot_to_compact
        self.w2i = word_to_idx

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        corpus = self.corpus
        if worker_info is not None:
            n = len(corpus)
            per_worker = (n + worker_info.num_workers - 1) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = min(start + per_worker, n)
            corpus = corpus[start:end]

        for sentence in corpus:
            words = WakeDataset._tokenize(sentence)
            positions = _process_one_sentence_vectorized(
                words, self.retina, self.phi_encoder,
                self.rho, self.N, self.eps_q, True, self.phi_mode)
            for active_slots, feature, gold_word in positions:
                # Map to compact ids
                local_ids = np.array(
                    [self.s2c[int(s)] for s in active_slots if int(s) in self.s2c],
                    dtype=np.int64)
                if len(local_ids) == 0:
                    continue
                feat = feature[:len(local_ids)]
                yield local_ids, feat, self.w2i.get(gold_word, 0)


def _collate_fn(batch):
    max_M = max(ids.shape[0] for ids, _, _ in batch)
    B = len(batch)
    D = batch[0][1].shape[1]

    active = torch.zeros(B, max_M, dtype=torch.long)
    phi = torch.zeros(B, max_M, D, dtype=torch.float32)
    mask = torch.zeros(B, max_M, dtype=torch.float32)
    targets = []

    for i, (ids, feat, tgt) in enumerate(batch):
        n = len(ids)
        active[i, :n] = torch.from_numpy(ids)
        phi[i, :n] = torch.from_numpy(feat)
        mask[i, :n] = 1.0
        targets.append(tgt)

    target = torch.tensor(targets, dtype=torch.long)
    return active, phi, mask, target


def bench_e2e_streaming(corpus, model, device, S, V, D, d_rank,
                        word_proj_t, slot_to_compact, word_to_idx,
                        batch_size=128, num_workers=4):
    """端到端: DataLoader streaming → GPU train"""
    from mathbrain.trainer import _SleepModule

    P_init = model.phi_encoder.P
    net = _SleepModule(
        S, d_rank, D, model.config.N, word_proj_t,
        phi_mode='precomputed', learnable_P=False, P_init=P_init
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)

    retina = model.retina
    phi_encoder = model.phi_encoder
    rho = model.config.rho.astype(np.float32)

    dataset = _BenchStreamDataset(
        corpus, retina, phi_encoder, rho, model.config.N,
        float(model.config.EPS_Q), 'chaos', slot_to_compact, word_to_idx)

    use_pin = (device.type == 'cuda')
    actual_workers = num_workers if device.type == 'cuda' else 0

    loader = DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=actual_workers,
        pin_memory=use_pin,
        prefetch_factor=2 if actual_workers > 0 else None,
    )

    # Warm up
    warmup_iter = iter(loader)
    for _ in range(3):
        try:
            ba, bp, bm, bt = next(warmup_iter)
            ba, bp, bm, bt = (ba.to(device), bp.to(device),
                              bm.to(device), bt.to(device))
            loss = net(ba, bp, bm, bt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except StopIteration:
            break

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed run
    t0 = time.time()
    total_pos = 0
    n_batches = 0
    wait_times = []

    for ba, bp, bm, bt in loader:
        t_transfer = time.time()
        nb = use_pin
        ba = ba.to(device, non_blocking=nb)
        bp = bp.to(device, non_blocking=nb)
        bm = bm.to(device, non_blocking=nb)
        bt = bt.to(device, non_blocking=nb)
        wait_times.append(time.time() - t_transfer)

        loss = net(ba, bp, bm, bt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_pos += ba.shape[0]
        n_batches += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    pos_per_sec = total_pos / elapsed if elapsed > 0 else 0
    avg_wait = np.mean(wait_times) * 1000 if wait_times else 0

    return pos_per_sec, total_pos, elapsed, n_batches, avg_wait


# ── 4. E2E prebatch streaming benchmark ─────────────────────────

class _BenchStreamPrebatchDataset(IterableDataset):
    """IterableDataset: worker 内部组装完整 batch tensor, yield batch."""
    def __init__(self, corpus, retina, phi_encoder, rho, N, eps_q,
                 phi_mode, slot_to_compact, word_to_idx, batch_size):
        self.corpus = corpus
        self.retina = retina
        self.phi_encoder = phi_encoder
        self.rho = rho
        self.N = N
        self.eps_q = eps_q
        self.phi_mode = phi_mode
        self.s2c = slot_to_compact
        self.w2i = word_to_idx
        self.batch_size = batch_size

    def _collate_buffer(self, buffer):
        """在 worker 进程内完成 collate → tensor"""
        max_M = max(ids.shape[0] for ids, _, _ in buffer)
        B = len(buffer)
        D = buffer[0][1].shape[1]

        active = torch.zeros(B, max_M, dtype=torch.long)
        phi = torch.zeros(B, max_M, D, dtype=torch.float32)
        mask = torch.zeros(B, max_M, dtype=torch.float32)
        targets = []

        for i, (ids, feat, tgt) in enumerate(buffer):
            n = len(ids)
            active[i, :n] = torch.from_numpy(ids)
            phi[i, :n] = torch.from_numpy(feat)
            mask[i, :n] = 1.0
            targets.append(tgt)

        target = torch.tensor(targets, dtype=torch.long)
        return active, phi, mask, target

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        corpus = self.corpus
        if worker_info is not None:
            n = len(corpus)
            per_worker = (n + worker_info.num_workers - 1) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = min(start + per_worker, n)
            corpus = corpus[start:end]

        buffer = []
        for sentence in corpus:
            words = WakeDataset._tokenize(sentence)
            positions = _process_one_sentence_vectorized(
                words, self.retina, self.phi_encoder,
                self.rho, self.N, self.eps_q, True, self.phi_mode)
            for active_slots, feature, gold_word in positions:
                local_ids = np.array(
                    [self.s2c[int(s)] for s in active_slots if int(s) in self.s2c],
                    dtype=np.int64)
                if len(local_ids) == 0:
                    continue
                feat = feature[:len(local_ids)]
                buffer.append((local_ids, feat, self.w2i.get(gold_word, 0)))

                if len(buffer) >= self.batch_size:
                    yield self._collate_buffer(buffer)
                    buffer = []

        if buffer:
            yield self._collate_buffer(buffer)


def bench_e2e_prebatch(corpus, model, device, S, V, D, d_rank,
                       word_proj_t, slot_to_compact, word_to_idx,
                       batch_size=128, num_workers=4):
    """端到端: worker 内部 prebatch → GPU train"""
    from mathbrain.trainer import _SleepModule

    P_init = model.phi_encoder.P
    net = _SleepModule(
        S, d_rank, D, model.config.N, word_proj_t,
        phi_mode='precomputed', learnable_P=False, P_init=P_init
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)

    retina = model.retina
    phi_encoder = model.phi_encoder
    rho = model.config.rho.astype(np.float32)

    dataset = _BenchStreamPrebatchDataset(
        corpus, retina, phi_encoder, rho, model.config.N,
        float(model.config.EPS_Q), 'chaos', slot_to_compact, word_to_idx,
        batch_size)

    use_pin = (device.type == 'cuda')
    actual_workers = num_workers if device.type == 'cuda' else 0

    loader = DataLoader(
        dataset,
        batch_size=None,          # 已经是 batch 了!
        num_workers=actual_workers,
        pin_memory=use_pin,
        prefetch_factor=2 if actual_workers > 0 else None,
    )

    # Warm up
    warmup_iter = iter(loader)
    for _ in range(3):
        try:
            ba, bp, bm, bt = next(warmup_iter)
            ba, bp, bm, bt = (ba.to(device), bp.to(device),
                              bm.to(device), bt.to(device))
            loss = net(ba, bp, bm, bt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except StopIteration:
            break

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed run
    t0 = time.time()
    total_pos = 0
    n_batches = 0
    wait_times = []

    for ba, bp, bm, bt in loader:
        t_transfer = time.time()
        nb = use_pin
        ba = ba.to(device, non_blocking=nb)
        bp = bp.to(device, non_blocking=nb)
        bm = bm.to(device, non_blocking=nb)
        bt = bt.to(device, non_blocking=nb)
        wait_times.append(time.time() - t_transfer)

        loss = net(ba, bp, bm, bt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_pos += ba.shape[0]
        n_batches += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    pos_per_sec = total_pos / elapsed if elapsed > 0 else 0
    avg_wait = np.mean(wait_times) * 1000 if wait_times else 0

    return pos_per_sec, total_pos, elapsed, n_batches, avg_wait


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CPU vs GPU 吞吐量基准测试')
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--rho', type=parse_rho, default='0.3,0.75,0.93,0.98')
    parser.add_argument('--D', type=int, default=8)
    parser.add_argument('--cp-rank', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    with open(args.corpus) as f:
        corpus = [l.strip() for l in f if l.strip()]
    print(f"Corpus: {args.corpus}, {len(corpus)} sentences")
    print(f"Device: {device}")

    rho = args.rho
    N = len(rho)
    D = args.D

    cfg = MathBrainConfig(N=N, RHO=tuple(rho), D_PHI=D, CP_RANK=args.cp_rank)
    model = MathBrain(cfg)

    # Register all words
    for s in corpus:
        for w in WakeDataset._tokenize(s):
            model._ensure_word(w)

    # Build slot universe (prescan)
    print("\n[1/4] Pre-scan: slot universe + vocab...")
    t0 = time.time()
    all_slots = set()
    for s in corpus:
        words = WakeDataset._tokenize(s)
        for w in words:
            enc = model.retina.encode(w)
            all_slots.update(enc.keys())
    for word in model.vocab:
        if word in model.word_to_slots:
            all_slots.update(int(s) for s in model.word_to_slots[word])

    slot_universe = sorted(all_slots)
    slot_to_compact = {s: i for i, s in enumerate(slot_universe)}
    S = len(slot_universe)

    decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, cfg.K)
    vocab_list = decoder.word_list
    V = len(vocab_list)
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}

    word_proj = np.zeros((V, S), dtype=np.float32)
    for wi, w in enumerate(vocab_list):
        if w in model.word_to_slots:
            slots = model.word_to_slots[w]
            compact = [slot_to_compact[int(sl)] for sl in slots if int(sl) in slot_to_compact]
            if compact:
                word_proj[wi, compact] = 1.0 / len(compact)
    word_proj_t = torch.from_numpy(word_proj.T).float().to(device)

    print(f"  S={S}, V={V}, prescan={time.time()-t0:.1f}s")
    d_rank = cfg.CP_RANK

    # ── Benchmark 1: CPU multi-core ──
    import multiprocessing as mp
    n_cpu = min(mp.cpu_count(), args.num_workers)
    print(f"\n[2/4] CPU {n_cpu}-core throughput...")
    try:
        par_speed, par_total, par_time = bench_cpu_parallel(corpus, model, n_cpu)
        print(f"  → {par_speed:,.0f} positions/sec  ({par_total:,} pos in {par_time:.1f}s)")
    except Exception as e:
        print(f"  parallel failed: {e}")
        par_speed = 10000

    # ── Benchmark 3: GPU pure throughput ──
    print(f"\n[4/4] GPU forward+backward throughput (random data)...")
    gpu_speed, gpu_total, gpu_time = bench_gpu(
        model, device, S, V, D, d_rank, word_proj_t,
        batch_size=args.batch_size, max_active=100, n_batches=100)
    print(f"  → {gpu_speed:,.0f} positions/sec  ({gpu_total:,} pos in {gpu_time:.1f}s)")

    # ── Verdict ──
    print(f"\n{'='*60}")
    print(f"VERDICT:")
    print(f"  CPU ({n_cpu} cores): {par_speed:>10,.0f} pos/sec")
    print(f"  GPU (pure):         {gpu_speed:>10,.0f} pos/sec")
    ratio = par_speed / gpu_speed if gpu_speed > 0 else float('inf')
    if ratio >= 1.0:
        print(f"  ✅ CPU {ratio:.1f}x faster — streaming 可行!")
    else:
        print(f"  ⚠️  GPU {1/ratio:.1f}x faster — CPU 可能成为瓶颈")
        print(f"      需要 {int(np.ceil(1/ratio * n_cpu))} 个 CPU worker 才能匹配 GPU")
    print(f"{'='*60}")

    # ── Bonus A: E2E streaming (per-position yield) ──
    print(f"\n[Bonus A] E2E per-position streaming (1 epoch)...")
    try:
        e2e_speed, e2e_total, e2e_time, n_batches, avg_wait = bench_e2e_streaming(
            corpus, model, device, S, V, D, d_rank,
            word_proj_t, slot_to_compact, word_to_idx,
            batch_size=args.batch_size,
            num_workers=n_cpu if device.type == 'cuda' else 0)
        print(f"  → {e2e_speed:,.0f} positions/sec  ({e2e_total:,} pos, "
              f"{n_batches} batches, {e2e_time:.1f}s)")
        print(f"  avg transfer wait: {avg_wait:.2f}ms/batch")
        print(f"  E2E vs GPU-pure: {e2e_speed/gpu_speed*100:.0f}% GPU utilization")
    except Exception as e:
        import traceback
        print(f"  E2E per-position failed: {e}")
        traceback.print_exc()

    # ── Bonus B: E2E prebatch streaming (worker yields full batch tensors) ──
    print(f"\n[Bonus B] E2E prebatch streaming (1 epoch)...")
    try:
        pb_speed, pb_total, pb_time, pb_batches, pb_wait = bench_e2e_prebatch(
            corpus, model, device, S, V, D, d_rank,
            word_proj_t, slot_to_compact, word_to_idx,
            batch_size=args.batch_size,
            num_workers=n_cpu if device.type == 'cuda' else 0)
        print(f"  → {pb_speed:,.0f} positions/sec  ({pb_total:,} pos, "
              f"{pb_batches} batches, {pb_time:.1f}s)")
        print(f"  avg transfer wait: {pb_wait:.2f}ms/batch")
        print(f"  E2E vs GPU-pure: {pb_speed/gpu_speed*100:.0f}% GPU utilization")
    except Exception as e:
        import traceback
        print(f"  E2E prebatch failed: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()

