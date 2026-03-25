"""Streaming 预处理器 — 一次预处理 + 流式训练

设计:
  初始化: CPU 多核处理全部句子 → 存 (slots, Q_values, targets, offsets) 到 CPU 内存
  训练:   每 epoch 只 shuffle 索引 → 按 batch 切片 → async H2D → GPU phi+train

Q_values 比 phi 小 4x (N=8 vs D=32), 1000 句 ~1.8GB, 可接受。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from .config import MathBrainConfig
from .retina import HashRetina, IdentityRetina, BPERetina
from .data import tokenize_auto


# =====================================================================
# VocabInfo
# =====================================================================

@dataclass
class VocabInfo:
    word_to_idx: Dict[str, int]
    vocab_list: List[str]
    slot_universe: np.ndarray
    slot_to_compact: Dict[int, int]
    S: int
    V: int
    wp_word_offsets: np.ndarray
    wp_slot_indices: np.ndarray
    wp_slot_weights: np.ndarray


def build_vocab(corpus: List[str], retina, *,
                verbose=True) -> VocabInfo:
    t0 = time.time()
    word_to_slots: Dict[str, np.ndarray] = {}
    all_slots = set()
    for s in corpus:
        for w in tokenize_auto(s):
            if w not in word_to_slots:
                slots = retina.get_slots(w)
                word_to_slots[w] = slots
                all_slots.update(int(si) for si in slots)

    slot_universe = np.array(sorted(all_slots), dtype=np.int32)
    slot_to_compact = {int(si): i for i, si in enumerate(slot_universe)}
    S = len(slot_universe)
    vocab_list = sorted(word_to_slots.keys())
    V = len(vocab_list)
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}

    wp_offsets, wp_slots, wp_weights = [0], [], []
    for w in vocab_list:
        compact = [slot_to_compact[int(si)]
                   for si in word_to_slots[w] if int(si) in slot_to_compact]
        if compact:
            v = 1.0 / len(compact)
            wp_slots.extend(compact)
            wp_weights.extend([v] * len(compact))
        wp_offsets.append(len(wp_slots))

    if verbose:
        print(f"  vocab: S={S}, V={V}, {time.time()-t0:.2f}s")

    return VocabInfo(
        word_to_idx=word_to_idx, vocab_list=vocab_list,
        slot_universe=slot_universe, slot_to_compact=slot_to_compact,
        S=S, V=V,
        wp_word_offsets=np.array(wp_offsets, dtype=np.int64),
        wp_slot_indices=np.array(wp_slots, dtype=np.int64),
        wp_slot_weights=np.array(wp_weights, dtype=np.float32),
    )


# =====================================================================
# Worker
# =====================================================================

_w_retina = None
_w_rho = None
_w_N = None
_w_eps_q = None
_w_stc = None
_w_wti = None


def _worker_init(config_dict, slot_to_compact, word_to_idx):
    global _w_retina, _w_rho, _w_N, _w_eps_q, _w_stc, _w_wti
    cfg = MathBrainConfig(**config_dict)
    if cfg.RETINA_MODE == 'identity':
        _w_retina = IdentityRetina(cfg)
    elif cfg.RETINA_MODE == 'bpe':
        _w_retina = BPERetina(cfg)
    else:
        _w_retina = HashRetina(cfg)
    _w_rho = cfg.rho
    _w_N = cfg.N
    _w_eps_q = float(cfg.EPS_Q)
    _w_stc = slot_to_compact
    _w_wti = word_to_idx


def _worker_process(sentences):
    retina, rho, N, eps_q = _w_retina, _w_rho, _w_N, _w_eps_q
    stc, wti = _w_stc, _w_wti

    slots_parts, Q_parts, targets, counts = [], [], [], []

    for s in sentences:
        words = tokenize_auto(s)
        if len(words) < 2:
            continue
        encoded = [retina.encode(w) for w in words]
        all_s = set()
        for enc in encoded[:-1]:
            all_s.update(enc.keys())
        if not all_s:
            continue

        slot_list = sorted(all_s)
        s2d = {si: i for i, si in enumerate(slot_list)}
        ns = len(slot_list)
        T = len(words) - 1

        x = np.zeros((T, ns), dtype=np.float32)
        for t in range(T):
            for sid, cnt in encoded[t].items():
                x[t, s2d[sid]] = float(cnt)

        Q = np.zeros((T, ns, N), dtype=np.float32)
        Q[0] = x[0, :, np.newaxis]
        rr = rho[np.newaxis, :]
        for t in range(1, T):
            Q[t] = Q[t-1] * rr + x[t, :, np.newaxis]

        alive = np.max(np.abs(Q), axis=2) >= eps_q
        sarr = np.array(slot_list, dtype=np.int32)

        for t in range(T):
            at = alive[t]
            if not np.any(at):
                continue
            idx = np.where(at)[0]
            slots_parts.append(np.array([stc[int(sarr[i])] for i in idx], np.int64))
            Q_parts.append(Q[t, idx])
            targets.append(wti.get(words[t+1], 0))
            counts.append(len(idx))

    if not counts:
        return (np.empty(0, np.int64), np.empty((0, N), np.float32),
                np.empty(0, np.int64), np.array([0], np.int64))

    off = np.zeros(len(counts)+1, np.int64)
    off[1:] = np.cumsum(counts)
    return (np.concatenate(slots_parts), np.concatenate(Q_parts),
            np.array(targets, np.int64), off)


# =====================================================================
# PreprocessedData — 一次预处理, 常驻 CPU 内存
# =====================================================================

@dataclass
class PreprocessedData:
    """全部句子预处理后的 compact 数据 (CPU 内存)"""
    flat_slots: np.ndarray    # (total_active,) int64
    flat_Q: np.ndarray        # (total_active, N) float32
    targets: np.ndarray       # (n_pos,) int64
    pos_lo: np.ndarray        # (n_pos,) int64 — 每个 position 在 flat 中的起始
    pos_counts: np.ndarray    # (n_pos,) int64
    n_pos: int
    total_active: int


def preprocess_once(corpus: List[str], vocab: VocabInfo,
                    config: MathBrainConfig, *,
                    num_workers: int = None, verbose=True) -> PreprocessedData:
    """多核处理全部句子 (一次性), 存入 CPU 内存"""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    config_dict = {
        'N': int(config.N), 'RHO': tuple(config.RHO),
        'D_PHI': int(config.D_PHI), 'CP_RANK': int(config.CP_RANK),
        'NGRAM_SCALES': config.NGRAM_SCALES,
        'K': int(config.K), 'EPS_Q': float(config.EPS_Q),
        'THETA_Q': float(config.THETA_Q),
        'PHI_SIGMA': float(config.PHI_SIGMA),
        'CHAOS_N_FOLDS': int(config.CHAOS_N_FOLDS),
        'CHAOS_ALPHA': float(config.CHAOS_ALPHA),
        'RETINA_MODE': config.RETINA_MODE,
    }

    t0 = time.time()

    n_w = min(num_workers, len(corpus))
    chunk_size = max(1, (len(corpus) + n_w - 1) // n_w)
    chunks = [corpus[i:i+chunk_size]
              for i in range(0, len(corpus), chunk_size)]

    with mp.Pool(processes=n_w, initializer=_worker_init,
                 initargs=(config_dict, vocab.slot_to_compact,
                           vocab.word_to_idx)) as pool:
        results = pool.map(_worker_process, chunks)

    # 合并
    all_s, all_q, all_t = [], [], []
    pos_starts = []
    cursor = 0
    for fs, fq, tgt, off in results:
        if len(tgt) == 0:
            continue
        all_s.append(fs)
        all_q.append(fq)
        all_t.append(tgt)
        pos_starts.append(off[:-1] + cursor)
        cursor += int(off[-1])

    flat_slots = np.concatenate(all_s)
    flat_Q = np.concatenate(all_q)
    targets = np.concatenate(all_t)
    pos_lo = np.concatenate(pos_starts)
    n_pos = len(targets)

    # pos_counts from pos_lo
    pos_counts = np.empty(n_pos, np.int64)
    pos_counts[:-1] = pos_lo[1:] - pos_lo[:-1]
    pos_counts[-1] = cursor - pos_lo[-1]

    elapsed = time.time() - t0
    mem_mb = (flat_slots.nbytes + flat_Q.nbytes + targets.nbytes +
              pos_lo.nbytes + pos_counts.nbytes) / 1e6

    if verbose:
        avg = cursor / max(n_pos, 1)
        print(f"  {n_pos:,} positions, {cursor:,} active (avg {avg:.0f}), "
              f"{mem_mb:.1f}MB CPU, {elapsed:.2f}s ({n_w} workers)")

    return PreprocessedData(
        flat_slots=flat_slots, flat_Q=flat_Q, targets=targets,
        pos_lo=pos_lo, pos_counts=pos_counts,
        n_pos=n_pos, total_active=cursor,
    )


# =====================================================================
# RawBatch + batch 迭代
# =====================================================================

@dataclass
class RawBatch:
    flat_slots: torch.Tensor    # (total,) int64 — GPU
    flat_Q: torch.Tensor        # (total, N) float32 — GPU
    targets: torch.Tensor       # (B,) int64 — GPU
    pos_lo: torch.Tensor        # (B,) int64 — GPU
    counts: torch.Tensor        # (B,) int64 — GPU


def _gather_into_pinned(data, bidx, np_views):
    """CPU: numpy gather → write into pinned buffer's numpy views (releases GIL)

    np_views = (slots_np, Q_np, tgt_np, lo_np, cnt_np) — numpy views of pinned tensors
    Returns: (total_active, B)
    """
    B = len(bidx)
    b_lo = data.pos_lo[bidx]
    b_cnt = data.pos_counts[bidx]
    max_cnt = int(b_cnt.max())
    rng = np.arange(max_cnt, dtype=np.int64)
    idx_2d = rng[np.newaxis, :] + b_lo[:, np.newaxis]
    mask = rng[np.newaxis, :] < b_cnt[:, np.newaxis]
    valid = idx_2d[mask]
    total = len(valid)

    bpl = np.zeros(B, dtype=np.int64)
    bpl[1:] = np.cumsum(b_cnt[:-1])

    # 纯 numpy 写入 pinned buffer (释放 GIL!)
    np_views[0][:total] = data.flat_slots[valid]
    np_views[1][:total] = data.flat_Q[valid]
    np_views[2][:B] = data.targets[bidx]
    np_views[3][:B] = bpl
    np_views[4][:B] = b_cnt.astype(np.int64)

    return total, B


def iter_batches_prefetch(data: PreprocessedData, batch_size: int,
                          device: torch.device, shuffle=True):
    """预分配 pinned buffer + 纯 numpy 后台写入 (释放 GIL) + async DMA

    后台线程: numpy gather → numpy 写 pinned buf (全程释放 GIL)
    主线程:   CUDA train (也释放 GIL) → 真正并行!
    """
    import threading

    n_pos = data.n_pos
    N = data.flat_Q.shape[1]
    perm = np.random.permutation(n_pos) if shuffle else np.arange(n_pos)

    batch_indices = []
    for b_start in range(0, n_pos, batch_size):
        b_end = min(b_start + batch_size, n_pos)
        batch_indices.append(perm[b_start:b_end])

    n_batches = len(batch_indices)
    if n_batches == 0:
        return

    # 估算最大 batch 的 active 数
    max_active = 0
    for bidx in batch_indices:
        cnt_sum = int(data.pos_counts[bidx].sum())
        if cnt_sum > max_active:
            max_active = cnt_sum
    max_B = batch_size

    # 预分配 2 组 pinned buffers + numpy views
    def alloc_bufs():
        t_slots = torch.empty(max_active, dtype=torch.long, pin_memory=True)
        t_Q = torch.empty(max_active, N, dtype=torch.float32, pin_memory=True)
        t_tgt = torch.empty(max_B, dtype=torch.long, pin_memory=True)
        t_lo = torch.empty(max_B, dtype=torch.long, pin_memory=True)
        t_cnt = torch.empty(max_B, dtype=torch.long, pin_memory=True)
        tensors = (t_slots, t_Q, t_tgt, t_lo, t_cnt)
        np_views = (t_slots.numpy(), t_Q.numpy(), t_tgt.numpy(),
                    t_lo.numpy(), t_cnt.numpy())
        return tensors, np_views

    tens_A, np_A = alloc_bufs()
    tens_B, np_B = alloc_bufs()

    copy_stream = torch.cuda.Stream(device=device)

    # 准备 batch 0 → buf A
    total_0, B_0 = _gather_into_pinned(data, batch_indices[0], np_A)
    with torch.cuda.stream(copy_stream):
        cur_gpu = RawBatch(
            flat_slots=tens_A[0][:total_0].to(device, non_blocking=True),
            flat_Q=tens_A[1][:total_0].to(device, non_blocking=True),
            targets=tens_A[2][:B_0].to(device, non_blocking=True),
            pos_lo=tens_A[3][:B_0].to(device, non_blocking=True),
            counts=tens_A[4][:B_0].to(device, non_blocking=True),
        )

    # 双缓冲状态
    next_sizes = [0, 0]
    bg_thread = None

    def bg_gather(idx, nv):
        next_sizes[0], next_sizes[1] = _gather_into_pinned(
            data, batch_indices[idx], nv)

    try:
        for i in range(n_batches):
            # 交替使用 A/B buffer
            next_tens = tens_B if i % 2 == 0 else tens_A
            next_np = np_B if i % 2 == 0 else np_A

            # 后台线程: gather + numpy 写 pinned buf (释放 GIL)
            if i + 1 < n_batches:
                bg_thread = threading.Thread(target=bg_gather,
                                             args=(i + 1, next_np))
                bg_thread.start()

            # 等 DMA 完成 + 让 default stream 等 copy_stream (event sync)
            copy_stream.synchronize()
            copy_event = copy_stream.record_event()
            torch.cuda.current_stream(device).wait_event(copy_event)

            # yield 给训练 (GPU default stream)
            yield cur_gpu

            # 等后台完成, 启动下一 batch 的 async DMA
            if i + 1 < n_batches:
                bg_thread.join()
                bg_thread = None
                total_n, B_n = next_sizes
                with torch.cuda.stream(copy_stream):
                    cur_gpu = RawBatch(
                        flat_slots=next_tens[0][:total_n].to(device, non_blocking=True),
                        flat_Q=next_tens[1][:total_n].to(device, non_blocking=True),
                        targets=next_tens[2][:B_n].to(device, non_blocking=True),
                        pos_lo=next_tens[3][:B_n].to(device, non_blocking=True),
                        counts=next_tens[4][:B_n].to(device, non_blocking=True),
                    )
    finally:
        # 清理: 如果 generator 被 break/close(), 等后台线程结束
        if bg_thread is not None:
            bg_thread.join()


# fallback
def iter_batches(data: PreprocessedData, batch_size: int,
                 device: torch.device, shuffle=True):
    n_pos = data.n_pos
    perm = np.random.permutation(n_pos) if shuffle else np.arange(n_pos)
    for b_start in range(0, n_pos, batch_size):
        b_end = min(b_start + batch_size, n_pos)
        bidx = perm[b_start:b_end]
        B = len(bidx)
        b_lo = data.pos_lo[bidx]
        b_cnt = data.pos_counts[bidx]
        max_cnt = int(b_cnt.max())
        rng = np.arange(max_cnt, dtype=np.int64)
        idx_2d = rng[np.newaxis, :] + b_lo[:, np.newaxis]
        mask = rng[np.newaxis, :] < b_cnt[:, np.newaxis]
        valid = idx_2d[mask]
        bpl = np.zeros(B, dtype=np.int64)
        bpl[1:] = np.cumsum(b_cnt[:-1])
        yield RawBatch(
            flat_slots=torch.from_numpy(data.flat_slots[valid].copy()).to(device),
            flat_Q=torch.from_numpy(data.flat_Q[valid].copy()).to(device),
            targets=torch.from_numpy(data.targets[bidx].copy()).to(device),
            pos_lo=torch.from_numpy(bpl).to(device),
            counts=torch.from_numpy(b_cnt.copy().astype(np.int64)).to(device),
        )


