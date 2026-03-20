"""GPU 预处理器 v2 — 动态 per-batch EMA

Pipeline:
  CPU workers: 句子 → retina hash → (x_indicator, slot_indices, targets) per sentence
  存储: per-sentence 数据在 GPU (~12MB for 1000 sentences, vs 1.44GB flat_Q)
  训练: shuffle 句子 → batch 句子 → GPU EMA parallel scan (~1ms) → alive filter → phi → train

vs v1 (precomputed):
  - v1: precompute flat_Q (45M×8=1.44GB GPU 常驻) + 17s 预处理
  - v2: 存 x_indicators (~12MB GPU) + 2s 预处理 + 1ms/batch 动态 EMA
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import triton
import triton.language as tl

from .config import MathBrainConfig
from .retina import HashRetina
from .data import tokenize


# ═══════════════════════════════════════════════════════════════
# Triton EMA Parallel Scan (复用)
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _combine(a_l, b_l, a_r, b_r):
    return a_l * a_r, a_r * b_l + b_r


@triton.jit
def ema_scan_kernel(
    X_ptr, Q_ptr, RHO_ptr,
    stride_xb, stride_xt, stride_xs,
    stride_qb, stride_qt, stride_qs, stride_qn,
    T, S, N,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid % N
    pid_sn = pid // N
    s = pid_sn % S
    b = pid_sn // S

    rho = tl.load(RHO_ptr + n)
    t_offs = tl.arange(0, BLOCK_T)
    mask = t_offs < T

    x_vals = tl.load(X_ptr + b * stride_xb + t_offs * stride_xt + s * stride_xs,
                     mask=mask, other=0.0)

    a_vals = tl.where(mask, rho, 1.0)
    _, q_vals = tl.associative_scan((a_vals, x_vals), 0, _combine)

    tl.store(Q_ptr + b * stride_qb + t_offs * stride_qt + s * stride_qs + n * stride_qn,
             q_vals, mask=mask)


def gpu_ema(x_gpu: torch.Tensor, rho_gpu: torch.Tensor,
            chunk_size: int = 4096) -> torch.Tensor:
    """GPU EMA via Triton parallel scan

    x_gpu: (B, T, S) float32
    rho_gpu: (N,) float32
    Returns: Q (B, T, S, N) float32
    """
    B, T, S = x_gpu.shape
    N = rho_gpu.shape[0]
    Q = torch.empty(B, T, S, N, device=x_gpu.device, dtype=torch.float32)

    if T <= chunk_size:
        BLOCK_T = triton.next_power_of_2(T)
        grid = (B * S * N,)
        ema_scan_kernel[grid](
            x_gpu, Q, rho_gpu,
            x_gpu.stride(0), x_gpu.stride(1), x_gpu.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            T, S, N, BLOCK_T=BLOCK_T,
        )
    else:
        carry = torch.zeros(B, S, N, device=x_gpu.device, dtype=torch.float32)
        BLOCK_T = triton.next_power_of_2(chunk_size)
        grid = (B * S * N,)

        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            chunk_len = t_end - t_start
            x_chunk = x_gpu[:, t_start:t_end].contiguous()
            Q_chunk = torch.empty(B, chunk_len, S, N,
                                  device=x_gpu.device, dtype=torch.float32)
            BT = triton.next_power_of_2(chunk_len)
            ema_scan_kernel[grid](
                x_chunk, Q_chunk, rho_gpu,
                x_chunk.stride(0), x_chunk.stride(1), x_chunk.stride(2),
                Q_chunk.stride(0), Q_chunk.stride(1),
                Q_chunk.stride(2), Q_chunk.stride(3),
                chunk_len, S, N, BLOCK_T=BT,
            )
            if t_start > 0:
                t_indices = torch.arange(1, chunk_len + 1, device=x_gpu.device,
                                         dtype=torch.float32)
                rho_powers = rho_gpu.unsqueeze(0).pow(t_indices.unsqueeze(1))
                Q_chunk += carry.unsqueeze(1) * rho_powers.unsqueeze(0).unsqueeze(2)
            Q[:, t_start:t_end] = Q_chunk
            carry = Q[:, t_end - 1, :, :].clone()

    return Q


# ═══════════════════════════════════════════════════════════════
# CPU Worker: Retina Hash Only
# ═══════════════════════════════════════════════════════════════

_w_retina = None
_w_stc = None
_w_wti = None


def _worker_init(config_dict, slot_to_compact, word_to_idx):
    global _w_retina, _w_stc, _w_wti
    cfg = MathBrainConfig(**config_dict)
    _w_retina = HashRetina(cfg)
    _w_stc = slot_to_compact
    _w_wti = word_to_idx


def _worker_hash(sentences):
    """CPU worker: retina hash only → per sentence data"""
    retina, stc, wti = _w_retina, _w_stc, _w_wti
    results = []

    for s in sentences:
        words = tokenize(s)
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
                if sid in s2d:
                    x[t, s2d[sid]] = float(cnt)

        compact_slots = np.array([stc[si] for si in slot_list], dtype=np.int64)
        targets = np.array([wti.get(words[t+1], 0) for t in range(T)], dtype=np.int64)

        results.append((x, compact_slots, targets))

    return results


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class SentenceGPU:
    """单句的 GPU 数据 — 不含 EMA (动态计算)"""
    x_indicator: torch.Tensor   # (T, S_local) float32
    slot_indices: torch.Tensor  # (S_local,) int64
    targets: torch.Tensor       # (T,) int64
    n_pos: int                  # T
    n_slots: int                # S_local


@dataclass
class GPUData:
    """全数据 — per-sentence, 不存 flat_Q"""
    sentences: List[SentenceGPU]
    rho: torch.Tensor           # (N,) float32
    eps_q: float
    N: int
    total_pos: int


@dataclass
class RawBatch:
    flat_slots: torch.Tensor    # (total_active,) int64
    flat_Q: torch.Tensor        # (total_active, N) float32
    targets: torch.Tensor       # (n_pos,) int64
    pos_lo: torch.Tensor        # (n_pos,) int64
    counts: torch.Tensor        # (n_pos,) int64


# ═══════════════════════════════════════════════════════════════
# GPU Preprocess (简化: 只存 hash, 不做 EMA)
# ═══════════════════════════════════════════════════════════════

def gpu_preprocess(corpus: List[str], vocab, config: MathBrainConfig,
                   device: torch.device, *, num_workers=None,
                   verbose=True) -> GPUData:
    """CPU hash → 存到 GPU (不做 EMA)"""
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
    }

    # CPU retina hash
    t0 = time.time()
    n_w = min(num_workers, len(corpus))
    chunk_size = max(1, (len(corpus) + n_w - 1) // n_w)
    chunks = [corpus[i:i+chunk_size] for i in range(0, len(corpus), chunk_size)]

    with mp.Pool(processes=n_w, initializer=_worker_init,
                 initargs=(config_dict, vocab.slot_to_compact,
                           vocab.word_to_idx)) as pool:
        worker_results = pool.map(_worker_hash, chunks)

    all_sentences_raw = []
    for chunk_result in worker_results:
        all_sentences_raw.extend(chunk_result)

    cpu_time = time.time() - t0

    # Move to GPU as per-sentence structures
    t1 = time.time()
    sentences = []
    total_pos = 0
    total_bytes = 0

    for x, slots, tgt in all_sentences_raw:
        sg = SentenceGPU(
            x_indicator=torch.from_numpy(x).to(device),
            slot_indices=torch.from_numpy(slots).to(device),
            targets=torch.from_numpy(tgt).to(device),
            n_pos=x.shape[0],
            n_slots=x.shape[1],
        )
        sentences.append(sg)
        total_pos += sg.n_pos
        total_bytes += x.nbytes + slots.nbytes + tgt.nbytes

    rho_gpu = torch.tensor(config.rho, dtype=torch.float32, device=device)
    gpu_time = time.time() - t1

    if verbose:
        print(f"  CPU hash: {len(sentences)} sentences, "
              f"{total_bytes/1e6:.1f}MB, {cpu_time:.2f}s ({n_w} workers)")
        print(f"  GPU upload: {total_bytes/1e6:.1f}MB, {gpu_time:.2f}s")
        print(f"  Total positions: {total_pos:,}")

    return GPUData(
        sentences=sentences,
        rho=rho_gpu,
        eps_q=float(config.EPS_Q),
        N=config.N,
        total_pos=total_pos,
    )


# ═══════════════════════════════════════════════════════════════
# Dynamic Batch Iterator: shuffle 句子, per-batch GPU EMA
# ═══════════════════════════════════════════════════════════════

def _batch_ema_and_filter(sents: List[SentenceGPU],
                          rho: torch.Tensor, eps_q: float,
                          device: torch.device) -> RawBatch:
    """全 batch 向量化: pad → EMA → alive → flatten, ~10 kernel launches total"""
    N = rho.shape[0]
    K = len(sents)
    max_T = max(s.n_pos for s in sents)
    max_S = max(s.n_slots for s in sents)

    # ── Pad 全部到 (K, max_T, max_S) ──
    x_padded = torch.zeros(K, max_T, max_S, device=device, dtype=torch.float32)
    slots_padded = torch.zeros(K, max_S, device=device, dtype=torch.long)
    tgt_padded = torch.zeros(K, max_T, device=device, dtype=torch.long)
    valid_mask = torch.zeros(K, max_T, max_S, device=device, dtype=torch.bool)

    for j, s in enumerate(sents):
        x_padded[j, :s.n_pos, :s.n_slots] = s.x_indicator
        slots_padded[j, :s.n_slots] = s.slot_indices
        tgt_padded[j, :s.n_pos] = s.targets
        valid_mask[j, :s.n_pos, :s.n_slots] = True

    # ── GPU EMA (1 call) ──
    Q = gpu_ema(x_padded, rho)  # (K, max_T, max_S, N)

    # ── Batched alive filter (NO per-sentence loop!) ──
    alive = Q.abs().amax(dim=3) >= eps_q   # (K, max_T, max_S)
    alive = alive & valid_mask              # 排除 padding 区域

    # 1 次 nonzero: 获取所有 alive (k, t, s) 坐标
    k_idx, t_idx, s_idx = alive.nonzero(as_tuple=True)

    if len(k_idx) == 0:
        empty_s = torch.empty(0, dtype=torch.long, device=device)
        empty_q = torch.empty(0, N, dtype=torch.float32, device=device)
        return RawBatch(empty_s, empty_q, empty_s, empty_s, empty_s)

    # 1 次 gather: 获取 flat_slots 和 flat_Q
    flat_slots = slots_padded[k_idx, s_idx]
    flat_Q = Q[k_idx, t_idx, s_idx, :]     # (n_alive, N)

    # Per-position counts: 先算 (K, max_T) 的 alive 计数
    counts_full = alive.sum(dim=2)           # (K, max_T)
    pos_valid = counts_full > 0              # (K, max_T)
    k_pos, t_pos = pos_valid.nonzero(as_tuple=True)

    targets = tgt_padded[k_pos, t_pos]
    counts = counts_full[k_pos, t_pos]

    n_pos = counts.shape[0]
    pos_lo = torch.zeros(n_pos, dtype=torch.long, device=device)
    if n_pos > 1:
        pos_lo[1:] = counts[:-1].cumsum(0)

    return RawBatch(
        flat_slots=flat_slots,
        flat_Q=flat_Q,
        targets=targets,
        pos_lo=pos_lo,
        counts=counts,
    )


def iter_batches_gpu(data: GPUData, batch_size: int,
                     device: torch.device, shuffle=True):
    """动态 EMA batch iterator — per-batch GPU EMA + batched alive filter"""
    n_sents = len(data.sentences)
    if shuffle:
        perm = torch.randperm(n_sents).tolist()
    else:
        perm = list(range(n_sents))

    i = 0
    while i < n_sents:
        batch_sents = []
        batch_pos = 0
        while i < n_sents and batch_pos < batch_size:
            s = data.sentences[perm[i]]
            batch_sents.append(s)
            batch_pos += s.n_pos
            i += 1

        if not batch_sents:
            continue

        batch = _batch_ema_and_filter(batch_sents, data.rho, data.eps_q, device)
        if batch.counts.shape[0] == 0:
            continue

        yield batch

