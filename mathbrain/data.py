"""MathBrain 数据管道 — 高效预处理 + CSR 零填充 DataLoader

核心设计:
  1. batch phi: 全句 Q_values concat 后一次性 encode
  2. CSR 紧凑存储: 变长 active — 零 padding
  3. flat collate: batch 内所有 active 打平, 用 batch_idx 做 scatter_add
     → 完全消除 padding, GPU 算力 = 精确 active 数
  4. sparse word_proj: COO → CUDA sparse_mm; MPS/CPU → gather+mean
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import MathBrainConfig
from .retina import HashRetina
from .phi_encoder import CosineChaosEncoder


# =====================================================================
# 工具
# =====================================================================

def tokenize(sentence: str) -> List[str]:
    return [
        w.strip()
        for w in sentence.lower()
            .replace('.', ' .').replace('?', ' ?').replace(',', ' ,')
            .split()
        if w.strip()
    ]


# =====================================================================
# 预处理
# =====================================================================

def _process_sentence(words, retina, phi_encoder, rho, N, eps_q):
    """处理单句 → List[(active_slots, phi_hat, gold_word)]
    全句一次性 batch encode, 无 per-position 循环调 encode。
    """
    if len(words) < 2:
        return []

    encoded = [retina.encode(w) for w in words]

    all_slots_set: set = set()
    for enc in encoded[:-1]:
        all_slots_set.update(enc.keys())
    if not all_slots_set:
        return []

    slot_list = sorted(all_slots_set)
    slot_to_dense = {s: i for i, s in enumerate(slot_list)}
    n_slots = len(slot_list)
    T = len(words) - 1

    # dense 输入
    x = np.zeros((T, n_slots), dtype=np.float32)
    for t in range(T):
        for sid, cnt in encoded[t].items():
            x[t, slot_to_dense[sid]] = float(cnt)

    # vectorized EMA
    Q = np.zeros((T, n_slots, N), dtype=np.float32)
    Q[0] = x[0, :, np.newaxis]
    rho_row = rho[np.newaxis, :]
    for t in range(1, T):
        Q[t] = Q[t - 1] * rho_row + x[t, :, np.newaxis]

    # alive mask
    alive_matrix = np.max(np.abs(Q), axis=2) >= eps_q

    q_parts, slot_parts, pos_infos = [], [], []
    slot_arr = np.array(slot_list, dtype=np.int32)
    cursor = 0

    for t in range(T):
        alive_t = alive_matrix[t]
        if not np.any(alive_t):
            continue
        idx = np.where(alive_t)[0]
        n_a = len(idx)
        q_parts.append(Q[t, idx])
        slot_parts.append(slot_arr[idx])
        pos_infos.append((cursor, n_a, words[t + 1]))
        cursor += n_a

    if not q_parts:
        return []

    # batch encode — 核心: 一次 matmul 全句
    Q_all = np.concatenate(q_parts, axis=0)
    phi_all = phi_encoder.encode_normalized(Q_all)

    positions = []
    for i, (start, n_a, gw) in enumerate(pos_infos):
        positions.append((slot_parts[i], phi_all[start:start + n_a], gw))
    return positions


# =====================================================================
# 预处理结果 (CSR)
# =====================================================================

@dataclass
class PreprocessResult:
    offsets: np.ndarray       # (n_pos+1,) int64
    flat_active: np.ndarray   # (total_active,) int64 — compact slot
    flat_phi: np.ndarray      # (total_active, D) float32
    targets: np.ndarray       # (n_pos,) int64 — target word idx
    n_pos: int
    S: int; V: int; D: int
    slot_universe: np.ndarray
    slot_to_compact: Dict[int, int]
    word_to_idx: Dict[str, int]
    vocab_list: List[str]
    word_to_slots: Dict[str, np.ndarray]
    # word→slot 映射 (for gather-based word_proj)
    wp_word_offsets: np.ndarray   # (V+1,) int64
    wp_slot_indices: np.ndarray   # (nnz,) int64
    wp_slot_weights: np.ndarray   # (nnz,) float32


def _worker_process_sentences(args):
    """Multiprocessing worker: 处理一组句子, 返回 positions + vocab."""
    sentences, config_dict = args
    cfg = MathBrainConfig(**config_dict)
    retina = HashRetina(cfg)
    phi_enc = CosineChaosEncoder(cfg)
    rho = cfg.rho
    N = cfg.N
    eps_q = float(cfg.EPS_Q)

    local_vocab = {}
    positions = []
    for s in sentences:
        words = tokenize(s)
        for w in words:
            if w not in local_vocab:
                local_vocab[w] = retina.get_slots(w)
        positions.extend(
            _process_sentence(words, retina, phi_enc, rho, N, eps_q))
    return positions, local_vocab


def preprocess_corpus(corpus, retina, phi_encoder, config, *,
                      verbose=True, num_workers=None):
    import multiprocessing as mp

    t0 = time.time()

    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(corpus))

    # 拆分语料到各 worker
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

    chunk_size = max(1, (len(corpus) + num_workers - 1) // num_workers)
    chunks = [corpus[i:i + chunk_size]
              for i in range(0, len(corpus), chunk_size)]
    worker_args = [(c, config_dict) for c in chunks]

    if num_workers <= 1:
        results = [_worker_process_sentences(a) for a in worker_args]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(_worker_process_sentences, worker_args))

    # 合并结果
    word_to_slots = {}
    all_positions = []
    for positions, local_vocab in results:
        all_positions.extend(positions)
        word_to_slots.update(local_vocab)

    if not all_positions:
        raise ValueError("No valid positions")

    n_pos = len(all_positions)

    # slot universe
    all_slots = set()
    for act, _, _ in all_positions:
        all_slots.update(int(s) for s in act)
    for sl in word_to_slots.values():
        all_slots.update(int(s) for s in sl)

    slot_universe = np.array(sorted(all_slots), dtype=np.int32)
    slot_to_compact = {int(s): i for i, s in enumerate(slot_universe)}
    S = len(slot_universe)

    vocab_list = sorted(word_to_slots.keys())
    V = len(vocab_list)
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}

    # CSR
    offsets = np.zeros(n_pos + 1, dtype=np.int64)
    for i, (act, _, _) in enumerate(all_positions):
        offsets[i + 1] = offsets[i] + len(act)

    total = int(offsets[-1])
    D = all_positions[0][1].shape[1]
    flat_active = np.empty(total, dtype=np.int64)
    flat_phi = np.empty((total, D), dtype=np.float32)
    targets = np.empty(n_pos, dtype=np.int64)

    for i, (act, phi, gw) in enumerate(all_positions):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        flat_active[lo:hi] = [slot_to_compact[int(s)] for s in act]
        flat_phi[lo:hi] = phi
        targets[i] = word_to_idx.get(gw, 0)

    # word→slot gather 映射 (替代 sparse matmul)
    wp_offsets = [0]
    wp_slots = []
    wp_weights = []
    for wi, w in enumerate(vocab_list):
        if w in word_to_slots:
            slots = word_to_slots[w]
            compact = [slot_to_compact[int(s)]
                       for s in slots if int(s) in slot_to_compact]
            if compact:
                v = 1.0 / len(compact)
                wp_slots.extend(compact)
                wp_weights.extend([v] * len(compact))
        wp_offsets.append(len(wp_slots))

    elapsed = time.time() - t0
    if verbose:
        avg = total / n_pos
        mx = max(int(offsets[i+1]-offsets[i]) for i in range(n_pos))
        print(f"  {n_pos:,} pos, avg_active={avg:.1f}, max={mx}, "
              f"S={S}, V={V}, D={D}, {elapsed:.2f}s")

    return PreprocessResult(
        offsets=offsets, flat_active=flat_active, flat_phi=flat_phi,
        targets=targets, n_pos=n_pos, S=S, V=V, D=D,
        slot_universe=slot_universe, slot_to_compact=slot_to_compact,
        word_to_idx=word_to_idx, vocab_list=vocab_list,
        word_to_slots=word_to_slots,
        wp_word_offsets=np.array(wp_offsets, dtype=np.int64),
        wp_slot_indices=np.array(wp_slots, dtype=np.int64),
        wp_slot_weights=np.array(wp_weights, dtype=np.float32),
    )


# =====================================================================
# Dataset + Flat Collate (零填充!)
# =====================================================================

class WakeMapDataset(Dataset):
    def __init__(self, result: PreprocessResult):
        self.offsets = result.offsets
        self.flat_active = result.flat_active
        self.flat_phi = result.flat_phi
        self.targets = result.targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        lo = int(self.offsets[idx])
        hi = int(self.offsets[idx + 1])
        return (
            self.flat_active[lo:hi].copy(),
            self.flat_phi[lo:hi].copy(),
            int(self.targets[idx]),
        )


def collate_flat(batch):
    """零填充 collate: 打平所有 active, 用 batch_idx 索引

    Returns:
        flat_active: (total,) int64
        flat_phi:    (total, D) float32
        batch_idx:   (total,) int64 — 每个 active 属于哪个 sample
        counts:      (B,) int64 — 每个 sample 有多少 active
        targets:     (B,) int64
    """
    actives, phis, tgts = [], [], []
    batch_indices = []
    counts = []

    for i, (act, phi, tgt) in enumerate(batch):
        n = len(act)
        actives.append(torch.from_numpy(act))
        phis.append(torch.from_numpy(phi))
        batch_indices.append(torch.full((n,), i, dtype=torch.long))
        counts.append(n)
        tgts.append(tgt)

    return (
        torch.cat(actives),                              # (total,)
        torch.cat(phis),                                 # (total, D)
        torch.cat(batch_indices),                        # (total,)
        torch.tensor(counts, dtype=torch.long),          # (B,)
        torch.tensor(tgts, dtype=torch.long),            # (B,)
    )


def build_dataloader(result, *, batch_size=512, shuffle=True,
                     num_workers=0, pin_memory=False):
    dataset = WakeMapDataset(result)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_flat, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False)


class WordProjection:
    """word_proj: slot_scores (B, S) → word_logits (B, V)

    使用 gather + scatter_add 替代矩阵乘。
    word_idx 预计算, forward 零循环。
    """

    def __init__(self, result: PreprocessResult, device: torch.device):
        self.indices = torch.from_numpy(result.wp_slot_indices).to(device)
        self.weights = torch.from_numpy(result.wp_slot_weights).to(device)
        self.V = result.V
        self.device = device

        # 预计算 word_idx: 每个 nnz 元素属于哪个 word
        offsets = result.wp_word_offsets
        lengths = np.diff(offsets)  # (V,)
        self._word_idx = torch.from_numpy(
            np.repeat(np.arange(result.V, dtype=np.int64), lengths)
        ).to(device)

    def _precompute_word_idx(self):
        """已在 __init__ 中完成, 此方法为兼容保留"""
        pass

    def forward(self, slot_scores: torch.Tensor) -> torch.Tensor:
        """slot_scores: (B, S) → (B, V) — 零循环"""
        B = slot_scores.shape[0]
        gathered = slot_scores[:, self.indices]           # (B, nnz)
        weighted = gathered * self.weights.unsqueeze(0)   # (B, nnz)

        out = torch.zeros(B, self.V, dtype=slot_scores.dtype,
                          device=self.device)
        out.scatter_add_(1, self._word_idx.unsqueeze(0).expand(B, -1),
                         weighted)
        return out
