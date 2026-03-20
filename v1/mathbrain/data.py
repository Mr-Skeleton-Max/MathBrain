"""MathBrain 高效数据管道 — 替代 data_cache.py

关键优化:
  1. batch phi: 全句 Q_values concat 后一次性 encode (消除 per-position numpy overhead)
  2. CSR 紧凑存储: 变长 active slots, 零 padding 浪费
  3. per-batch 动态 padding: DataLoader collate_fn 只 pad 到 batch 内最大值
  4. sparse word_proj: 稀疏矩阵乘法

用法:
    from mathbrain.data import preprocess_corpus, load_dataset, collate_wake_batch
    result = preprocess_corpus(corpus, model)
    dataset = load_dataset(result)
    loader = DataLoader(dataset, batch_size=512, collate_fn=collate_wake_batch, shuffle=True)
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import MathBrainConfig
from .wake_dataset import WakeDataset


# =====================================================================
# 1. 预处理: batch phi 计算
# =====================================================================

def _process_sentence_batch_phi(words, retina, phi_encoder, rho, N, eps_q):
    """处理单个句子: vectorized EMA → batch phi (一次性 encode 全句)

    与 data_cache._process_one_sentence_vectorized 的区别:
      - phi_encoder.encode() 只调一次 (全句 Q_values concat)
      - 不逐 position 循环 encode

    Returns:
        List[(active_slot_ids: ndarray, phi_hat: ndarray, gold_word: str)]
    """
    if len(words) < 2:
        return []

    encoded = [retina.encode(w) for w in words]

    # Step 1: 收集所有出现过的 slot, 建立 dense 索引
    all_slots_set = set()
    for enc in encoded[:-1]:
        all_slots_set.update(enc.keys())
    if not all_slots_set:
        return []

    slot_list = sorted(all_slots_set)
    slot_to_dense = {s: i for i, s in enumerate(slot_list)}
    n_slots = len(slot_list)
    T = len(words) - 1

    # Step 2: dense 输入矩阵 x[t, slot]
    x = np.zeros((T, n_slots), dtype=np.float32)
    for t in range(T):
        for slot_id, count in encoded[t].items():
            x[t, slot_to_dense[slot_id]] = float(count)

    # Step 3: vectorized EMA Q[t] = ρ·Q[t-1] + x[t]
    Q = np.zeros((T, n_slots, N), dtype=np.float32)
    Q[0] = x[0, :, np.newaxis]
    rho_row = rho[np.newaxis, :]
    for t in range(1, T):
        Q[t] = Q[t - 1] * rho_row + x[t, :, np.newaxis]

    # Step 4: 提取每个 position 的 alive mask (bulk numpy, 不调 encode)
    # alive[t, s] = max(|Q[t,s,:]|) >= eps_q
    alive_matrix = np.max(np.abs(Q), axis=2) >= eps_q  # (T, n_slots)

    # 收集所有 alive positions 的 Q_values → concat → batch encode
    position_infos = []  # (start_in_flat, n_active, gold_word)
    q_parts = []
    slot_parts = []

    for t in range(T):
        alive_t = alive_matrix[t]
        if not np.any(alive_t):
            continue
        idx = np.where(alive_t)[0]
        active_slots = np.array([slot_list[i] for i in idx], dtype=np.int32)
        q_active = Q[t, idx]  # (n_active, N)

        flat_start = sum(len(p) for p in q_parts)
        q_parts.append(q_active)
        slot_parts.append(active_slots)
        position_infos.append((flat_start, len(idx), words[t + 1]))

    if not q_parts:
        return []

    # Step 5: 一次性 batch encode — 核心优化点!
    Q_all = np.concatenate(q_parts, axis=0)  # (total_active, N)
    phi_all = phi_encoder.encode_normalized(Q_all)  # (total_active, D) 一次 matmul

    # Step 6: 按 position 切分回去
    positions = []
    for i, (flat_start, n_active, gold_word) in enumerate(position_infos):
        phi_pos = phi_all[flat_start:flat_start + n_active]
        positions.append((slot_parts[i], phi_pos, gold_word))

    return positions


# =====================================================================
# 2. 语料预处理主函数
# =====================================================================

class PreprocessResult:
    """预处理结果: CSR 紧凑格式"""
    __slots__ = [
        'offsets', 'flat_active', 'flat_phi', 'targets',
        'n_pos', 'S', 'V', 'D',
        'slot_universe', 'slot_to_compact', 'word_to_idx', 'vocab_list',
        'word_proj_indices', 'word_proj_values',
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def preprocess_corpus(corpus: List[str], model, *,
                      phi_mode: str = 'chaos',
                      verbose: bool = True) -> PreprocessResult:
    """高效预处理语料 → CSR 紧凑格式

    核心优化:
      - batch phi: 全句一次性 encode
      - CSR 存储: 零 padding
      - sparse word_proj: 只存非零索引和值
    """
    t0 = time.time()
    config = model.config
    retina = model.retina
    phi_encoder = model.phi_encoder
    rho = config.rho.astype(np.float32)
    N = config.N
    eps_q = float(config.EPS_Q)

    # Ensure all words in vocab
    for s in corpus:
        for w in WakeDataset._tokenize(s):
            model._ensure_word(w)

    # Phase 1: 逐句处理 (句间可并行, 句内 batch phi)
    all_positions = []
    for sentence in corpus:
        words = WakeDataset._tokenize(sentence)
        positions = _process_sentence_batch_phi(
            words, retina, phi_encoder, rho, N, eps_q)
        all_positions.extend(positions)

    if not all_positions:
        raise ValueError("No valid positions found in corpus")

    n_pos = len(all_positions)

    # Phase 2: 构建 slot universe + word projection
    all_slots = set()
    for act, _, gw in all_positions:
        all_slots.update(int(s) for s in act)
    for word in model.vocab:
        if word in model.word_to_slots:
            all_slots.update(int(s) for s in model.word_to_slots[word])

    slot_universe = np.array(sorted(all_slots), dtype=np.int32)
    slot_to_compact = {int(s): i for i, s in enumerate(slot_universe)}
    S = len(slot_universe)

    # Build vocab
    from .inference_fast import SparseMatrixDecoder
    decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, config.K)
    vocab_list = decoder.word_list
    V = len(vocab_list)
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}

    # Phase 3: 转换为 CSR 紧凑格式 (零 padding!)
    offsets = np.zeros(n_pos + 1, dtype=np.int64)
    for i, (act, _, _) in enumerate(all_positions):
        offsets[i + 1] = offsets[i] + len(act)

    total_active = int(offsets[-1])
    D = all_positions[0][1].shape[1]

    flat_active = np.empty(total_active, dtype=np.int64)
    flat_phi = np.empty((total_active, D), dtype=np.float32)
    targets = np.empty(n_pos, dtype=np.int64)

    for i, (act, phi, gold_word) in enumerate(all_positions):
        lo = int(offsets[i])
        hi = int(offsets[i + 1])
        # Map slot IDs to compact indices
        flat_active[lo:hi] = np.array(
            [slot_to_compact[int(s)] for s in act], dtype=np.int64)
        flat_phi[lo:hi] = phi
        targets[i] = word_to_idx.get(gold_word, 0)

    # Phase 4: 稀疏 word_proj (只存非零元素)
    #   word_proj[w, s] = 1/n_slots if slot s belongs to word w
    wp_rows = []
    wp_cols = []
    wp_vals = []
    for wi, w in enumerate(vocab_list):
        if w in model.word_to_slots:
            slots = model.word_to_slots[w]
            compact = [slot_to_compact[int(sl)]
                       for sl in slots if int(sl) in slot_to_compact]
            if compact:
                v = 1.0 / len(compact)
                for c in compact:
                    wp_rows.append(wi)
                    wp_cols.append(c)
                    wp_vals.append(v)

    elapsed = time.time() - t0

    if verbose:
        avg_active = total_active / n_pos
        storage_mb = (total_active * (8 + D * 4) + n_pos * 8) / (1024 ** 2)
        old_storage_mb = (n_pos * max(len(p[0]) for p in all_positions)
                          * (8 + D * 4)) / (1024 ** 2)
        print(f"  预处理完成: {n_pos:,} positions, "
              f"avg_active={avg_active:.1f}, S={S}, V={V}, D={D}")
        print(f"  CSR 存储: {storage_mb:.1f}MB "
              f"(padded 需 {old_storage_mb:.1f}MB, "
              f"压缩 {old_storage_mb/max(storage_mb,1):.1f}x)")
        print(f"  耗时: {elapsed:.1f}s")

    return PreprocessResult(
        offsets=offsets,
        flat_active=flat_active,
        flat_phi=flat_phi,
        targets=targets,
        n_pos=n_pos,
        S=S, V=V, D=D,
        slot_universe=slot_universe,
        slot_to_compact=slot_to_compact,
        word_to_idx=word_to_idx,
        vocab_list=vocab_list,
        word_proj_indices=(np.array(wp_rows, dtype=np.int64),
                           np.array(wp_cols, dtype=np.int64)),
        word_proj_values=np.array(wp_vals, dtype=np.float32),
    )


# =====================================================================
# 3. Dataset + DataLoader (per-batch 动态 padding)
# =====================================================================

class WakeMapDataset(Dataset):
    """CSR 紧凑格式的 MapDataset — 按 position 随机访问"""

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
            self.flat_active[lo:hi],   # (n_active,) int64
            self.flat_phi[lo:hi],      # (n_active, D) float32
            self.targets[idx],         # int64
        )


def collate_wake_batch(batch):
    """per-batch 动态 padding: 只 pad 到 batch 内最大 active

    batch: List[(active, phi, target)]
    Returns: (active, phi, mask, target) — padded tensors
    """
    max_M = max(act.shape[0] for act, _, _ in batch)
    B = len(batch)
    D = batch[0][1].shape[1]

    active = torch.zeros(B, max_M, dtype=torch.long)
    phi = torch.zeros(B, max_M, D, dtype=torch.float32)
    mask = torch.zeros(B, max_M, dtype=torch.float32)
    targets = []

    for i, (act, feat, tgt) in enumerate(batch):
        n = len(act)
        active[i, :n] = torch.from_numpy(act)
        phi[i, :n] = torch.from_numpy(feat)
        mask[i, :n] = 1.0
        targets.append(tgt)

    target = torch.tensor(targets, dtype=torch.long)
    return active, phi, mask, target


def build_dataloader(result: PreprocessResult, *,
                     batch_size: int = 512,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = False) -> DataLoader:
    """构建 DataLoader — 动态 padding, 零内存浪费"""
    dataset = WakeMapDataset(result)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_wake_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def build_sparse_word_proj(result: PreprocessResult,
                           device: torch.device) -> torch.Tensor:
    """构建稀疏 word_proj_t 矩阵 (S, V)

    word_proj[w, s] 是 dense → 99.9% 是零
    改用 sparse: CUDA 上用 torch.sparse_mm, 10x+ 加速
    """
    rows, cols = result.word_proj_indices
    vals = result.word_proj_values

    # word_proj: (V, S) → word_proj_t: (S, V)
    indices = torch.stack([
        torch.from_numpy(cols.astype(np.int64)),
        torch.from_numpy(rows.astype(np.int64)),
    ])
    values = torch.from_numpy(vals)

    word_proj_t = torch.sparse_coo_tensor(
        indices, values,
        size=(result.S, result.V),
        dtype=torch.float32,
    ).coalesce().to(device)

    return word_proj_t
