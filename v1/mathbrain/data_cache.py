"""预处理 + mmap 数据缓存

预处理阶段（一次性，离线）:
  corpus.txt → preprocess → cache/corpus_N8_D32.bin

训练阶段:
  mmap 打开 .bin → numpy view → torch.from_numpy → .to(gpu)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import MathBrainConfig
from .wake_dataset import WakeDataset
from .inference_fast import SparseMatrixDecoder

HEADER_SIZE = 512


def _corpus_hash(corpus: List[str]) -> str:
    """SHA256 of joined corpus for cache invalidation."""
    h = hashlib.sha256()
    for s in corpus:
        h.update(s.encode('utf-8'))
    return f"sha256:{h.hexdigest()[:16]}"


# Bump this when binary layout changes (e.g. dtype float16→float32)
_CACHE_FORMAT_VERSION = 2


def _cache_filename(corpus_path: str, config: MathBrainConfig) -> str:
    """Generate deterministic cache filename from corpus path + config."""
    stem = Path(corpus_path).stem if corpus_path else "corpus"
    return f"{stem}_N{config.N}_D{config.D_PHI}_v{_CACHE_FORMAT_VERSION}.bin"


# ── Vectorized sentence processing ─────────────────────────────────

def _process_one_sentence_vectorized(
        words, retina, phi_encoder, rho, N, eps_q,
        precompute, phi_mode):
    """处理单个句子: 向量化 EMA + batch phi 编码。

    EMA 是线性递推 Q[t] = ρ·Q[t-1] + x[t]，
    对全部 slot 展开为 dense 矩阵，一次性向量化计算所有位置的 Q 状态。
    然后用 batch phi_encoder.encode() 计算所有位置的特征。

    Returns: List[(active_slots, feature, gold_word)]
    """
    if len(words) < 2:
        return []

    encoded = [retina.encode(w) for w in words]

    # ── Step 1: 收集所有出现过的 slot，建立 dense 索引 ─────
    all_slots_set = set()
    for enc in encoded[:-1]:  # 最后一个词只做 target，不做输入
        all_slots_set.update(enc.keys())
    if not all_slots_set:
        return []

    slot_list = sorted(all_slots_set)
    slot_to_dense = {s: i for i, s in enumerate(slot_list)}
    n_slots = len(slot_list)
    T = len(words) - 1  # 有 T 个预测位置

    # ── Step 2: 构建 dense 输入矩阵 x[t, slot] ─────
    x = np.zeros((T, n_slots), dtype=np.float32)
    for t in range(T):
        for slot_id, count in encoded[t].items():
            x[t, slot_to_dense[slot_id]] = float(count)

    # ── Step 3: 向量化 EMA 递推 Q[t] = ρ·Q[t-1] + x[t] ─────
    # rho shape: (N,), Q shape: (T, n_slots, N)
    Q = np.zeros((T, n_slots, N), dtype=np.float32)
    # Q[0, :, :] = x[0, :, np.newaxis]  (broadcast x to all N scales)
    Q[0] = x[0, :, np.newaxis]
    for t in range(1, T):
        # vectorized: (n_slots, N) * (N,) + (n_slots, 1)
        Q[t] = Q[t - 1] * rho[np.newaxis, :] + x[t, :, np.newaxis]

    # ── Step 4: 逐位置提取活跃 slot + batch phi 编码 ─────
    alpha_scale = phi_encoder.alpha_scale
    positions = []

    for t in range(T):
        Q_t = Q[t]  # (n_slots, N)

        # 裁剪: 仅保留 |Q| >= eps_q 的 slot
        alive = np.max(np.abs(Q_t), axis=1) >= eps_q
        if not np.any(alive):
            continue

        active_dense_idx = np.where(alive)[0]
        active_slots = np.array([slot_list[i] for i in active_dense_idx],
                                dtype=np.int32)
        Q_active = Q_t[active_dense_idx]  # (n_active, N)

        # Phi 编码
        q_scaled = (Q_active * alpha_scale).astype(np.float32)
        if precompute and phi_mode == 'chaos':
            phi = phi_encoder.encode(Q_active)
            norms = np.linalg.norm(phi, axis=1, keepdims=True)
            feature = (phi / (norms + 1e-8)).astype(np.float32)
        elif precompute and phi_mode == 'raw':
            norms = np.linalg.norm(q_scaled, axis=1, keepdims=True)
            feature = (q_scaled / (norms + 1e-8)).astype(np.float32)
        else:
            feature = q_scaled

        gold_word = words[t + 1]
        positions.append((active_slots, feature, gold_word))

    return positions


def _worker_process_chunk(args):
    """multiprocessing worker: 处理一批句子。"""
    sentences, retina, phi_encoder, rho, N, eps_q, precompute, phi_mode = args
    results = []
    for sentence in sentences:
        words = WakeDataset._tokenize(sentence)
        results.extend(_process_one_sentence_vectorized(
            words, retina, phi_encoder, rho, N, eps_q,
            precompute, phi_mode))
    return results


def _process_all_sentences(corpus, retina, phi_encoder, rho, N, eps_q,
                           theta_q, precompute, phi_mode):
    """并行处理所有句子: 句间 multiprocessing + 句内向量化 EMA。"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    n_workers = min(mp.cpu_count(), len(corpus), 8)

    if n_workers <= 1 or len(corpus) < 4:
        # 小语料或单核: 直接串行（避免 fork/pickle 开销）
        positions = []
        for sentence in corpus:
            words = WakeDataset._tokenize(sentence)
            positions.extend(_process_one_sentence_vectorized(
                words, retina, phi_encoder, rho, N, eps_q,
                precompute, phi_mode))
        return positions

    # 切分 corpus 为 n_workers 个 chunk
    chunk_size = max(1, (len(corpus) + n_workers - 1) // n_workers)
    chunks = [corpus[i:i + chunk_size] for i in range(0, len(corpus), chunk_size)]

    # 打包 worker 参数
    worker_args = [
        (chunk, retina, phi_encoder, rho, N, eps_q, precompute, phi_mode)
        for chunk in chunks
    ]

    print(f"  并行预处理: {len(chunks)} chunks × {n_workers} workers")

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            chunk_results = list(pool.map(_worker_process_chunk, worker_args))
        # 合并所有 chunk 的结果
        positions = []
        for chunk_pos in chunk_results:
            positions.extend(chunk_pos)
    except Exception as e:
        # Pickle 失败时 fallback 到串行
        print(f"  并行失败 ({e}), fallback 串行...")
        positions = []
        for sentence in corpus:
            words = WakeDataset._tokenize(sentence)
            positions.extend(_process_one_sentence_vectorized(
                words, retina, phi_encoder, rho, N, eps_q,
                precompute, phi_mode))

    return positions


def preprocess_corpus(corpus: List[str], model, *,
                      cache_dir: str = 'cache/',
                      corpus_path: str = None,
                      phi_mode: str = 'chaos',
                      learnable_P: bool = False,
                      storage_dtype: str = 'fp32') -> str:
    """预处理语料 → 二进制缓存文件。返回缓存路径。

    如果缓存已存在且 corpus_hash 匹配，直接返回路径（跳过预处理）。
    """
    config = model.config
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = _cache_filename(corpus_path, config)
    cache_path = str(cache_dir / fname)
    c_hash = _corpus_hash(corpus)

    # Check existing cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                raw_header = f.read(HEADER_SIZE)
            meta = json.loads(raw_header.rstrip(b'\x00').decode('utf-8'))
            # Validate: corpus hash + feat_dim must both match
            expected_D = model.phi_encoder.D if (not learnable_P and phi_mode == 'chaos') else config.N
            cached_D = meta.get('feat_dim', -1)
            if meta.get('corpus_hash') == c_hash and cached_D == expected_D:
                print(f"  缓存命中: {cache_path}")
                return cache_path
            elif meta.get('corpus_hash') != c_hash:
                print(f"  缓存过期 (hash mismatch), 重新预处理...")
            else:
                print(f"  缓存过期 (feat_dim {cached_D}≠{expected_D}), 重新预处理...")
        except Exception:
            print(f"  缓存损坏, 重新预处理...")

    t0 = time.time()
    print(f"  预处理 → {cache_path} ...")

    # Ensure all words in vocab
    for s in corpus:
        for w in WakeDataset._tokenize(s):
            model._ensure_word(w)

    # Extract serializable params for parallel workers
    retina = model.retina
    phi_encoder = model.phi_encoder
    rho = config.rho.astype(np.float32)
    N = config.N
    eps_q = float(config.EPS_Q)
    theta_q = float(config.THETA_Q)
    precompute = not learnable_P

    # Phase 1: parallel sentence processing
    positions = _process_all_sentences(
        corpus, retina, phi_encoder, rho, N, eps_q, theta_q,
        precompute, phi_mode)

    if not positions:
        raise ValueError("No valid positions found in corpus")

    # Build slot universe + word projection (same as trainer._build_tensors)
    all_slots = set()
    for act, _, gw in positions:
        all_slots.update(act.tolist())
    for word in model.vocab:
        if word in model.word_to_slots:
            all_slots.update(int(s) for s in model.word_to_slots[word])
    slot_universe = np.array(sorted(all_slots), dtype=np.int32)
    slot_to_compact = {int(s): i for i, s in enumerate(slot_universe.tolist())}
    S = len(slot_universe)

    decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, config.K)
    vocab_list = decoder.word_list
    V = len(vocab_list)
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}

    # Word projection matrix
    word_proj = np.zeros((V, S), dtype=np.float32)
    for wi, w in enumerate(vocab_list):
        if w in model.word_to_slots:
            slots = model.word_to_slots[w]
            compact = [slot_to_compact[int(sl)]
                       for sl in slots if int(sl) in slot_to_compact]
            if compact:
                word_proj[wi, compact] = 1.0 / len(compact)

    # Pad positions to max_active
    n_pos = len(positions)
    feat_dim = positions[0][1].shape[1]
    max_active = max(len(p[0]) for p in positions)

    print(f"  n_pos={n_pos}, max_active={max_active}, feat_dim={feat_dim}, "
          f"S={S}, V={V}")

    # Write binary cache
    _write_cache(cache_path, positions, slot_to_compact, word_to_idx,
                 word_proj, n_pos, max_active, feat_dim, S, V,
                 config, c_hash, phi_mode, storage_dtype)

    # Save sidecar metadata (slot_universe, vocab_list)
    meta_path = cache_path + '.meta.npz'
    np.savez(meta_path,
             slot_universe=slot_universe,
             vocab_list=np.array(vocab_list, dtype=object))

    elapsed = time.time() - t0
    file_size = os.path.getsize(cache_path)
    print(f"  预处理完成: {file_size/1e9:.2f}GB, {elapsed:.1f}s")

    return cache_path


def _write_cache(cache_path, positions, slot_to_compact, word_to_idx,
                 word_proj, n_pos, max_active, feat_dim, S, V,
                 config, c_hash, phi_mode, storage_dtype='fp32'):
    """Write padded binary cache: single-pass array construction + streaming write."""
    M = max_active
    D = feat_dim

    # Resolve storage dtypes
    if storage_dtype == 'bf16':
        phi_dtype = np.float16      # on-disk (bfloat16 ≈ float16 for data)
        phi_bpc = 2
        active_dtype = np.int16 if S < 32768 else np.int32
        active_bpc = active_dtype().itemsize
    else:
        phi_dtype = np.float32
        phi_bpc = 4
        active_dtype = np.int32
        active_bpc = 4

    # Build header
    meta = {
        'version': 1,
        'n_pos': n_pos,
        'max_active': M,
        'feat_dim': D,
        'S': S,
        'V': V,
        'd_rank': int(config.CP_RANK),
        'N': int(config.N),
        'rho': list(config.RHO),
        'phi_mode': phi_mode,
        'corpus_hash': c_hash,
        'storage_dtype': storage_dtype,
    }
    header_bytes = json.dumps(meta).encode('utf-8')
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_bytes)} > {HEADER_SIZE}")
    header_bytes = header_bytes.ljust(HEADER_SIZE, b'\x00')

    # Compute section sizes
    active_bytes = n_pos * M * active_bpc
    phi_bytes = n_pos * M * D * phi_bpc
    mask_bytes = n_pos * M              # uint8
    target_bytes = n_pos * 4            # int32
    wp_bytes = V * S * 4               # float32

    total = HEADER_SIZE + active_bytes + phi_bytes + mask_bytes + target_bytes + wp_bytes
    print(f"  cache size: {total/1e9:.2f}GB "
          f"(active={active_bytes/1e9:.2f}, phi={phi_bytes/1e9:.2f}, "
          f"mask={mask_bytes/1e9:.2f}, target={target_bytes/1e9:.2f}, "
          f"word_proj={wp_bytes/1e9:.2f})"
          f"  [storage={storage_dtype}]")

    # ── Single-pass: build all arrays at once ─────────────────────
    CHUNK = 10000

    with open(cache_path, 'wb') as f:
        f.write(header_bytes)

        for start in range(0, n_pos, CHUNK):
            end = min(start + CHUNK, n_pos)
            sz = end - start

            a_buf = np.zeros((sz, M), dtype=active_dtype)
            p_buf = np.zeros((sz, M, D), dtype=np.float32)
            m_buf = np.zeros((sz, M), dtype=np.uint8)
            t_buf = np.zeros(sz, dtype=np.int32)

            for j, idx in enumerate(range(start, end)):
                act, feat, gw = positions[idx]
                n = len(act)
                a_buf[j, :n] = [slot_to_compact[int(a)] for a in act]
                p_buf[j, :n] = feat
                m_buf[j, :n] = 1
                t_buf[j] = word_to_idx.get(gw, 0)

            # Write active section (accumulate across chunks)
            f.seek(HEADER_SIZE + start * M * active_bpc)
            f.write(a_buf.astype(active_dtype).tobytes())

        # Write phi (second pass — could interleave but layout needs contiguous sections)
        f.seek(HEADER_SIZE + n_pos * M * active_bpc)
        for start in range(0, n_pos, CHUNK):
            end = min(start + CHUNK, n_pos)
            sz = end - start
            p_buf = np.zeros((sz, M, D), dtype=np.float32)
            for j, idx in enumerate(range(start, end)):
                act, feat, gw = positions[idx]
                p_buf[j, :len(act)] = feat
            f.write(p_buf.astype(phi_dtype).tobytes())

        # Write mask
        for start in range(0, n_pos, CHUNK):
            end = min(start + CHUNK, n_pos)
            sz = end - start
            m_buf = np.zeros((sz, M), dtype=np.uint8)
            for j, idx in enumerate(range(start, end)):
                act, _, _ = positions[idx]
                m_buf[j, :len(act)] = 1
            f.write(m_buf.tobytes())

        # Write target
        for start in range(0, n_pos, CHUNK):
            end = min(start + CHUNK, n_pos)
            sz = end - start
            t_buf = np.zeros(sz, dtype=np.int32)
            for j, idx in enumerate(range(start, end)):
                _, _, gw = positions[idx]
                t_buf[j] = word_to_idx.get(gw, 0)
            f.write(t_buf.tobytes())

        # Word projection: always float32
        f.write(word_proj.astype(np.float32).tobytes())


def load_cache(cache_path: str, device) -> dict:
    """mmap 打开缓存文件，返回 numpy view 或 GPU tensor。

    自动检测 storage_dtype (fp32/bf16)，加载时转换为 fp32。

    Returns:
      {
        'active': numpy/torch (n_pos, M),
        'phi': numpy/torch (n_pos, M, D),
        'mask': numpy/torch (n_pos, M),
        'target': torch (n_pos,) int64 on device,
        'word_proj_t': torch (S, V) float32 on device,
        'meta': dict,
        'on_gpu': bool,
      }
    """
    import torch

    # Read header
    with open(cache_path, 'rb') as f:
        raw_header = f.read(HEADER_SIZE)
    meta = json.loads(raw_header.rstrip(b'\x00').decode('utf-8'))

    n_pos = meta['n_pos']
    M = meta['max_active']
    D = meta['feat_dim']
    S = meta['S']
    V = meta['V']
    storage_dtype = meta.get('storage_dtype', 'fp32')

    # Resolve on-disk dtypes
    if storage_dtype == 'bf16':
        phi_disk_dtype = np.float16
        phi_bpc = 2
        active_disk_dtype = np.int16 if S < 32768 else np.int32
        active_bpc = active_disk_dtype().itemsize
    else:
        phi_disk_dtype = np.float32
        phi_bpc = 4
        active_disk_dtype = np.int32
        active_bpc = 4

    # Compute offsets
    off_active = HEADER_SIZE
    off_phi = off_active + n_pos * M * active_bpc
    off_mask = off_phi + n_pos * M * D * phi_bpc
    off_target = off_mask + n_pos * M
    off_wp = off_target + n_pos * 4

    # mmap the file
    mm = np.memmap(cache_path, dtype=np.uint8, mode='r')

    active_np = np.ndarray((n_pos, M), dtype=active_disk_dtype,
                           buffer=mm, offset=off_active)
    phi_np = np.ndarray((n_pos, M, D), dtype=phi_disk_dtype,
                        buffer=mm, offset=off_phi)
    mask_np = np.ndarray((n_pos, M), dtype=np.uint8,
                         buffer=mm, offset=off_mask)
    target_np = np.ndarray((n_pos,), dtype=np.int32,
                           buffer=mm, offset=off_target)
    wp_np = np.ndarray((V, S), dtype=np.float32,
                       buffer=mm, offset=off_wp)

    # Word projection → GPU (always fits)
    word_proj_t = torch.from_numpy(wp_np.T.copy()).float().to(device)
    target = torch.from_numpy(target_np.copy()).long().to(device)

    # Try to fit data on GPU
    # Compute GPU size (always fp32 after conversion)
    data_bytes_gpu = n_pos * M * (8 + D * 4 + 4)  # int64 + float32 + float32
    on_gpu = False

    if device.type == 'cuda':
        free_mem = torch.cuda.mem_get_info()[0]
        if data_bytes_gpu <= free_mem - 200 * 1024 * 1024:
            on_gpu = True

    disk_bytes = n_pos * M * (active_bpc + D * phi_bpc + 1)
    if on_gpu:
        print(f"  → GPU ({disk_bytes/1e9:.1f}GB disk, "
              f"{data_bytes_gpu/1e9:.1f}GB GPU)  [{storage_dtype}]")
        active = torch.from_numpy(active_np.astype(np.int64)).to(device)
        phi = torch.from_numpy(phi_np.astype(np.float32)).to(device)
        mask = torch.from_numpy(mask_np.astype(np.float32)).to(device)
        del mm  # release mmap
    else:
        print(f"  → mmap ({disk_bytes/1e9:.1f}GB"
              f", {'GPU 不够' if device.type == 'cuda' else 'CPU mode'})"
              f"  [{storage_dtype}]")
        # Keep as mmap numpy views — OS page cache handles hot data
        active = active_np
        phi = phi_np
        mask = mask_np

    # Load sidecar metadata
    meta_path = cache_path + '.meta.npz'
    sidecar = np.load(meta_path, allow_pickle=True)
    slot_universe = sidecar['slot_universe']
    vocab_list = sidecar['vocab_list'].tolist()

    return {
        'active': active,
        'phi': phi,
        'mask': mask,
        'target': target,
        'word_proj_t': word_proj_t,
        'meta': meta,
        'on_gpu': on_gpu,
        'slot_universe': slot_universe,
        'vocab_list': vocab_list,
        '_mm': mm if not on_gpu else None,  # keep mmap alive
    }
