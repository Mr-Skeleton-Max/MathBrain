"""MLX 稀疏局部 proposal 训练器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import time
import numpy as np

from .wake_dataset import WakeDataset

try:
    import mlx.core as mx
    HAS_MLX = True
except Exception:
    HAS_MLX = False
    mx = None


@dataclass
class SparseWakeDatasetMLX:
    slot_universe: np.ndarray
    pair_keys_mx: object
    src_idx_mx: object
    tgt_idx_mx: object
    phi_mx: object
    start_idx_mx: object
    end_idx_mx: object
    count_mx: object
    phi_mean_mx: object
    unique_keys_mx: object
    src_unique_mx: object
    tgt_unique_mx: object
    active_pos_mx: object
    active_src_mx: object
    active_phi_mx: object
    active_pos_sorted_mx: object
    active_phi_sorted_mx: object
    active_src_unique_np: np.ndarray
    active_src_unique_mx: object
    active_src_start_mx: object
    active_src_end_mx: object
    target_offsets_np: np.ndarray
    target_compact_np: np.ndarray
    unique_keys_np: np.ndarray
    total_positions: int
    total_pairs: int
    n_shards: int
    pos_ids_sorted_mx: object = None


class TreeWakeTrainerMLX:
    def __init__(self, model):
        if not HAS_MLX:
            raise RuntimeError("MLX is not available for TreeWakeTrainerMLX")
        self.model = model
        self.b = model.b
        self.a = model.a
        self.config = model.config
        self.K = model.config.K
        self.D = model.config.d
        self.slot_universe = None
        self.slot_to_compact = None
        self.b_dense = None
        self.e_v = None
        self.e_q = None
        self._a_mode = None
        self._dense_dirty = False
        self._tracked_keys = None
        self._b_flat = None
        self._a_edge_cache = None
        self._a_edge_cache_keys = None
        self._a_edge_cache_cycle = -1
        self.profile = {}
        self._warmup_done = False

    def _warmup_mlx_build_ops(self):
        if self._warmup_done:
            return
        tiny_keys = mx.array(np.array([3, 1, 2, 1], dtype=np.int64))
        tiny_vals = mx.array(np.arange(16, dtype=np.float32).reshape(4, 4))
        tiny_idx = mx.array(np.array([0, 2, 4], dtype=np.int32))
        order = mx.argsort(tiny_keys)
        sorted_vals = tiny_vals[order]
        prefix = mx.concatenate([mx.zeros((1, 4), dtype=mx.float32), mx.cumsum(sorted_vals, axis=0)], axis=0)
        seg = prefix[tiny_idx[1:]] - prefix[tiny_idx[:-1]]
        mx.eval(order, prefix, seg)
        self._warmup_done = True

    def build_dataset(self, corpus: List[str], verbose: bool = True, shard_size: int = 128) -> SparseWakeDatasetMLX:
        self._warmup_mlx_build_ops()
        t_build0 = time.time()
        t_stage0 = time.time()
        base = WakeDataset.from_corpus(self.model, corpus, verbose=verbose)
        t_stage1 = time.time()

        if len(base.active_slots) == 0 and len(base.target_slots) == 0:
            self.slot_universe = np.zeros((0,), dtype=np.int32)
            self.slot_to_compact = {}
            return SparseWakeDatasetMLX(
                slot_universe=self.slot_universe,
                pair_keys_mx=mx.zeros((0,), dtype=mx.int64),
                src_idx_mx=mx.zeros((0,), dtype=mx.int32),
                tgt_idx_mx=mx.zeros((0,), dtype=mx.int32),
                phi_mx=mx.zeros((0, self.D), dtype=mx.float32),
                start_idx_mx=mx.zeros((0,), dtype=mx.int32),
                end_idx_mx=mx.zeros((0,), dtype=mx.int32),
                count_mx=mx.zeros((0,), dtype=mx.float32),
                phi_mean_mx=mx.zeros((0, self.D), dtype=mx.float32),
                unique_keys_mx=mx.zeros((0,), dtype=mx.int64),
                src_unique_mx=mx.zeros((0,), dtype=mx.int32),
                tgt_unique_mx=mx.zeros((0,), dtype=mx.int32),
                active_pos_mx=mx.zeros((0,), dtype=mx.int32),
                active_src_mx=mx.zeros((0,), dtype=mx.int32),
                active_phi_mx=mx.zeros((0, self.D), dtype=mx.float32),
                active_pos_sorted_mx=mx.zeros((0,), dtype=mx.int32),
                active_phi_sorted_mx=mx.zeros((0, self.D), dtype=mx.float32),
                active_src_unique_np=np.zeros((0,), dtype=np.int32),
                active_src_unique_mx=mx.zeros((0,), dtype=mx.int32),
                active_src_start_mx=mx.zeros((0,), dtype=mx.int32),
                active_src_end_mx=mx.zeros((0,), dtype=mx.int32),
                target_offsets_np=np.zeros((1,), dtype=np.int64),
                target_compact_np=np.zeros((0,), dtype=np.int32),
                unique_keys_np=np.zeros((0,), dtype=np.int64),
                total_positions=0,
                total_pairs=0,
                n_shards=0,
            )

        slot_universe = np.unique(np.concatenate([base.active_slots, base.target_slots]))
        active_compact = np.searchsorted(slot_universe, base.active_slots).astype(np.int32, copy=False)
        target_compact = np.searchsorted(slot_universe, base.target_slots).astype(np.int32, copy=False)
        t_stage2 = time.time()

        active_offsets = base.active_offsets.astype(np.int64, copy=False)
        target_offsets = base.target_offsets.astype(np.int64, copy=False)
        active_counts = np.diff(active_offsets)
        target_counts = np.diff(target_offsets)
        active_pos = np.repeat(np.arange(base.total_positions, dtype=np.int32), active_counts)
        active_src_eval = active_compact.astype(np.int32, copy=False)
        active_phi_eval = base.phi_hat.astype(np.float32, copy=False)
        target_compact_eval = target_compact.astype(np.int32, copy=False)
        if len(active_src_eval) > 0:
            active_order = np.argsort(active_src_eval, kind='stable')
            active_src_sorted = active_src_eval[active_order]
            active_pos_sorted = active_pos[active_order]
            active_phi_sorted = active_phi_eval[active_order]
            active_boundary = np.empty((len(active_src_sorted),), dtype=bool)
            active_boundary[0] = True
            active_boundary[1:] = active_src_sorted[1:] != active_src_sorted[:-1]
            active_src_start = np.flatnonzero(active_boundary).astype(np.int32, copy=False)
            active_src_end = np.concatenate([active_src_start[1:], np.array([len(active_src_sorted)], dtype=np.int32)]).astype(np.int32, copy=False)
            active_src_unique = active_src_sorted[active_src_start].astype(np.int32, copy=False)
        else:
            active_pos_sorted = np.zeros((0,), dtype=np.int32)
            active_phi_sorted = np.zeros((0, self.D), dtype=np.float32)
            active_src_start = np.zeros((0,), dtype=np.int32)
            active_src_end = np.zeros((0,), dtype=np.int32)
            active_src_unique = np.zeros((0,), dtype=np.int32)
        pair_counts = active_counts * target_counts
        total_pairs = int(pair_counts.sum())
        s = len(slot_universe)

        if total_pairs > 0:
            t_pair0 = time.time()
            pos_ids = np.repeat(np.arange(base.total_positions, dtype=np.int32), pair_counts)
            pair_offsets = np.arange(total_pairs, dtype=np.int64) - np.repeat(
                np.cumsum(pair_counts, dtype=np.int64) - pair_counts,
                pair_counts,
            )
            tgt_counts_rep = target_counts[pos_ids].astype(np.int64, copy=False)
            src_local = (pair_offsets // tgt_counts_rep).astype(np.int64, copy=False)
            tgt_local = (pair_offsets % tgt_counts_rep).astype(np.int64, copy=False)

            src_rows = (active_offsets[pos_ids] + src_local).astype(np.int64, copy=False)
            tgt_rows = (target_offsets[pos_ids] + tgt_local).astype(np.int64, copy=False)

            src_idx = active_compact[src_rows].astype(np.int32, copy=False)
            tgt_idx = target_compact[tgt_rows].astype(np.int32, copy=False)
            phi_rep = base.phi_hat[src_rows].astype(np.float32, copy=False)

            keys = src_idx.astype(np.int64) * int(s) + tgt_idx.astype(np.int64)
            keys_mx = mx.array(keys)
            src_idx_mx_sorted = mx.array(src_idx)
            tgt_idx_mx_sorted = mx.array(tgt_idx)
            phi_rep_mx = mx.array(phi_rep)
            pos_ids_mx = mx.array(pos_ids)
            order_mx = mx.argsort(keys_mx)
            keys_sorted_mx = keys_mx[order_mx]
            src_idx_mx_sorted = src_idx_mx_sorted[order_mx]
            tgt_idx_mx_sorted = tgt_idx_mx_sorted[order_mx]
            phi_rep_mx = phi_rep_mx[order_mx]
            pos_ids_sorted_mx = pos_ids_mx[order_mx]
            mx.eval(keys_sorted_mx, src_idx_mx_sorted, tgt_idx_mx_sorted, phi_rep_mx, pos_ids_sorted_mx)
            t_pair1 = time.time()

            boundary_np = np.empty((total_pairs,), dtype=bool)
            boundary_np[0] = True
            boundary_np[1:] = np.array(keys_sorted_mx[1:] != keys_sorted_mx[:-1])
            start_idx = np.flatnonzero(boundary_np).astype(np.int32, copy=False)
            end_idx = np.concatenate([start_idx[1:], np.array([total_pairs], dtype=np.int32)]).astype(np.int32, copy=False)
            counts = (end_idx - start_idx).astype(np.float32, copy=False)
            start_idx_mx_tmp = mx.array(start_idx)
            end_idx_mx_tmp = mx.array(end_idx)
            phi_prefix = mx.concatenate([
                mx.zeros((1, self.D), dtype=mx.float32),
                mx.cumsum(phi_rep_mx, axis=0),
            ], axis=0)
            phi_sum_mx = phi_prefix[end_idx_mx_tmp] - phi_prefix[start_idx_mx_tmp]
            counts_mx_tmp = mx.array(counts)
            phi_mean = np.array(phi_sum_mx / counts_mx_tmp[:, None], copy=False).astype(np.float32, copy=False)
            unique_keys = np.array(keys_sorted_mx[start_idx_mx_tmp], copy=False).astype(np.int64, copy=False)
            src_unique = (unique_keys // s).astype(np.int32, copy=False)
            tgt_unique = (unique_keys % s).astype(np.int32, copy=False)
            src_idx = None
            tgt_idx = None
            phi_rep = None
            t_pair2 = time.time()
        else:
            t_pair0 = t_pair1 = t_pair2 = time.time()
            src_idx = np.zeros((0,), dtype=np.int32)
            tgt_idx = np.zeros((0,), dtype=np.int32)
            phi_rep = np.zeros((0, self.D), dtype=np.float32)
            start_idx = np.zeros((0,), dtype=np.int32)
            end_idx = np.zeros((0,), dtype=np.int32)
            counts = np.zeros((0,), dtype=np.float32)
            phi_mean = np.zeros((0, self.D), dtype=np.float32)
            unique_keys = np.zeros((0,), dtype=np.int64)
            src_unique = np.zeros((0,), dtype=np.int32)
            tgt_unique = np.zeros((0,), dtype=np.int32)

        self.slot_universe = slot_universe.astype(np.int32, copy=False)
        self.slot_to_compact = {int(slot): idx for idx, slot in enumerate(self.slot_universe.tolist())}

        build_ms = (time.time() - t_build0) * 1000
        n_shards = int((base.total_positions + max(shard_size, 1) - 1) // max(shard_size, 1)) if base.total_positions else 0
        self.profile['dataset_build_ms'] = build_ms
        self.profile['dataset_from_corpus_ms'] = (t_stage1 - t_stage0) * 1000
        self.profile['dataset_slot_map_ms'] = (t_stage2 - t_stage1) * 1000
        self.profile['dataset_pair_expand_sort_ms'] = (t_pair1 - t_pair0) * 1000
        self.profile['dataset_group_premean_ms'] = (t_pair2 - t_pair1) * 1000
        self.profile['dataset_total_pairs'] = float(total_pairs)
        self.profile['dataset_total_shards'] = float(n_shards)
        self.profile['dataset_unique_edges'] = float(len(unique_keys))
        if verbose:
            print(
                f"Sparse MLX WakeDataset: positions={base.total_positions}, "
                f"slots={len(self.slot_universe)}, pairs={total_pairs}, edges={len(unique_keys)}, "
                f"build_ms={build_ms:.1f}"
            )

        return SparseWakeDatasetMLX(
            slot_universe=self.slot_universe,
            pos_ids_sorted_mx=pos_ids_sorted_mx if total_pairs > 0 else mx.zeros((0,), dtype=mx.int32),
            pair_keys_mx=keys_sorted_mx if total_pairs > 0 else mx.zeros((0,), dtype=mx.int64),
            src_idx_mx=src_idx_mx_sorted if total_pairs > 0 else mx.zeros((0,), dtype=mx.int32),
            tgt_idx_mx=tgt_idx_mx_sorted if total_pairs > 0 else mx.zeros((0,), dtype=mx.int32),
            phi_mx=phi_rep_mx if total_pairs > 0 else mx.zeros((0, self.D), dtype=mx.float32),
            start_idx_mx=mx.array(start_idx),
            end_idx_mx=mx.array(end_idx),
            count_mx=mx.array(counts),
            phi_mean_mx=mx.array(phi_mean),
            unique_keys_mx=mx.array(unique_keys),
            src_unique_mx=mx.array(src_unique),
            tgt_unique_mx=mx.array(tgt_unique),
            active_pos_mx=mx.array(active_pos),
            active_src_mx=mx.array(active_src_eval),
            active_phi_mx=mx.array(active_phi_eval),
            active_pos_sorted_mx=mx.array(active_pos_sorted),
            active_phi_sorted_mx=mx.array(active_phi_sorted),
            active_src_unique_np=active_src_unique,
            active_src_unique_mx=mx.array(active_src_unique),
            active_src_start_mx=mx.array(active_src_start),
            active_src_end_mx=mx.array(active_src_end),
            target_offsets_np=target_offsets,
            target_compact_np=target_compact_eval,
            unique_keys_np=unique_keys,
            total_positions=base.total_positions,
            total_pairs=total_pairs,
            n_shards=n_shards,
        )

    def _init_dense_b(self):
        s = len(self.slot_universe)
        b_dense_np = np.zeros((s, s, self.D), dtype=np.float32)
        tracked_keys = []
        n = int(self.b._n_entries)
        if n > 0:
            for idx in range(n):
                src = int(self.b._srcs[idx])
                tgt = int(self.b._tgts[idx])
                src_compact = self.slot_to_compact.get(src)
                tgt_compact = self.slot_to_compact.get(tgt)
                if src_compact is not None and tgt_compact is not None:
                    b_dense_np[src_compact, tgt_compact] = self.b._vecs[idx]
                    tracked_keys.append(src_compact * s + tgt_compact)
        self.b_dense = mx.array(b_dense_np)
        self._b_flat = self.b_dense.reshape((s * s, self.D))
        self._tracked_keys = np.unique(np.asarray(tracked_keys, dtype=np.int64)) if tracked_keys else np.zeros((0,), dtype=np.int64)

    def _prepare_a_dense(self):
        s = len(self.slot_universe)
        if not self.a.has_knowledge:
            self.e_v = None
            self.e_q = None
            self._a_mode = None
            return

        self.b._precompute_a_embeddings(self.a)
        cache = self.b._a_cache
        if cache is None:
            self.e_v = None
            self.e_q = None
            self._a_mode = None
            return

        if 'E_V' in cache and 'E_Q' in cache:
            e_v_np = np.zeros((s, self.D), dtype=np.float32)
            e_q_np = np.zeros((s, self.D), dtype=np.float32)
            for compact, slot in enumerate(self.slot_universe):
                idx = cache['slot_to_idx'][slot] if slot <= cache['max_slot'] else -1
                if idx >= 0:
                    e_v_np[compact] = cache['E_V'][idx]
                    e_q_np[compact] = cache['E_Q'][idx]
            self.e_v = mx.array(e_v_np)
            self.e_q = mx.array(e_q_np)
            self._a_mode = 'legacy'
            return

        if 'E_src' in cache and 'E_tgt' in cache:
            d_rank = int(cache.get('d', cache['E_src'].shape[1]))
            e_v_np = np.zeros((s, d_rank, self.D), dtype=np.float32)
            e_q_np = np.zeros((s, d_rank, self.D), dtype=np.float32)
            for compact, slot in enumerate(self.slot_universe):
                idx = cache['slot_to_idx'][slot] if slot <= cache['max_slot'] else -1
                if idx >= 0:
                    e_v_np[compact] = cache['E_src'][idx]
                    e_q_np[compact] = cache['E_tgt'][idx]
            self.e_v = mx.array(e_v_np)
            self.e_q = mx.array(e_q_np)
            self._a_mode = 'v2'
            return

        self.e_v = None
        self.e_q = None
        self._a_mode = None

    def _sync_dense_b_to_hash(self):
        if self.b_dense is None:
            return

        s = len(self.slot_universe)
        tracked_keys = self._tracked_keys if self._tracked_keys is not None else np.zeros((0,), dtype=np.int64)
        if len(tracked_keys) == 0:
            self.b._hash_table.clear()
            self.b._n_entries = 0
            self.b.n_b = 0
            self._dense_dirty = False
            return

        src_c = (tracked_keys // s).astype(np.int32, copy=False)
        tgt_c = (tracked_keys % s).astype(np.int32, copy=False)
        vecs_np = np.array(self._b_flat[mx.array(tracked_keys)])
        norms = np.linalg.norm(vecs_np, axis=1)
        keep_mask = norms > self.config.EPS_PRUNE

        kept_keys = tracked_keys[keep_mask]
        kept_src_c = src_c[keep_mask]
        kept_tgt_c = tgt_c[keep_mask]
        kept_vecs = vecs_np[keep_mask]

        self.b._hash_table.clear()
        self.b._n_entries = 0
        self.b.n_b = 0
        self.b.global_step += 1
        step = int(self.b.global_step)

        n_needed = len(kept_keys)
        while self.b._capacity < n_needed:
            self.b._expand_capacity()

        for idx in range(n_needed):
            src = int(self.slot_universe[kept_src_c[idx]])
            tgt = int(self.slot_universe[kept_tgt_c[idx]])
            self.b._srcs[idx] = src
            self.b._tgts[idx] = tgt
            self.b._vecs[idx] = kept_vecs[idx]
            self.b._times[idx] = step
            self.b._hash_table[(src, tgt)] = idx
        self.b._n_entries = n_needed
        self.b.n_b = n_needed
        self._tracked_keys = kept_keys
        self._dense_dirty = False

    def sync_to_hash(self):
        if self._dense_dirty:
            t0 = time.time()
            self._sync_dense_b_to_hash()
            self.profile['sync_ms'] = (time.time() - t0) * 1000
        else:
            self.profile['sync_ms'] = 0.0

    def invalidate_dense_state(self):
        self.b_dense = None
        self.e_v = None
        self.e_q = None
        self._a_mode = None
        self._dense_dirty = False
        self._tracked_keys = None
        self._b_flat = None
        self._a_edge_cache = None
        self._a_edge_cache_keys = None
        self._a_edge_cache_cycle = -1

    def dense_stats(self):
        tracked_keys = self._tracked_keys if self._tracked_keys is not None else np.zeros((0,), dtype=np.int64)
        if self.b_dense is None or len(tracked_keys) == 0:
            return {'n_entries': 0, 'norm_mean': 0.0, 'norm_max': 0.0, 'norm_min': 0.0}

        s = len(self.slot_universe)
        src_c = (tracked_keys // s).astype(np.int32, copy=False)
        tgt_c = (tracked_keys % s).astype(np.int32, copy=False)
        vecs = np.array(self._b_flat[mx.array(tracked_keys)])
        norms = np.linalg.norm(vecs, axis=1)
        keep = norms > self.config.EPS_PRUNE
        if not np.any(keep):
            return {'n_entries': 0, 'norm_mean': 0.0, 'norm_max': 0.0, 'norm_min': 0.0}
        norms = norms[keep]
        return {
            'n_entries': int(len(norms)),
            'norm_mean': float(norms.mean()),
            'norm_max': float(norms.max()),
            'norm_min': float(norms.min()),
        }

    def _segment_sum(self, values: object, start_idx: object, end_idx: object):
        prefix = mx.concatenate([mx.zeros((1,), dtype=mx.float32), mx.cumsum(values, axis=0)], axis=0)
        return prefix[end_idx] - prefix[start_idx]

    def _prepare_a_edge_cache(self, dataset: SparseWakeDatasetMLX, edge_batch_size: int = 32768):
        if self.e_v is None or self.e_q is None or self._a_mode is None:
            self._a_edge_cache = None
            self._a_edge_cache_keys = None
            self._a_edge_cache_cycle = -1
            return

        current_cycle = int(getattr(self.a, 'sleep_cycle_count', -1))
        if (
            self._a_edge_cache is not None
            and self._a_edge_cache_keys is not None
            and self._a_edge_cache_cycle == current_cycle
            and np.array_equal(self._a_edge_cache_keys, dataset.unique_keys_np)
        ):
            return

        n_edges = int(dataset.unique_keys_mx.shape[0])
        if n_edges == 0:
            self._a_edge_cache = mx.zeros((0, self.D), dtype=mx.float32)
            self._a_edge_cache_keys = dataset.unique_keys_np.copy()
            self._a_edge_cache_cycle = current_cycle
            return

        chunks = []
        for start in range(0, n_edges, edge_batch_size):
            end = min(start + edge_batch_size, n_edges)
            src_chunk = self.e_v[dataset.src_unique_mx[start:end]]
            tgt_chunk = self.e_q[dataset.tgt_unique_mx[start:end]]
            if self._a_mode == 'v2':
                edge_chunk = mx.sum(src_chunk * tgt_chunk, axis=1)
            else:
                edge_chunk = src_chunk * tgt_chunk
            chunks.append(edge_chunk)

        if len(chunks) == 1:
            self._a_edge_cache = chunks[0]
        else:
            self._a_edge_cache = mx.concatenate(chunks, axis=0)
        mx.eval(self._a_edge_cache)
        self._a_edge_cache_keys = dataset.unique_keys_np.copy()
        self._a_edge_cache_cycle = current_cycle

    def run_round(self, dataset: SparseWakeDatasetMLX, verbose: bool = False) -> Dict[str, float]:
        # per-position A strength 分支
        _a_mode = getattr(self, 'a_strength_mode', 'edge')
        if _a_mode == 'position' and self.a.has_knowledge:
            a_word_gold = self.compute_a_word_per_position(dataset)
            return self.run_round_word_gap(dataset, a_word_gold, verbose=verbose)

        t0 = time.time()
        if self.b_dense is None:
            self._init_dense_b()
        t_init = time.time()
        self._prepare_a_dense()
        t_a_dense = time.time()
        self._prepare_a_edge_cache(dataset)
        t_a = time.time()

        t_prop = time.time()
        s = len(self.slot_universe)
        n_unique = int(dataset.count_mx.shape[0])
        if n_unique > 0:
            b_vecs = self._b_flat[dataset.unique_keys_mx]
            b_strength = mx.sum(b_vecs * dataset.phi_mean_mx, axis=1)
            if self._a_edge_cache is not None:
                a_strength = mx.sum(self._a_edge_cache * dataset.phi_mean_mx, axis=1)
                # a_strength 不再除以 d_rank (已在 A predict 中移除归一化)
            else:
                a_strength = mx.zeros((n_unique,), dtype=mx.float32)
            _lam = getattr(self, 'namida', None)
            _gap_tgt = getattr(self, 'gap_target', 1.0)
            if _lam is not None:
                gap_mean = _gap_tgt - ((1 - _lam) * a_strength + _lam * b_strength)
            else:
                gap_mean = _gap_tgt - (a_strength + b_strength)
            mx.eval(gap_mean)
        else:
            gap_mean = mx.zeros((0,), dtype=mx.float32)
        shard_compute_ms = (time.time() - t_prop) * 1000

        t_merge = time.time()
        if n_unique > 0:
            delta = self.b._eta * gap_mean[:, None] * dataset.phi_mean_mx
            self._b_flat = self._b_flat.at[dataset.unique_keys_mx].add(delta)
            self.b_dense = self._b_flat.reshape((s, s, self.D))
            mx.eval(self._b_flat)
            if self._tracked_keys is None or len(self._tracked_keys) == 0:
                self._tracked_keys = dataset.unique_keys_np.copy()
            self._dense_dirty = True

        t_end = time.time()
        info = {
            'n_shards': float(dataset.n_shards),
            'n_shard_unique_total': float(n_unique),
            'n_unique': float(n_unique),
            'init_b_ms': (t_init - t0) * 1000,
            'prep_a_dense_ms': (t_a_dense - t_init) * 1000,
            'prep_a_edge_ms': (t_a - t_a_dense) * 1000,
            'prep_a_ms': (t_a - t_init) * 1000,
            'shard_compute_ms': shard_compute_ms,
            'final_merge_ms': (t_end - t_merge) * 1000,
            'round_total_ms': (t_end - t0) * 1000,
        }
        self.profile['last_round'] = info
        if verbose:
            print(
                f"  mlx-sparse-round: shards={dataset.n_shards}, "
                f"merged_edges={n_unique}, round_ms={info['round_total_ms']:.1f}"
            )
        return info

    def compute_a_word_per_position(self, dataset: SparseWakeDatasetMLX) -> object:
        """计算每个 position 的 word-level A score for gold target。

        返回 mx.array shape (n_positions,)，其中每个元素是
        A_word(gold_word) at that position。
        """
        self._prepare_a_dense()
        p = int(dataset.total_positions)
        s = len(self.slot_universe)
        v_proj = self._prepare_word_projection_cached(dataset)
        gold_idx = self._gold_word_idx_cached(dataset)

        # 用 evaluator 同样的逻辑计算 A word scores per position
        active_counts_np = np.bincount(
            np.array(dataset.active_pos_sorted_mx, copy=False).astype(np.int32, copy=False),
            minlength=p
        ).astype(np.float32, copy=False)
        active_counts_np = np.maximum(active_counts_np, 1.0)
        active_counts_mx = mx.array(active_counts_np[:, None])

        word_scores_a = mx.zeros((p, v_proj.shape[0]), dtype=mx.float32)
        for idx, src in enumerate(dataset.active_src_unique_np.tolist()):
            start_i = int(np.array(dataset.active_src_start_mx[idx:idx + 1])[0])
            end_i = int(np.array(dataset.active_src_end_mx[idx:idx + 1])[0])
            pos_rows = dataset.active_pos_sorted_mx[start_i:end_i]
            phi_rows = dataset.active_phi_sorted_mx[start_i:end_i]

            if self.e_v is not None and self._a_mode == 'v2':
                n_rows = int(phi_rows.shape[0])
                src_embed = self.e_v[int(src)][None, :, :]
                ctx_flat = mx.reshape(src_embed * phi_rows[:, None, :],
                                      (n_rows, int(self._e_tgt_flat.shape[1])))
                slot_scores = mx.matmul(ctx_flat, mx.transpose(self._e_tgt_flat))
                word_scores_rows = mx.matmul(slot_scores, self._word_proj_t)
                word_scores_a = word_scores_a.at[pos_rows].add(word_scores_rows)
            elif self.e_v is not None and self._a_mode == 'legacy':
                ctx_rows = self.e_v[int(src)][None, :] * phi_rows
                a_word_mx = mx.matmul(mx.transpose(self.e_q), self._word_proj_t)
                word_scores_rows = mx.matmul(ctx_rows, a_word_mx)
                word_scores_a = word_scores_a.at[pos_rows].add(word_scores_rows)

        if self._a_mode == 'v2' and self.e_q is not None:
            word_scores_a = word_scores_a / active_counts_mx
        elif self._a_mode == 'legacy':
            d_rank = float(max(1, getattr(self.a, 'd', 1)))
            word_scores_a = word_scores_a / (active_counts_mx * d_rank)

        mx.eval(word_scores_a)

        # word-level 归一化：每个 position 的 top-1 word score 拉到 1.0
        if getattr(self, 'normalize_a_word', False):
            a_max = mx.max(word_scores_a, axis=1, keepdims=True)
            a_max = mx.maximum(a_max, 1e-8)
            word_scores_a = word_scores_a / a_max
            mx.eval(word_scores_a)

        # 提取每个 position 的 gold word score
        gold_idx_mx = mx.array(gold_idx)
        a_word_gold = mx.zeros((p,), dtype=mx.float32)
        for pos_i in range(p):
            gi = int(gold_idx[pos_i])
            if gi >= 0:
                a_word_gold = a_word_gold.at[pos_i].add(word_scores_a[pos_i, gi])
        mx.eval(a_word_gold)
        return a_word_gold

    def _prepare_word_projection_cached(self, dataset):
        """缓存 word projection matrix。"""
        if hasattr(self, '_word_proj_mx') and self._word_proj_mx is not None:
            return self._word_proj_mx
        from .inference_fast import SparseMatrixDecoder
        if self.model._decoder_dirty or self.model._decoder is None:
            self.model._decoder = SparseMatrixDecoder(
                self.model.vocab, self.model.word_to_slots, self.K)
            self.model._decoder_dirty = False
        decoder = self.model._decoder
        vocab_list = decoder.word_list
        v = len(vocab_list)
        s = len(self.slot_universe)
        stc = {int(slot): idx for idx, slot in enumerate(self.slot_universe.tolist())}
        proj_np = np.zeros((v, s), dtype=np.float32)
        for wi, w in enumerate(vocab_list):
            slots = self.model.word_to_slots[w]
            compact = sorted(stc[int(sl)] for sl in slots if int(sl) in stc)
            if compact:
                proj_np[wi, compact] = 1.0 / len(compact)
        self._word_proj_mx = mx.array(proj_np)
        self._word_proj_t = mx.transpose(self._word_proj_mx)
        # 也缓存 e_tgt_flat
        if self._a_mode == 'v2' and self.e_q is not None:
            shape = self.e_q.shape
            self._e_tgt_flat = mx.reshape(self.e_q, (int(shape[0]), int(shape[1]) * int(shape[2])))
        else:
            self._e_tgt_flat = None
        return self._word_proj_mx

    def _gold_word_idx_cached(self, dataset):
        """缓存 gold word index per position。"""
        if hasattr(self, '_gold_idx_np') and self._gold_idx_np is not None:
            return self._gold_idx_np
        from .inference_fast import SparseMatrixDecoder
        if self.model._decoder_dirty or self.model._decoder is None:
            self.model._decoder = SparseMatrixDecoder(
                self.model.vocab, self.model.word_to_slots, self.K)
            self.model._decoder_dirty = False
        decoder = self.model._decoder
        vocab_list = decoder.word_list
        s = len(self.slot_universe)
        stc = {int(slot): idx for idx, slot in enumerate(self.slot_universe.tolist())}
        gold_by_tgt = {}
        for wi, w in enumerate(vocab_list):
            slots = self.model.word_to_slots[w]
            compact = sorted(stc[int(sl)] for sl in slots if int(sl) in stc)
            if compact:
                gold_by_tgt[tuple(compact)] = wi

        target_offsets = dataset.target_offsets_np
        target_compact = dataset.target_compact_np
        gold = np.full((dataset.total_positions,), -1, dtype=np.int32)
        for pos in range(dataset.total_positions):
            t0, t1 = int(target_offsets[pos]), int(target_offsets[pos + 1])
            gold[pos] = gold_by_tgt.get(tuple(sorted(target_compact[t0:t1].tolist())), -1)
        self._gold_idx_np = gold
        return gold

    def compute_a_word_mean_per_edge(self, dataset: SparseWakeDatasetMLX,
                                      a_word_gold: object) -> object:
        """把 per-position A_word(gold) 聚合为 per-unique-edge 的均值。

        用 prefix sum 高效计算。
        """
        if dataset.pos_ids_sorted_mx is None or dataset.total_pairs == 0:
            return mx.zeros((0,), dtype=mx.float32)
        # a_word_gold: (n_positions,)
        # pos_ids_sorted_mx: (total_pairs,) - 每个 raw pair 对应的 position
        a_word_per_pair = a_word_gold[dataset.pos_ids_sorted_mx]
        # prefix sum 聚合
        prefix = mx.concatenate([
            mx.zeros((1,), dtype=mx.float32),
            mx.cumsum(a_word_per_pair, axis=0),
        ], axis=0)
        a_word_sum = prefix[dataset.end_idx_mx] - prefix[dataset.start_idx_mx]
        a_word_mean = a_word_sum / dataset.count_mx
        mx.eval(a_word_mean)
        return a_word_mean

    def run_round_word_gap(self, dataset: SparseWakeDatasetMLX,
                           a_word_gold: object,
                           verbose: bool = False) -> Dict[str, float]:
        """Word-level gap 写入。

        你的方案：
        1. 计算 A 的 word-level gold score
        2. word_gap = 1.0 - A_word_gold
        3. 对 gold word 的每个 slot，写入目标 = word_gap

        每个 edge (src, tgt) 的标准 wake 写入:
          target = word_gap(pos)           # word-level 目标
          current = B[src,tgt]·φ           # 当前 slot-level B 强度
          delta = η * (target - current) * φ

        这样 B 的 slot 会收敛到 word_gap 而不会无限叠加。
        """
        t0 = time.time()
        if self.b_dense is None:
            self._init_dense_b()

        s = len(self.slot_universe)
        D = self.D
        n_pairs = int(dataset.total_pairs)
        n_unique = int(dataset.count_mx.shape[0])
        if n_pairs > 0 and n_unique > 0:
            # per-pair B strength (current)
            b_vecs = self._b_flat[dataset.pair_keys_mx]            # (n_pairs, D)
            b_strength = mx.sum(b_vecs * dataset.phi_mx, axis=1)   # (n_pairs,)

            # per-pair A word-level gold score → word_gap (target)
            a_word_per_pair = a_word_gold[dataset.pos_ids_sorted_mx]  # (n_pairs,)
            _gap_tgt = getattr(self, 'gap_target', 1.0)
            word_gap = _gap_tgt - a_word_per_pair                     # (n_pairs,)

            # delta = η * (target - current) * φ
            gap = word_gap - b_strength                              # (n_pairs,)
            weighted_delta = gap[:, None] * dataset.phi_mx           # (n_pairs, D)
            abs_gap = mx.abs(gap)                                    # (n_pairs,)

            # scatter-add 聚合到 unique edge
            counts_int = np.array(dataset.count_mx).astype(np.int32)
            edge_idx = mx.array(
                np.repeat(np.arange(n_unique, dtype=np.int32), counts_int)
            )                                                        # (n_pairs,)
            sum_gap_phi = mx.zeros((n_unique, D), dtype=mx.float32)
            sum_gap_phi = sum_gap_phi.at[edge_idx].add(weighted_delta)
            sum_abs_gap = mx.zeros((n_unique,), dtype=mx.float32)
            sum_abs_gap = sum_abs_gap.at[edge_idx].add(abs_gap)

            # |gap|-weighted average per unique edge
            eps = 1e-8
            delta_edge = self.b._eta * sum_gap_phi / (sum_abs_gap[:, None] + eps)

            self._b_flat = self._b_flat.at[dataset.unique_keys_mx].add(delta_edge)
            self.b_dense = self._b_flat.reshape((s, s, D))
            mx.eval(self._b_flat)
            if self._tracked_keys is None or len(self._tracked_keys) == 0:
                self._tracked_keys = dataset.unique_keys_np.copy()
            self._dense_dirty = True

        t_end = time.time()
        info = {
            'n_pairs': float(n_pairs),
            'round_total_ms': (t_end - t0) * 1000,
        }
        self.profile['last_round'] = info
        if verbose:
            print(f"  word-gap-round: pairs={n_pairs}, unique={n_unique}, "
                  f"ms={info['round_total_ms']:.1f}")
        return info

    def run_round_position_gap(self, dataset: SparseWakeDatasetMLX,
                                verbose: bool = False) -> Dict[str, float]:
        """Position-level gap wake.

        1. 用 evaluator 逻辑计算每个 position 的 A+B word score
        2. gap(pos) = 1 - score(gold_word, pos)
        3. 对该 position 的每个 active src:
             delta = η * gap / n_active * φ
             写入 B[src, each_gold_slot]
        """
        t0 = time.time()
        if self.b_dense is None:
            self._init_dense_b()

        s = len(self.slot_universe)
        D = self.D
        p = int(dataset.total_positions)

        # -- 计算 A+B word score per position (evaluator-style) --
        word_proj = self._prepare_word_projection_cached(dataset)  # (v, s)
        v = int(word_proj.shape[0])
        gold_idx = self._gold_word_idx_cached(dataset)             # (p,) int32

        # active counts per position
        active_counts_np = np.bincount(
            np.array(dataset.active_pos_sorted_mx, copy=False).astype(np.int32),
            minlength=p
        ).astype(np.float32)
        active_counts_np = np.maximum(active_counts_np, 1.0)
        active_counts_mx = mx.array(active_counts_np[:, None])  # (p, 1)

        # B word scores
        b_src_d_t = mx.transpose(self._b_flat.reshape((s, s, D)), (0, 2, 1))  # (s, D, s)
        b_word = mx.matmul(b_src_d_t, mx.transpose(word_proj))  # (s, D, v)

        word_scores_b = mx.zeros((p, v), dtype=mx.float32)
        for idx, src in enumerate(dataset.active_src_unique_np.tolist()):
            si = int(np.array(dataset.active_src_start_mx[idx:idx + 1])[0])
            ei = int(np.array(dataset.active_src_end_mx[idx:idx + 1])[0])
            pos_rows = dataset.active_pos_sorted_mx[si:ei]
            phi_rows = dataset.active_phi_sorted_mx[si:ei]
            row_ws = mx.matmul(phi_rows, b_word[int(src)])  # (n, v)
            word_scores_b = word_scores_b.at[pos_rows].add(row_ws)
        word_scores_b = word_scores_b / active_counts_mx

        # A word scores
        self._prepare_a_dense()
        word_scores_a = mx.zeros((p, v), dtype=mx.float32)
        if self._a_mode == 'v2' and self.e_q is not None:
            e_tgt_flat = self._e_tgt_flat if hasattr(self, '_e_tgt_flat') and self._e_tgt_flat is not None \
                else mx.reshape(self.e_q, (int(self.e_q.shape[0]),
                                           int(self.e_q.shape[1]) * int(self.e_q.shape[2])))
            for idx, src in enumerate(dataset.active_src_unique_np.tolist()):
                si = int(np.array(dataset.active_src_start_mx[idx:idx + 1])[0])
                ei = int(np.array(dataset.active_src_end_mx[idx:idx + 1])[0])
                pos_rows = dataset.active_pos_sorted_mx[si:ei]
                phi_rows = dataset.active_phi_sorted_mx[si:ei]
                n_rows = int(phi_rows.shape[0])
                src_embed = self.e_v[int(src)][None, :, :]
                ctx_flat = mx.reshape(src_embed * phi_rows[:, None, :],
                                      (n_rows, int(e_tgt_flat.shape[1])))
                slot_scores = mx.matmul(ctx_flat, mx.transpose(e_tgt_flat))
                ws_a_rows = mx.matmul(slot_scores, mx.transpose(word_proj))
                word_scores_a = word_scores_a.at[pos_rows].add(ws_a_rows)
            word_scores_a = word_scores_a / active_counts_mx
        elif self.e_v is not None:
            ctx = mx.zeros((p, D), dtype=mx.float32)
            for idx, src in enumerate(dataset.active_src_unique_np.tolist()):
                si = int(np.array(dataset.active_src_start_mx[idx:idx + 1])[0])
                ei = int(np.array(dataset.active_src_end_mx[idx:idx + 1])[0])
                pos_rows = dataset.active_pos_sorted_mx[si:ei]
                phi_rows = dataset.active_phi_sorted_mx[si:ei]
                ctx_rows = self.e_v[int(src)][None, :] * phi_rows
                ctx = ctx.at[pos_rows].add(ctx_rows)
            a_word_mat = mx.matmul(mx.transpose(self.e_q), mx.transpose(word_proj))
            word_scores_a = mx.matmul(ctx, a_word_mat)
            d_rank = float(max(1, getattr(self.model.a, 'd', 1)))
            word_scores_a = word_scores_a / (active_counts_mx * d_rank)

        # total score and gap
        total_scores = word_scores_a + word_scores_b  # (p, v)
        gold_idx_mx = mx.array(gold_idx)
        valid_mask = gold_idx_mx >= 0

        # gold score per position
        # Gather: total_scores[pos, gold_idx[pos]]
        gold_scores = mx.zeros((p,), dtype=mx.float32)
        total_np = np.array(total_scores)
        gold_np = gold_idx
        for pos_i in range(p):
            if gold_np[pos_i] >= 0:
                gold_scores = gold_scores.at[pos_i].add(
                    mx.array(total_np[pos_i, gold_np[pos_i]]))
        mx.eval(gold_scores)

        _gap_tgt = getattr(self, 'gap_target', 1.0)
        gap = mx.where(valid_mask, _gap_tgt - gold_scores, 0.0)  # (p,)
        mx.eval(gap)

        t_score = time.time()

        # -- 写入 B: 对每个 active (pos, src, phi) 写到 gold slots --
        # 预计算 gold slots per position
        target_offsets = dataset.target_offsets_np  # (p+1,) or (v+1,)?
        target_compact = dataset.target_compact_np

        # target_offsets 索引的是 word list, 不是 position!
        # gold slots = slots of gold_word = 通过 word_proj 映射
        # 直接用 word_to_slots 获取 gold slots
        from .inference_fast import SparseMatrixDecoder
        if self.model._decoder_dirty or self.model._decoder is None:
            self.model._decoder = SparseMatrixDecoder(
                self.model.vocab, self.model.word_to_slots, self.K)
            self.model._decoder_dirty = False
        decoder = self.model._decoder
        vocab_list = decoder.word_list
        stc = {int(slot): idx for idx, slot in enumerate(self.slot_universe.tolist())}

        # gold_slot_indices per position: list of compact slot indices
        gold_slot_lists = []
        for pos_i in range(p):
            gi = gold_np[pos_i]
            if gi >= 0 and gi < len(vocab_list):
                w = vocab_list[gi]
                slots = self.model.word_to_slots[w]
                compact = [stc[int(sl)] for sl in slots if int(sl) in stc]
                gold_slot_lists.append(compact)
            else:
                gold_slot_lists.append([])

        # 展开 active rows → write operations
        # active_pos_sorted: (n_active_rows,) - position per row
        # active_phi_sorted: (n_active_rows, D) - phi per row
        # For each active row i at pos[i] with source src:
        #   for each gold_slot_j of pos[i]:
        #     key = src * s + gold_slot_j
        #     delta = eta * gap[pos[i]] / n_active[pos[i]] * phi[i]

        gap_np = np.array(gap)
        n_active_np = active_counts_np  # (p,)
        active_pos_np = np.array(dataset.active_pos_sorted_mx)
        active_phi_np = np.array(dataset.active_phi_sorted_mx)

        # 构建 src index per active row (from grouped structure)
        src_per_row = np.empty(len(active_pos_np), dtype=np.int32)
        for idx, src_slot_idx in enumerate(dataset.active_src_unique_np.tolist()):
            si = int(np.array(dataset.active_src_start_mx[idx:idx + 1])[0])
            ei = int(np.array(dataset.active_src_end_mx[idx:idx + 1])[0])
            src_per_row[si:ei] = src_slot_idx

        # 展开: 每个 active row × gold slots
        all_keys = []
        all_deltas = []
        eta = self.b._eta

        for i in range(len(active_pos_np)):
            pos_i = int(active_pos_np[i])
            g = gap_np[pos_i]
            if abs(g) < 1e-8:
                continue
            n_act = n_active_np[pos_i]
            src_idx = src_per_row[i]
            phi = active_phi_np[i]
            gold_slots = gold_slot_lists[pos_i]
            if not gold_slots:
                continue
            delta = eta * g / n_act * phi  # (D,)
            for tgt_idx in gold_slots:
                all_keys.append(src_idx * s + tgt_idx)
                all_deltas.append(delta)

        t_expand = time.time()

        if all_keys:
            # B 全局衰减：写入前衰减，防止范数单调增长
            lambda_b = self.b._lambda_b if hasattr(self.b, '_lambda_b') else 3e-4
            self._b_flat = self._b_flat * (1.0 - lambda_b)
            keys_mx = mx.array(np.array(all_keys, dtype=np.int64))
            deltas_mx = mx.array(np.array(all_deltas, dtype=np.float32))
            self._b_flat = self._b_flat.at[keys_mx].add(deltas_mx)
            self.b_dense = self._b_flat.reshape((s, s, D))
            mx.eval(self._b_flat)
            if self._tracked_keys is None or len(self._tracked_keys) == 0:
                self._tracked_keys = dataset.unique_keys_np.copy()
            self._dense_dirty = True

        t_end = time.time()
        info = {
            'score_ms': (t_score - t0) * 1000,
            'expand_ms': (t_expand - t_score) * 1000,
            'write_ms': (t_end - t_expand) * 1000,
            'round_total_ms': (t_end - t0) * 1000,
            'n_writes': len(all_keys),
        }
        self.profile['last_round'] = info
        if verbose:
            gap_valid = gap_np[gold_idx >= 0]
            print(f"  pos-gap-round: writes={len(all_keys)}, "
                  f"gap_mean={gap_valid.mean():.4f}, ms={info['round_total_ms']:.1f}")
        return info

    def run_corpus_parallel(self, dataset: SparseWakeDatasetMLX, rounds: int, shard_size: int = 4096, verbose: bool = False, sync_at_end: bool = True):
        history = []
        for round_idx in range(rounds):
            info = self.run_round(dataset, verbose=verbose)
            info['round'] = float(round_idx + 1)
            history.append(info)
        if sync_at_end:
            self.sync_to_hash()
        return history
