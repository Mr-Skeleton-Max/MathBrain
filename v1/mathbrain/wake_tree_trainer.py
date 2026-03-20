"""树形 / 语料级并行 Wake 实验器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .wake_dataset import WakeDataset, WakeShard


@dataclass
class FrozenBSnapshot:
    keys_sorted: np.ndarray
    vecs_sorted: np.ndarray
    step: int


@dataclass
class MergeStats:
    keys: np.ndarray
    phi_sum: np.ndarray
    gap_sum: np.ndarray
    count: np.ndarray


class TreeWakeTrainer:
    """冻结 B / A 后，对整份语料做 proposal + tree merge + round update。"""

    def __init__(self, model):
        self.model = model
        self.b = model.b
        self.a = model.a
        self.config = model.config
        self.K = model.config.K
        self.D = model.config.d

    def build_dataset(self, corpus: List[str], verbose: bool = True) -> WakeDataset:
        return WakeDataset.from_corpus(self.model, corpus, verbose=verbose)

    def sync_to_hash(self):
        return

    def invalidate_dense_state(self):
        return

    def export_frozen_b_snapshot(self) -> FrozenBSnapshot:
        step = int(self.b.global_step + 1)
        n = int(self.b._n_entries)
        if n == 0:
            return FrozenBSnapshot(
                keys_sorted=np.zeros((0,), dtype=np.int64),
                vecs_sorted=np.zeros((0, self.D), dtype=np.float32),
                step=step,
            )

        srcs = self.b._srcs[:n].astype(np.int64, copy=False)
        tgts = self.b._tgts[:n].astype(np.int64, copy=False)
        keys = srcs * self.K + tgts

        vecs = self.b._vecs[:n].copy()
        times = self.b._times[:n]
        dts = step - times
        decay_mask = (dts > 0) & (dts < self.b._max_dt)
        if np.any(decay_mask):
            vecs[decay_mask] *= self.b._decay_table[dts[decay_mask]][:, None]

        order = np.argsort(keys, kind='mergesort')
        return FrozenBSnapshot(
            keys_sorted=keys[order],
            vecs_sorted=vecs[order],
            step=step,
        )

    def _prepare_a_cache(self):
        if self.a.has_knowledge:
            self.b._precompute_a_embeddings(self.a)

    def _expand_shard_pairs(self, shard: WakeShard):
        n_positions = shard.n_positions
        active_counts = np.diff(shard.active_offsets)
        target_counts = np.diff(shard.target_offsets)
        pair_counts = active_counts * target_counts
        total_pairs = int(pair_counts.sum())

        all_src = np.empty((total_pairs,), dtype=np.int32)
        all_tgt = np.empty((total_pairs,), dtype=np.int32)
        all_phi = np.empty((total_pairs, self.D), dtype=np.float32)

        cursor = 0
        for pos in range(n_positions):
            a0, a1 = int(shard.active_offsets[pos]), int(shard.active_offsets[pos + 1])
            t0, t1 = int(shard.target_offsets[pos]), int(shard.target_offsets[pos + 1])
            active_slots = shard.active_slots[a0:a1]
            phi_hat = shard.phi_hat[a0:a1]
            target_slots = shard.target_slots[t0:t1]
            n_src = len(active_slots)
            n_tgt = len(target_slots)
            if n_src == 0 or n_tgt == 0:
                continue
            n_pairs = n_src * n_tgt
            end = cursor + n_pairs
            all_src[cursor:end] = np.repeat(active_slots, n_tgt)
            all_tgt[cursor:end] = np.tile(target_slots, n_src)
            all_phi[cursor:end] = np.repeat(phi_hat, n_tgt, axis=0)
            cursor = end

        return all_src[:cursor], all_tgt[:cursor], all_phi[:cursor]

    def _compute_b_strength_from_snapshot(
        self,
        all_src: np.ndarray,
        all_tgt: np.ndarray,
        phi_hat: np.ndarray,
        snapshot: FrozenBSnapshot,
    ) -> np.ndarray:
        n = len(all_src)
        if n == 0 or len(snapshot.keys_sorted) == 0:
            return np.zeros((n,), dtype=np.float32)

        query_keys = all_src.astype(np.int64) * self.K + all_tgt.astype(np.int64)
        idx = np.searchsorted(snapshot.keys_sorted, query_keys)
        matched = (idx < len(snapshot.keys_sorted)) & (snapshot.keys_sorted[idx] == query_keys)
        out = np.zeros((n,), dtype=np.float32)
        if np.any(matched):
            matched_vecs = snapshot.vecs_sorted[idx[matched]]
            out[matched] = np.sum(matched_vecs * phi_hat[matched], axis=1)
        return out

    def propose_shard(self, shard: WakeShard, snapshot: FrozenBSnapshot) -> Optional[MergeStats]:
        all_src, all_tgt, phi_hat = self._expand_shard_pairs(shard)
        if len(all_src) == 0:
            return None

        b_strength = self._compute_b_strength_from_snapshot(all_src, all_tgt, phi_hat, snapshot)
        if self.a.has_knowledge:
            a_strength = self.b._compute_a_strength_vectorized(all_src, all_tgt, phi_hat)
            pass  # a_strength 不再除以 d_rank
        else:
            a_strength = np.zeros_like(b_strength)

        gaps = 1.0 - (a_strength + b_strength)
        keys = all_src.astype(np.int64) * self.K + all_tgt.astype(np.int64)
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

        if len(unique_keys) == len(keys):
            phi_sum = phi_hat.copy()
            gap_sum = gaps.copy()
        else:
            order = np.argsort(inverse, kind='mergesort')
            inv_sorted = inverse[order]
            split_idx = np.flatnonzero(np.diff(inv_sorted)) + 1
            group_starts = np.concatenate(([0], split_idx))
            phi_sum = np.add.reduceat(phi_hat[order], group_starts, axis=0)
            gap_sum = np.add.reduceat(gaps[order], group_starts)

        return MergeStats(
            keys=unique_keys.astype(np.int64, copy=False),
            phi_sum=phi_sum.astype(np.float32, copy=False),
            gap_sum=gap_sum.astype(np.float32, copy=False),
            count=counts.astype(np.int32, copy=False),
        )

    def merge_stat_pair(self, left: MergeStats, right: MergeStats) -> MergeStats:
        all_keys = np.concatenate([left.keys, right.keys])
        all_phi_sum = np.concatenate([left.phi_sum, right.phi_sum], axis=0)
        all_gap_sum = np.concatenate([left.gap_sum, right.gap_sum], axis=0)
        all_count = np.concatenate([left.count, right.count], axis=0)

        unique_keys, inverse = np.unique(all_keys, return_inverse=True)
        if len(unique_keys) == len(all_keys):
            return MergeStats(unique_keys, all_phi_sum, all_gap_sum, all_count)

        order = np.argsort(inverse, kind='mergesort')
        inv_sorted = inverse[order]
        split_idx = np.flatnonzero(np.diff(inv_sorted)) + 1
        group_starts = np.concatenate(([0], split_idx))

        phi_sum = np.add.reduceat(all_phi_sum[order], group_starts, axis=0)
        gap_sum = np.add.reduceat(all_gap_sum[order], group_starts)
        count_sum = np.add.reduceat(all_count[order], group_starts)
        return MergeStats(unique_keys, phi_sum, gap_sum, count_sum)

    def tree_merge(self, stats_list: List[MergeStats]) -> Optional[MergeStats]:
        stats_list = [stats for stats in stats_list if stats is not None and len(stats.keys) > 0]
        if not stats_list:
            return None
        while len(stats_list) > 1:
            next_level = []
            for idx in range(0, len(stats_list), 2):
                if idx + 1 < len(stats_list):
                    next_level.append(self.merge_stat_pair(stats_list[idx], stats_list[idx + 1]))
                else:
                    next_level.append(stats_list[idx])
            stats_list = next_level
        return stats_list[0]

    def apply_merged_stats(self, merged: MergeStats):
        if merged is None or len(merged.keys) == 0:
            return

        self.b.global_step += 1
        step = int(self.b.global_step)
        phi_mean = merged.phi_sum / merged.count[:, None]
        gap_mean = merged.gap_sum / merged.count
        delta = self.b._eta * gap_mean[:, None] * phi_mean

        unique_src = (merged.keys // self.K).astype(np.int32)
        unique_tgt = (merged.keys % self.K).astype(np.int32)

        for i in range(len(unique_src)):
            src = int(unique_src[i])
            tgt = int(unique_tgt[i])
            key = (src, tgt)
            idx = self.b._hash_table.get(key)
            if idx is not None:
                dt = step - self.b._times[idx]
                if 0 < dt < self.b._max_dt:
                    self.b._vecs[idx] *= self.b._decay_table[dt]
                self.b._vecs[idx] += delta[i]
                self.b._times[idx] = step
            else:
                if self.b._n_entries >= self.b._capacity:
                    self.b._expand_capacity()
                idx = self.b._n_entries
                self.b._srcs[idx] = src
                self.b._tgts[idx] = tgt
                self.b._vecs[idx] = delta[i]
                self.b._times[idx] = step
                self.b._hash_table[key] = idx
                self.b._n_entries += 1
                self.b.n_b += 1

    def run_round(
        self,
        dataset: WakeDataset,
        shard_size: int = 4096,
        verbose: bool = False,
    ) -> Dict[str, float]:
        self._prepare_a_cache()
        snapshot = self.export_frozen_b_snapshot()
        stats_list = []
        n_shards = 0
        for shard in dataset.iter_shards(shard_size):
            stats = self.propose_shard(shard, snapshot)
            if stats is not None:
                stats_list.append(stats)
            n_shards += 1
        merged = self.tree_merge(stats_list)
        n_unique = 0 if merged is None else len(merged.keys)
        self.apply_merged_stats(merged)
        if verbose:
            print(f"  round: shards={n_shards}, merged_edges={n_unique}")
        return {
            'n_shards': float(n_shards),
            'n_unique': float(n_unique),
        }

    def run_corpus_parallel(
        self,
        dataset: WakeDataset,
        rounds: int,
        shard_size: int = 4096,
        verbose: bool = False,
        sync_at_end: bool = True,
    ) -> List[Dict[str, float]]:
        history = []
        for round_idx in range(rounds):
            info = self.run_round(dataset, shard_size=shard_size, verbose=verbose)
            info['round'] = float(round_idx + 1)
            history.append(info)
        return history
