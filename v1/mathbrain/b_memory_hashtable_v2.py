"""优化版哈希表B Memory：向量化批量查找

关键改进：
1. 预先批量查找所有keys
2. 向量化衰减和点积计算
3. 消除Python循环
"""

import numpy as np
from typing import List, Dict
from .config import MathBrainConfig


class BMemoryEnergyHashTableV2:
    """B Memory with vectorized hash table lookup"""

    def __init__(self, config: MathBrainConfig = None):
        cfg = config or MathBrainConfig()
        self.K = cfg.K
        self.D = cfg.d  # phi 维度（根据 PHI_MODE 自动计算：random=D_PHI, fourier=D_COSINE）
        self.d = cfg.CP_RANK  # A 模块 CP rank
        self._eta = cfg.ETA
        self._lambda_b = cfg.LAMBDA_B
        self._conservation = cfg.B_ENERGY_CONSERVATION  # 守恒开关
        self._gamma_decay = cfg.GAMMA_DECAY  # Sleep衰减因子

        # 哈希表：key=(src, tgt) → index
        self._hash_table = {}

        # 数据存储
        self._capacity = 100000
        self._n_entries = 0

        self._srcs = np.empty(self._capacity, dtype=np.int32)
        self._tgts = np.empty(self._capacity, dtype=np.int32)
        self._vecs = np.empty((self._capacity, self.D), dtype=np.float32)
        self._times = np.empty(self._capacity, dtype=np.int32)

        # Decay table
        self._max_dt = cfg.DECAY_TABLE_SIZE
        self._decay_table = cfg.decay_table

        self.global_step = 0
        self.n_b = 0

        # 裁剪参数
        self._prune_keep_ratio = 0.8
        self._prune_max_entries = 200000

        # A cache
        self._a_cache = None
        self._a_cache_dirty = True
        self._a_cache_cycle = -1

    def _expand_capacity(self):
        """扩容"""
        new_cap = self._capacity * 2
        new_srcs = np.empty(new_cap, dtype=np.int32)
        new_tgts = np.empty(new_cap, dtype=np.int32)
        new_vecs = np.empty((new_cap, self.D), dtype=np.float32)
        new_times = np.empty(new_cap, dtype=np.int32)

        new_srcs[:self._n_entries] = self._srcs[:self._n_entries]
        new_tgts[:self._n_entries] = self._tgts[:self._n_entries]
        new_vecs[:self._n_entries] = self._vecs[:self._n_entries]
        new_times[:self._n_entries] = self._times[:self._n_entries]

        self._srcs = new_srcs
        self._tgts = new_tgts
        self._vecs = new_vecs
        self._times = new_times
        self._capacity = new_cap

    def _precompute_a_embeddings(self, a_knowledge):
        """预计算A的嵌入矩阵"""
        # 兼容旧 checkpoint：历史对象可能没有这些新字段
        if not hasattr(self, "_a_cache"):
            self._a_cache = None
        if not hasattr(self, "_a_cache_dirty"):
            self._a_cache_dirty = True
        if not hasattr(self, "_a_cache_cycle"):
            self._a_cache_cycle = -1

        if not a_knowledge.has_knowledge:
            return

        current_cycle = getattr(a_knowledge, "sleep_cycle_count", -1)
        if (not self._a_cache_dirty and
                self._a_cache is not None and
                self._a_cache_cycle == current_cycle):
            return

        # 兼容 v2 (E_src/E_tgt dict)、v1 独立嵌入 (E_Q/E_V dict)、旧版共享嵌入 (E + W_Q/W_V)
        if hasattr(a_knowledge, 'E_src') and isinstance(getattr(a_knowledge, 'E_src', None), dict) and a_knowledge.E_src:
            all_slots = sorted(a_knowledge.E_src.keys())
        elif hasattr(a_knowledge, 'E_Q') and a_knowledge.E_Q:
            all_slots = sorted(a_knowledge.E_Q.keys())
        elif hasattr(a_knowledge, 'E') and a_knowledge.E:
            all_slots = sorted(a_knowledge.E.keys())
        else:
            self._a_cache = None
            return

        if not all_slots:
            self._a_cache = None
            return

        max_slot = max(all_slots)
        n_slots = len(all_slots)

        slot_to_idx_array = np.full(max_slot + 1, -1, dtype=np.int32)
        slot_to_idx_array[all_slots] = np.arange(n_slots, dtype=np.int32)

        # v2: E_src/E_tgt dict → (n_slots, d, D) 矩阵
        if hasattr(a_knowledge, 'E_src') and isinstance(getattr(a_knowledge, 'E_src', None), dict) and a_knowledge.E_src:
            sample = next(iter(a_knowledge.E_src.values()))
            d_rank = sample.shape[0]
            D_phi = sample.shape[1] if sample.ndim == 2 else 1
            E_src = np.zeros((n_slots, d_rank, D_phi), dtype=np.float32)
            E_tgt = np.zeros((n_slots, d_rank, D_phi), dtype=np.float32)
            for idx, slot in enumerate(all_slots):
                E_src[idx] = a_knowledge.E_src[slot]
                E_tgt[idx] = a_knowledge.E_tgt[slot]

            self._a_cache = {
                'slot_to_idx': slot_to_idx_array,
                'E_src': E_src,
                'E_tgt': E_tgt,
                'd': d_rank,
                'max_slot': max_slot
            }
        # v1 独立嵌入版本：直接从 E_Q/E_V dict 构建
        elif hasattr(a_knowledge, 'E_Q') and a_knowledge.E_Q:
            E_Q = np.zeros((n_slots, self.d), dtype=np.float32)
            E_V = np.zeros((n_slots, self.d), dtype=np.float32)
            for idx, slot in enumerate(all_slots):
                E_Q[idx] = a_knowledge.E_Q[slot]
                E_V[idx] = a_knowledge.E_V[slot]

            self._a_cache = {
                'slot_to_idx': slot_to_idx_array,
                'E_V': E_V,
                'E_Q': E_Q,
                'max_slot': max_slot
            }
        else:
            # 旧版兼容：E + W_Q/W_V
            E_dense = np.zeros((n_slots, self.d), dtype=np.float32)
            for idx, slot in enumerate(all_slots):
                E_dense[idx] = a_knowledge.E[slot]
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                E_V = E_dense @ a_knowledge.W_V
                E_Q = E_dense @ a_knowledge.W_Q

            self._a_cache = {
                'slot_to_idx': slot_to_idx_array,
                'E_V': E_V,
                'E_Q': E_Q,
                'max_slot': max_slot
            }
        self._a_cache_cycle = current_cycle
        self._a_cache_dirty = False

    def write(self, target_slot_counts: dict, active_slots: np.ndarray,
              phi: np.ndarray, slot_scores: np.ndarray,
              vocab_slots: set = None, a_knowledge=None,
              phi_hat: np.ndarray = None, phi_proj: np.ndarray = None,
              target_slots: np.ndarray = None):
        """单次写入（Wake 热路径优化版）"""
        if len(active_slots) == 0 or len(target_slot_counts) == 0:
            return

        self.global_step += 1
        step = self.global_step

        if a_knowledge is not None:
            self._precompute_a_embeddings(a_knowledge)

        # 归一化 phi（按 src 一次）；可直接使用预计算的 phi_hat
        active_slots = np.asarray(active_slots, dtype=np.int32)
        if phi_hat is None:
            phi = np.asarray(phi, dtype=np.float32)
            phi_norms = np.linalg.norm(phi, axis=1, keepdims=True)
            phi_hat_src = phi / (phi_norms + 1e-8)  # (n_src, D)
        else:
            phi_hat_src = np.asarray(phi_hat, dtype=np.float32)

        if target_slots is None:
            target_slots = np.fromiter(target_slot_counts.keys(), dtype=np.int32)
        else:
            target_slots = np.asarray(target_slots, dtype=np.int32)
        n_src = len(active_slots)
        n_tgt = len(target_slots)
        if n_tgt == 0:
            return

        # 展开 src×tgt 对（src-major）
        all_src = np.repeat(active_slots, n_tgt)
        all_tgt = np.tile(target_slots, n_src)
        phi_hat = np.repeat(phi_hat_src, n_tgt, axis=0)

        # B/A 强度
        b_strength = self._compute_b_strength_vectorized(all_src, all_tgt, phi_hat, step)

        if a_knowledge is not None and a_knowledge.has_knowledge:
            a_strength = self._compute_a_strength_single(
                active_slots, phi_hat_src, target_slots, proj_src=phi_proj
            )
            pass  # a_strength 不再除以 d_rank
        else:
            a_strength = np.zeros_like(b_strength)

        gaps = getattr(self, 'gap_target', 1.0) - (a_strength + b_strength)
        delta = self._eta * gaps[:, np.newaxis] * phi_hat

        hash_get = self._hash_table.get

        # 单次写入不会产生重复 key，直接逐对更新
        for i in range(len(all_src)):
            src = int(all_src[i])
            tgt = int(all_tgt[i])
            key = (src, tgt)

            idx = hash_get(key)
            if idx is not None:
                dt = step - self._times[idx]
                if 0 < dt < self._max_dt:
                    self._vecs[idx] *= self._decay_table[dt]
                self._vecs[idx] += delta[i]
                self._times[idx] = step
            else:
                if self._n_entries >= self._capacity:
                    self._expand_capacity()

                idx = self._n_entries
                self._srcs[idx] = src
                self._tgts[idx] = tgt
                self._vecs[idx] = delta[i]
                self._times[idx] = step

                self._hash_table[key] = idx
                self._n_entries += 1
                self.n_b += 1

            # 守恒机制：对同 src 的其他连接减去 phi 分量
            if self._conservation and abs(gaps[i]) > 1e-8:
                src_mask = self._srcs[:self._n_entries] == src
                src_indices = np.where(src_mask)[0]
                other_indices = [j for j in src_indices if j != idx]
                if len(other_indices) > 0:
                    phi_hat_i = phi_hat[i]
                    phi_norm = np.linalg.norm(phi_hat_i)
                    if phi_norm > 1e-8:
                        phi_hat_i = phi_hat_i / phi_norm
                        other_vecs = self._vecs[other_indices]
                        other_strengths = other_vecs @ phi_hat_i
                        positive_mask = other_strengths > 1e-8
                        if np.any(positive_mask):
                            positive_strengths = other_strengths[positive_mask]
                            total_positive_strength = float(np.sum(positive_strengths))
                            gain_strength = self._eta * gaps[i]
                            if total_positive_strength > 1e-8:
                                pos_idx = 0
                                for j, other_idx in enumerate(other_indices):
                                    if positive_mask[j]:
                                        decay_strength = gain_strength * (positive_strengths[pos_idx] / total_positive_strength)
                                        self._vecs[other_idx] -= decay_strength * phi_hat_i
                                        pos_idx += 1

    def _compute_a_strength_single(self, active_slots: np.ndarray,
                                   phi_hat_src: np.ndarray,
                                   target_slots: np.ndarray,
                                   proj_src: np.ndarray = None) -> np.ndarray:
        """单次写入的 A 强度计算（src×tgt 矩阵化，无 P 投影）"""
        n_src = len(active_slots)
        n_tgt = len(target_slots)
        out = np.zeros(n_src * n_tgt, dtype=np.float32)

        if self._a_cache is None:
            return out

        slot_to_idx = self._a_cache['slot_to_idx']
        E_V = self._a_cache['E_V']
        E_Q = self._a_cache['E_Q']
        max_slot = self._a_cache['max_slot']

        src_range = (active_slots >= 0) & (active_slots <= max_slot)
        tgt_range = (target_slots >= 0) & (target_slots <= max_slot)
        if not np.any(src_range) or not np.any(tgt_range):
            return out

        src_pos = np.where(src_range)[0]
        tgt_pos = np.where(tgt_range)[0]
        src_idx_raw = slot_to_idx[active_slots[src_pos]]
        tgt_idx_raw = slot_to_idx[target_slots[tgt_pos]]

        src_exists = src_idx_raw >= 0
        tgt_exists = tgt_idx_raw >= 0
        if not np.any(src_exists) or not np.any(tgt_exists):
            return out

        valid_src_pos = src_pos[src_exists]
        valid_tgt_pos = tgt_pos[tgt_exists]
        valid_src_idx = src_idx_raw[src_exists]
        valid_tgt_idx = tgt_idx_raw[tgt_exists]

        # D = d，phi_hat 已在同一空间，无需 P 投影
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            ctx = E_V[valid_src_idx] * phi_hat_src[valid_src_pos]  # (n_valid_src, d)
            strengths = ctx @ E_Q[valid_tgt_idx].T                 # (n_valid_src, n_valid_tgt)

        # 不取 abs: 保留方向信息，让 gap 正确修正 A 的反向预测
        strengths = strengths.astype(np.float32, copy=False)
        row_offsets = valid_src_pos.astype(np.int64) * int(n_tgt)
        flat_indices = (row_offsets[:, None] + valid_tgt_pos[None, :]).reshape(-1)
        out[flat_indices] = strengths.reshape(-1)
        return out

    def write_batch(self, batch_data: List[Dict], a_knowledge=None):
        """批量写入"""
        if len(batch_data) == 0:
            return

        self.global_step += 1
        step = self.global_step

        if a_knowledge is not None:
            self._precompute_a_embeddings(a_knowledge)

        # 收集所有更新请求
        all_src_parts = []
        all_tgt_parts = []
        all_phi_hat_parts = []

        for item in batch_data:
            active_slots = np.asarray(item['active_slots'], dtype=np.int32)
            phi = np.asarray(item['phi'], dtype=np.float32)
            target_slots = np.fromiter(item['target_slots'].keys(), dtype=np.int32)

            n_src = len(active_slots)
            n_tgt = len(target_slots)

            if n_src == 0 or n_tgt == 0:
                continue

            # 先按 src 归一化 phi，再按 target 展开，避免重复归一化
            phi_norms = np.linalg.norm(phi, axis=1, keepdims=True)
            phi_hat_src = phi / (phi_norms + 1e-8)

            all_src_parts.append(np.repeat(active_slots, n_tgt))
            all_tgt_parts.append(np.tile(target_slots, n_src))
            all_phi_hat_parts.append(np.repeat(phi_hat_src, n_tgt, axis=0))

        if len(all_src_parts) == 0:
            return

        if len(all_src_parts) == 1:
            all_src = all_src_parts[0]
            all_tgt = all_tgt_parts[0]
            phi_hat = all_phi_hat_parts[0]
        else:
            all_src = np.concatenate(all_src_parts)
            all_tgt = np.concatenate(all_tgt_parts)
            phi_hat = np.concatenate(all_phi_hat_parts, axis=0)

        # B强度计算（向量化版本）
        b_strength = self._compute_b_strength_vectorized(all_src, all_tgt, phi_hat, step)

        # A强度计算（向量化版本）
        a_strength = np.zeros_like(b_strength)
        if a_knowledge is not None and a_knowledge.has_knowledge:
            # 保留方向信息（不取 abs），让 gap 正确修正 A 的反向预测
            a_strength = self._compute_a_strength_vectorized(all_src, all_tgt, phi_hat)

        # 计算gap：g = target - (s_A + s_B)
        gaps = getattr(self, 'gap_target', 1.0) - (a_strength + b_strength)

        # 允许负gap（用于纠正A的过冲）

        # 合并
        keys = all_src.astype(np.int64) * self.K + all_tgt.astype(np.int64)
        unique_keys, inverse_indices, counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )

        n_unique = len(unique_keys)
        # 无重复键时直接走快路径，避免分组聚合开销
        if n_unique == len(keys):
            phi_mean = phi_hat
            gap_mean = gaps
        else:
            # 分组求和：比 np.add.at 对小批次更省开销
            order = np.argsort(inverse_indices, kind='mergesort')
            inv_sorted = inverse_indices[order]
            split_idx = np.flatnonzero(np.diff(inv_sorted)) + 1
            group_starts = np.concatenate(([0], split_idx))

            phi_sum = np.add.reduceat(phi_hat[order], group_starts, axis=0)
            gap_sum = np.add.reduceat(gaps[order], group_starts)
            phi_mean = phi_sum / counts[:, np.newaxis]
            gap_mean = gap_sum / counts

        delta = self._eta * gap_mean[:, np.newaxis] * phi_mean

        unique_src = (unique_keys // self.K).astype(np.int32)
        unique_tgt = (unique_keys % self.K).astype(np.int32)

        # 批量应用更新（使用get代替in）
        for i in range(len(unique_src)):
            src = int(unique_src[i])
            tgt = int(unique_tgt[i])
            key = (src, tgt)

            idx = self._hash_table.get(key)
            if idx is not None:
                # 更新已存在的
                dt = step - self._times[idx]
                if 0 < dt < self._max_dt:
                    decay_factor_time = self._decay_table[dt]
                    self._vecs[idx] *= decay_factor_time
                self._vecs[idx] += delta[i]
                self._times[idx] = step
            else:
                if self._n_entries >= self._capacity:
                    self._expand_capacity()

                idx = self._n_entries
                self._srcs[idx] = src
                self._tgts[idx] = tgt
                self._vecs[idx] = delta[i]
                self._times[idx] = step

                self._hash_table[key] = idx
                self._n_entries += 1
                self.n_b += 1

            # 守恒机制：对其他连接减去phi分量
            if self._conservation and abs(gap_mean[i]) > 1e-8:
                # 找到所有相同src的其他连接
                src_mask = self._srcs[:self._n_entries] == src
                src_indices = np.where(src_mask)[0]

                other_indices = [j for j in src_indices if j != idx]
                if len(other_indices) > 0:
                    # 归一化phi
                    phi_hat_i = phi_mean[i]
                    phi_norm = np.linalg.norm(phi_hat_i)
                    if phi_norm > 1e-8:
                        phi_hat_i = phi_hat_i / phi_norm

                        # 计算其他连接在当前phi下的强度
                        other_vecs = self._vecs[other_indices]
                        other_strengths = other_vecs @ phi_hat_i  # (n_other,)

                        # 只考虑正强度的连接
                        positive_mask = other_strengths > 1e-8
                        if np.any(positive_mask):
                            positive_strengths = other_strengths[positive_mask]
                            total_positive_strength = float(np.sum(positive_strengths))

                            # 目标连接增加的强度
                            gain_strength = self._eta * gap_mean[i]

                            # 按比例从正强度连接中减去phi分量
                            if total_positive_strength > 1e-8:
                                pos_idx = 0
                                for j, other_idx in enumerate(other_indices):
                                    if positive_mask[j]:
                                        # 这个连接应该减少的强度
                                        decay_strength = gain_strength * (positive_strengths[pos_idx] / total_positive_strength)
                                        # 减去对应的phi分量
                                        self._vecs[other_idx] -= decay_strength * phi_hat_i
                                        pos_idx += 1

    def _compute_b_strength_vectorized(self, all_src, all_tgt, phi_hat, step):
        """B强度计算（向量化版本）"""
        n = len(all_src)
        b_strength = np.zeros(n, dtype=np.float32)

        if self._n_entries == 0:
            return b_strength

        # 批量查找：构建查询列表（使用get代替in，避免性能下降）
        indices = []
        query_positions = []

        for i in range(n):
            src = int(all_src[i])
            tgt = int(all_tgt[i])
            key = (src, tgt)

            idx = self._hash_table.get(key)
            if idx is not None:
                indices.append(idx)
                query_positions.append(i)

        if len(indices) == 0:
            return b_strength

        # 向量化处理
        indices = np.array(indices, dtype=np.int32)
        query_positions = np.array(query_positions, dtype=np.int32)

        # 批量获取向量和时间
        matched_vecs = self._vecs[indices]  # (n_matched, D)
        matched_times = self._times[indices]  # (n_matched,)
        matched_phi = phi_hat[query_positions]  # (n_matched, D)

        # 向量化衰减
        dts = step - matched_times
        decay_mask = (dts > 0) & (dts < self._max_dt)
        decay_factors = np.ones(len(indices), dtype=np.float32)
        decay_factors[decay_mask] = self._decay_table[dts[decay_mask]]

        # 应用衰减
        decayed_vecs = matched_vecs * decay_factors[:, np.newaxis]

        # 向量化点积
        strengths = np.sum(matched_phi * decayed_vecs, axis=1)

        # 填充结果
        b_strength[query_positions] = strengths

        return b_strength

    def _compute_a_strength_vectorized(self, all_src, all_tgt, phi_hat):
        """A强度计算向量化（无 P 投影，D = d）"""
        if self._a_cache is None:
            return np.zeros(len(all_src), dtype=np.float32)

        slot_to_idx = self._a_cache['slot_to_idx']
        E_V = self._a_cache['E_V']
        E_Q = self._a_cache['E_Q']
        max_slot = self._a_cache['max_slot']

        n = len(all_src)

        valid_src_mask = all_src <= max_slot
        valid_tgt_mask = all_tgt <= max_slot
        range_valid_mask = valid_src_mask & valid_tgt_mask

        if not np.any(range_valid_mask):
            return np.zeros(n, dtype=np.float32)

        src_indices = slot_to_idx[all_src[range_valid_mask]]
        tgt_indices = slot_to_idx[all_tgt[range_valid_mask]]

        exists_mask = (src_indices >= 0) & (tgt_indices >= 0)

        a_strength = np.zeros(n, dtype=np.float32)

        if not np.any(exists_mask):
            return a_strength

        valid_indices = np.where(range_valid_mask)[0][exists_mask]
        valid_src_idx = src_indices[exists_mask]
        valid_tgt_idx = tgt_indices[exists_mask]

        # D = d，phi_hat 直接在同一空间，无需 P 投影
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            a_strength[valid_indices] = np.sum(
                E_Q[valid_tgt_idx] * E_V[valid_src_idx] * phi_hat[valid_indices],
                axis=1
            )

        return a_strength

    def predict(self, active_slots: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """预测下一个槽位的分数（向量化优化版本）

        Args:
            active_slots: 活跃源槽位 (n_src,)
            phi: 距离特征 (n_src, D)

        Returns:
            scores: 目标槽位分数 (K,)
        """
        scores = np.zeros(self.K, dtype=np.float32)

        if len(active_slots) == 0 or self._n_entries == 0:
            return scores

        # φ归一化
        phi_norms = np.linalg.norm(phi, axis=1, keepdims=True)
        phi_hat = phi / (phi_norms + 1e-8)

        step = self.global_step

        # 向量化查询：找到所有匹配active_slots的条目
        # 使用numpy的isin来批量匹配
        srcs_array = self._srcs[:self._n_entries]
        mask = np.isin(srcs_array, active_slots)

        if not np.any(mask):
            return scores

        # 获取匹配的索引
        matched_indices = np.where(mask)[0]

        # 批量获取数据
        matched_srcs = srcs_array[matched_indices]
        matched_tgts = self._tgts[matched_indices]
        matched_vecs = self._vecs[matched_indices]
        matched_times = self._times[matched_indices]

        # 惰性衰减（向量化）
        dts = step - matched_times
        decay_mask = (dts > 0) & (dts < self._max_dt)
        decay_factors = np.ones(len(matched_indices), dtype=np.float32)
        decay_factors[decay_mask] = self._decay_table[dts[decay_mask]]

        # 应用衰减
        matched_vecs = matched_vecs * decay_factors[:, np.newaxis]

        # 更新衰减后的值（批量）
        self._vecs[matched_indices] = matched_vecs
        self._times[matched_indices[decay_mask]] = step

        # 为每个src找到对应的phi索引
        # 创建映射数组：slot_id -> phi_index (-1表示不存在)
        max_slot = int(np.max(active_slots)) + 1
        slot_to_phi = np.full(max_slot, -1, dtype=np.int32)
        slot_to_phi[active_slots] = np.arange(len(active_slots), dtype=np.int32)

        # 获取每个matched_src对应的phi索引
        phi_indices = slot_to_phi[matched_srcs]
        valid_mask = phi_indices >= 0

        if not np.any(valid_mask):
            return scores

        # 过滤出有效的条目
        valid_phi_indices = phi_indices[valid_mask]
        valid_tgts = matched_tgts[valid_mask]
        valid_vecs = matched_vecs[valid_mask]

        # 向量化计算分数
        valid_phi = phi_hat[valid_phi_indices]  # (n_valid, D)
        dot_products = np.sum(valid_phi * valid_vecs, axis=1)  # (n_valid,)

        # 累加到scores（使用np.add.at处理重复的tgt）
        np.add.at(scores, valid_tgts, dot_products)

        scores = scores / float(max(1, len(active_slots)))

        # B top-1 mask: 只保留 top-1 word 的 slot scores
        if getattr(self, 'predict_top1_mask', False) and self._top1_decoder is not None:
            top1_word, _ = self._top1_decoder.decode_top_k(scores, 1)[0]
            masked = np.zeros_like(scores)
            if top1_word in self._top1_word_to_slots:
                for sl in self._top1_word_to_slots[top1_word]:
                    masked[int(sl)] = scores[int(sl)]
            return masked

        return scores

    def clear(self):
        """清空B"""
        self._hash_table.clear()
        self._n_entries = 0
        self.n_b = 0

    def decay(self, retention_ratio: float):
        """对所有B Memory条目进行衰减

        Args:
            retention_ratio: 保留比例 (0-1)，0表示完全清空，1表示完全保留
        """
        if retention_ratio <= 0.0:
            self.clear()
            return

        if retention_ratio >= 1.0:
            return  # 不衰减

        # 对所有向量进行衰减
        self._vecs[:self._n_entries] *= retention_ratio

    def stats(self) -> dict:
        """统计信息"""
        if self._n_entries == 0:
            return {
                'n_entries': 0,
                'norm_mean': 0,
                'norm_max': 0,
                'norm_min': 0
            }

        norms = np.linalg.norm(self._vecs[:self._n_entries], axis=1)
        return {
            'n_entries': self._n_entries,
            'norm_mean': float(np.mean(norms)),
            'norm_max': float(np.max(norms)),
            'norm_min': float(np.min(norms))
        }

    def get_entries(self):
        """获取所有条目"""
        entries = []
        for i in range(self._n_entries):
            entries.append((
                int(self._srcs[i]),
                int(self._tgts[i]),
                self._vecs[i].copy()
            ))
        return entries

    def prune_by_norm(self, keep_ratio=None, max_entries=None):
        """按范数裁剪：保留 Top-K% 或最多 N 个条目

        策略：
        - 模型越惊讶的连接，写入的范数越大
        - 保留范数最大的连接 = 保留最重要的连接
        - 避免时间偏差（相对排序不受衰减影响）

        Args:
            keep_ratio: 保留比例（默认使用实例变量或0.8）
            max_entries: 最大条目数（默认使用实例变量或200k）

        Returns:
            n_pruned: 删除的条目数
        """
        import time
        t0 = time.time()

        if self._n_entries == 0:
            return 0

        # 优先使用实例变量，否则使用参数，最后使用默认值
        if keep_ratio is None:
            keep_ratio = getattr(self, '_prune_keep_ratio', 0.8)
        if max_entries is None:
            max_entries = getattr(self, '_prune_max_entries', 200000)

        # 计算所有条目的范数
        print(f"    [裁剪] 计算 {self._n_entries:,} 个范数...")
        t1 = time.time()
        norms = np.linalg.norm(self._vecs[:self._n_entries], axis=1)
        print(f"    [裁剪] 范数计算完成，耗时 {(time.time()-t1)*1000:.0f}ms")

        # 计算保留数量（取两个限制的最小值）
        n_keep_ratio = int(self._n_entries * keep_ratio)
        n_keep = min(n_keep_ratio, max_entries)

        if n_keep >= self._n_entries:
            # 不需要裁剪
            return 0

        # 找到阈值（第 n_keep 大的范数）
        print(f"    [裁剪] 查找第 {n_keep:,} 大的阈值...")
        t1 = time.time()
        threshold = np.partition(norms, -n_keep)[-n_keep]
        print(f"    [裁剪] 阈值查找完成，耗时 {(time.time()-t1)*1000:.0f}ms")

        # 保留范数 >= threshold 的条目
        keep_mask = norms >= threshold

        # 如果保留数量超过 n_keep（因为有相同范数的条目），随机选择
        if np.sum(keep_mask) > n_keep:
            keep_indices = np.where(keep_mask)[0]
            selected = np.random.choice(keep_indices, n_keep, replace=False)
            keep_mask = np.zeros(self._n_entries, dtype=bool)
            keep_mask[selected] = True

        n_pruned = self._n_entries - np.sum(keep_mask)

        if n_pruned > 0:
            # 重建哈希表和数组
            print(f"    [裁剪] 重建哈希表，保留 {n_keep:,} 条...")
            t1 = time.time()
            new_hash_table = {}
            new_n_entries = 0

            for i in range(self._n_entries):
                if keep_mask[i]:
                    src = int(self._srcs[i])
                    tgt = int(self._tgts[i])
                    key = (src, tgt)

                    # 复制到新位置
                    self._srcs[new_n_entries] = src
                    self._tgts[new_n_entries] = tgt
                    self._vecs[new_n_entries] = self._vecs[i]
                    self._times[new_n_entries] = self._times[i]

                    # 更新哈希表
                    new_hash_table[key] = new_n_entries
                    new_n_entries += 1

            print(f"    [裁剪] 哈希表重建完成，耗时 {(time.time()-t1)*1000:.0f}ms")

            # 替换哈希表
            self._hash_table = new_hash_table
            self._n_entries = new_n_entries
            self.n_b = new_n_entries

            print(f"    [裁剪] 总耗时 {(time.time()-t0)*1000:.0f}ms")
            return n_pruned

        return 0
