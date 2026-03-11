"""A 通道 v2: Slice-wise 矩阵分解

每个距离切片 m 独立做 rank-d 矩阵分解，d 和 D 解耦。

存储:
  E_src[slot] → (d, D)   source 角色
  E_tgt[slot] → (d, D)   target 角色

重建:
  A[tgt, src] = Σ_r E_tgt[tgt, r, :] * E_src[src, r, :]  → (D,)  与 B 同空间
"""

import numpy as np
import warnings

from .config import MathBrainConfig
from .b_memory_hashtable_v2 import BMemoryEnergyHashTableV2

try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False


class AKnowledgeV2:
    """长期语义知识 v2 (Slice-wise 矩阵分解)

    每个距离切片独立 rank-d 分解，d 和 D 解耦。

    因子: E_src (dict[slot→(d,D)]), E_tgt (dict[slot→(d,D)])
    Â[tgt, src] = Σ_r E_tgt[tgt,r,:] * E_src[src,r,:]  → (D,)

    预分配缓存:
      _E_src:      (n_slots, d, D)
      _E_tgt:      (n_slots, d, D)
      _E_tgt_flat: (n_slots, d*D)  — 用于快速 matmul
      _slot_arr:   (n_slots,)
      _slot_to_idx: dict
    """

    def __init__(self, D: int, d: int, K: int = 65536):
        self.K = K
        self.D = D   # phi 维度 (e.g. 8)
        self.d = d   # CP rank  (e.g. 64)

        self.E_src: dict = {}   # slot_id → ndarray(d, D)
        self.E_tgt: dict = {}   # slot_id → ndarray(d, D)
        self.has_knowledge = False

        # 自适应衰减
        self.slot_last_update: dict = {}
        self.sleep_cycle_count: int = 0

        # 预分配缓存
        self._E_src: np.ndarray | None = None
        self._E_tgt: np.ndarray | None = None
        self._E_tgt_flat: np.ndarray | None = None
        self._slot_arr: np.ndarray | None = None
        self._slot_to_idx: dict | None = None

    def _build_cache(self):
        """从 E_src, E_tgt 构建预分配矩阵"""
        if not self.E_src:
            self._E_src = None
            self._E_tgt = None
            self._E_tgt_flat = None
            self._slot_arr = None
            self._slot_to_idx = None
            return

        slot_list = sorted(self.E_src.keys())
        n = len(slot_list)

        E_src_mat = np.empty((n, self.d, self.D), dtype=np.float32)
        E_tgt_mat = np.empty((n, self.d, self.D), dtype=np.float32)
        slot_to_idx = {}
        for i, s in enumerate(slot_list):
            E_src_mat[i] = self.E_src[s]
            E_tgt_mat[i] = self.E_tgt[s]
            slot_to_idx[s] = i

        self._E_src = E_src_mat
        self._E_tgt = E_tgt_mat
        self._E_tgt_flat = E_tgt_mat.reshape(n, self.d * self.D)
        self._slot_arr = np.array(slot_list, dtype=np.int32)
        self._slot_to_idx = slot_to_idx

    # ==================== 预测 ====================

    def predict(self, active_slots: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """A 通道预测 → slot_scores (K,)

        C_A = Σ_i E_src[src_i] * φ̂[i]          — (d, D)
        scores[tgt] = Frobenius(E_tgt[tgt], C_A)  — scalar per tgt
        """
        scores = np.zeros(self.K, dtype=np.float32)
        if not self.has_knowledge or len(active_slots) == 0:
            return scores
        if self._E_src is None:
            return scores

        # φ 归一化
        norms = np.linalg.norm(phi, axis=1, keepdims=True)
        phi_hat = np.where(norms > 1e-8, phi / norms, phi)   # (n_src, D)

        # 匹配 active_slots → cache index
        s2i = self._slot_to_idx
        src_indices = []
        phi_indices = []
        for i, src in enumerate(active_slots):
            idx = s2i.get(int(src))
            if idx is not None:
                src_indices.append(idx)
                phi_indices.append(i)

        if not src_indices:
            return scores

        src_idx = np.array(src_indices, dtype=np.int32)
        phi_idx = np.array(phi_indices, dtype=np.int32)
        n_active = max(1, len(src_idx))

        # C_A = Σ E_src[src] * φ̂[i]   broadcast: (n, d, D) * (n, 1, D) → sum → (d, D)
        phi_expanded = phi_hat[phi_idx][:, None, :]            # (n, 1, D)
        C_A = np.sum(self._E_src[src_idx] * phi_expanded, axis=0)  # (d, D)

        c_flat = C_A.reshape(-1)                                # (d*D,)
        if np.dot(c_flat, c_flat) < 1e-12:
            return scores

        # scores = E_tgt_flat @ c_flat
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            all_scores = self._E_tgt_flat @ c_flat              # (n_slots,)
        scores[self._slot_arr] = all_scores / float(n_active)

        return scores

    # ==================== 单条查询 ====================

    def compute_entry(self, src: int, tgt: int) -> np.ndarray:
        """单条 A[tgt, src] = Σ_r E_tgt[tgt,r,:] * E_src[src,r,:]  → (D,)"""
        if not self.has_knowledge:
            return np.zeros(self.D, dtype=np.float32)

        if self._slot_to_idx is not None:
            src_idx = self._slot_to_idx.get(src)
            tgt_idx = self._slot_to_idx.get(tgt)
            if src_idx is None or tgt_idx is None:
                return np.zeros(self.D, dtype=np.float32)
            return np.sum(self._E_tgt[tgt_idx] * self._E_src[src_idx], axis=0)

        # fallback
        if src not in self.E_src or tgt not in self.E_tgt:
            return np.zeros(self.D, dtype=np.float32)
        return np.sum(self.E_tgt[tgt] * self.E_src[src], axis=0)

    def get_a_strength(self, src: int, tgt: int, proj: np.ndarray) -> float:
        """快速计算 A 强度 s_A = compute_entry(src,tgt) · proj

        proj = φ̂[i]（归一化的 φ, D 维）
        """
        entry = self.compute_entry(src, tgt)
        return float(np.dot(entry, proj))

    # ==================== Sleep 接口 ====================

    def update_from_sleep(self, E_src_dict: dict, E_tgt_dict: dict,
                          updated_slots: set = None):
        """接收 Sleep 结果，重建预分配缓存"""
        self.E_src = E_src_dict
        self.E_tgt = E_tgt_dict
        self.has_knowledge = True

        if updated_slots is not None:
            for slot in updated_slots:
                self.slot_last_update[slot] = self.sleep_cycle_count

        self.sleep_cycle_count += 1
        self._build_cache()

    # ==================== 自适应衰减 ====================

    def get_adaptive_gamma(self, slot_id: int, gamma_initial=0.99,
                           gamma_min=0.9, decay_rate=0.9):
        if slot_id not in self.slot_last_update:
            return gamma_min
        cycles_since_update = self.sleep_cycle_count - self.slot_last_update[slot_id]
        gamma = gamma_initial * (decay_rate ** cycles_since_update)
        return max(gamma_min, gamma)

    def get_adaptive_gamma_batch(self, slot_ids, gamma_initial=0.99,
                                 gamma_min=0.9, decay_rate=0.9):
        gammas = np.full(len(slot_ids), gamma_min, dtype=np.float32)
        for i, slot_id in enumerate(slot_ids):
            if slot_id in self.slot_last_update:
                cycles_since_update = self.sleep_cycle_count - self.slot_last_update[slot_id]
                gamma = gamma_initial * (decay_rate ** cycles_since_update)
                gammas[i] = max(gamma_min, gamma)
        return gammas

    # ==================== 诊断 API ====================

    def stats(self) -> dict:
        if not self.has_knowledge:
            return {'has_knowledge': False, 'n_embeddings': 0}
        src_norms = [np.linalg.norm(v) for v in self.E_src.values()]
        tgt_norms = [np.linalg.norm(v) for v in self.E_tgt.values()]
        return {
            'has_knowledge': True,
            'n_embeddings': len(self.E_src),
            'd': self.d,
            'D': self.D,
            'E_src_norm_mean': float(np.mean(src_norms)) if src_norms else 0,
            'E_tgt_norm_mean': float(np.mean(tgt_norms)) if tgt_norms else 0,
        }

    def similarity(self, slot1: int, slot2: int) -> dict:
        if slot1 not in self.E_src or slot2 not in self.E_src:
            return {'comparable': False}
        # Frobenius cosine similarity
        s1, s2 = self.E_src[slot1].ravel(), self.E_src[slot2].ravel()
        cos_src = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-8)
        t1, t2 = self.E_tgt[slot1].ravel(), self.E_tgt[slot2].ravel()
        cos_tgt = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-8)
        return {'comparable': True, 'cosine_src': float(cos_src), 'cosine_tgt': float(cos_tgt)}


# ================================================================
# BMemoryV2Wrapper: B Memory 适配 v2 A 模块
# ================================================================

class BMemoryV2Wrapper(BMemoryEnergyHashTableV2):
    """B Memory wrapper for v2 A module (slice-wise matrix factorization)

    覆写 A 相关方法以处理 (d, D) 形状的嵌入。
    其余方法（write, predict 等）全部继承不变。

    GPU 加速：A strength 计算在 GPU 上执行（MPS）。
    """

    def __init__(self, config: MathBrainConfig = None, cp_rank: int = 384, use_gpu: bool = False):
        super().__init__(config)
        self._cp_rank = cp_rank
        self._use_gpu = use_gpu and HAS_TORCH and HAS_MPS  # 默认禁用 GPU
        if self._use_gpu:
            self._device = torch.device("mps")
        else:
            self._device = None

    def _precompute_a_embeddings(self, a_knowledge):
        """预计算 A 的嵌入矩阵 — v2: (n_slots, d, D)，支持 GPU 加速"""
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

        if not hasattr(a_knowledge, 'E_src') or not a_knowledge.E_src:
            self._a_cache = None
            return

        all_slots = sorted(a_knowledge.E_src.keys())
        if not all_slots:
            self._a_cache = None
            return

        max_slot = max(all_slots)
        n_slots = len(all_slots)
        d = a_knowledge.d
        D = a_knowledge.D

        slot_to_idx_array = np.full(max_slot + 1, -1, dtype=np.int32)
        slot_to_idx_array[all_slots] = np.arange(n_slots, dtype=np.int32)

        E_src = np.zeros((n_slots, d, D), dtype=np.float32)
        E_tgt = np.zeros((n_slots, d, D), dtype=np.float32)
        for idx, slot in enumerate(all_slots):
            E_src[idx] = a_knowledge.E_src[slot]
            E_tgt[idx] = a_knowledge.E_tgt[slot]

        # GPU 加速：将嵌入移到 GPU
        if self._use_gpu:
            E_src_flat_torch = torch.tensor(
                E_src.reshape(n_slots, d * D), dtype=torch.float32, device=self._device
            )
            E_tgt_flat_torch = torch.tensor(
                E_tgt.reshape(n_slots, d * D), dtype=torch.float32, device=self._device
            )
            self._a_cache = {
                'slot_to_idx': slot_to_idx_array,
                'E_src': E_src,  # CPU 版本（用于 fallback）
                'E_tgt': E_tgt,
                'E_src_flat_torch': E_src_flat_torch,  # GPU 版本
                'E_tgt_flat_torch': E_tgt_flat_torch,
                'max_slot': max_slot,
                'd': d,
                'D': D,
                'use_gpu': True,
            }
        else:
            # CPU 版本
            E_src_flat = E_src.reshape(n_slots, d * D)
            E_tgt_flat = E_tgt.reshape(n_slots, d * D)
            self._a_cache = {
                'slot_to_idx': slot_to_idx_array,
                'E_src': E_src,
                'E_tgt': E_tgt,
                'E_src_flat': E_src_flat,
                'E_tgt_flat': E_tgt_flat,
                'max_slot': max_slot,
                'd': d,
                'D': D,
                'use_gpu': False,
            }

        self._a_cache_cycle = current_cycle
        self._a_cache_dirty = False

    def _compute_a_strength_single(self, active_slots: np.ndarray,
                                   phi_hat_src: np.ndarray,
                                   target_slots: np.ndarray,
                                   proj_src: np.ndarray = None) -> np.ndarray:
        """A 强度计算 — v2: slice-wise 矩阵分解，GPU 加速

        ctx = E_src[src] * φ̂           → (d, D)
        strengths = ctx_flat @ E_tgt_flat.T → (n_src, n_tgt)
        """
        n_src = len(active_slots)
        n_tgt = len(target_slots)
        out = np.zeros(n_src * n_tgt, dtype=np.float32)

        if self._a_cache is None:
            return out

        slot_to_idx = self._a_cache['slot_to_idx']
        max_slot = self._a_cache['max_slot']
        d = self._a_cache['d']
        D_phi = self._a_cache['D']
        use_gpu = self._a_cache.get('use_gpu', False)

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

        # GPU 加速路径 - 直接用缓存的 GPU tensor，避免重复分配
        if use_gpu and self._use_gpu:
            E_src = self._a_cache['E_src']  # (n_slots, d, D) numpy - 仅用于 reshape
            E_src_flat_torch = self._a_cache['E_src_flat_torch']  # (n_slots, d*D) torch - 已在 GPU
            E_tgt_flat_torch = self._a_cache['E_tgt_flat_torch']  # 已在 GPU

            # phi_hat 小数据，一次性移到 GPU
            phi_expanded = phi_hat_src[valid_src_pos][:, None, :]  # (n_valid_src, 1, D)
            phi_expanded_torch = torch.tensor(phi_expanded, dtype=torch.float32, device=self._device)

            # 在 GPU 上索引，不创建新 tensor
            with torch.no_grad():
                # E_src 在 GPU 上索引并 reshape
                E_src_subset_flat = E_src_flat_torch[valid_src_idx]  # (n_valid_src, d*D) - GPU 索引
                E_src_subset = E_src_subset_flat.reshape(len(valid_src_idx), d, D_phi)  # GPU reshape

                # ctx = E_src * phi_expanded (都在 GPU)
                ctx = E_src_subset * phi_expanded_torch  # (n_valid_src, d, D)
                ctx_flat = ctx.reshape(len(valid_src_idx), d * D_phi)

                # E_tgt 在 GPU 上索引
                E_tgt_subset = E_tgt_flat_torch[valid_tgt_idx]  # (n_valid_tgt, d*D) - GPU 索引
                # 不取 abs: 保留方向信息，让 gap 正确修正 A 的反向预测
                strengths = (ctx_flat @ E_tgt_subset.T)  # (n_valid_src, n_valid_tgt)
                strengths_np = strengths.cpu().numpy().astype(np.float32)

            row_offsets = valid_src_pos.astype(np.int64) * int(n_tgt)
            flat_indices = (row_offsets[:, None] + valid_tgt_pos[None, :]).reshape(-1)
            out[flat_indices] = strengths_np.reshape(-1)
            return out

        # CPU fallback
        E_src = self._a_cache['E_src']
        E_tgt_flat = self._a_cache['E_tgt_flat']

        phi_expanded = phi_hat_src[valid_src_pos][:, None, :]   # (n_valid_src, 1, D)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            ctx = E_src[valid_src_idx] * phi_expanded            # (n_valid_src, d, D)
            ctx_flat = ctx.reshape(len(valid_src_idx), d * D_phi)
            strengths = ctx_flat @ E_tgt_flat[valid_tgt_idx].T   # (n_valid_src, n_valid_tgt)

        # 不取 abs: 保留方向信息，让 gap 正确修正 A 的反向预测
        strengths = strengths.astype(np.float32, copy=False)
        row_offsets = valid_src_pos.astype(np.int64) * int(n_tgt)
        flat_indices = (row_offsets[:, None] + valid_tgt_pos[None, :]).reshape(-1)
        out[flat_indices] = strengths.reshape(-1)
        return out

    def _compute_a_strength_vectorized(self, all_src, all_tgt, phi_hat):
        """A 强度计算向量化 — v2: slice-wise 矩阵分解，GPU 加速"""
        if self._a_cache is None:
            return np.zeros(len(all_src), dtype=np.float32)

        slot_to_idx = self._a_cache['slot_to_idx']
        max_slot = self._a_cache['max_slot']
        use_gpu = self._a_cache.get('use_gpu', False)

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

        # GPU 加速路径 - 直接用缓存的 GPU tensor
        if use_gpu and self._use_gpu:
            E_src = self._a_cache['E_src']  # (n_slots, d, D) numpy - 仅用于获取形状
            E_tgt = self._a_cache['E_tgt']
            d = E_src.shape[1]
            D = E_src.shape[2]

            # phi_hat 小数据，一次性移到 GPU
            phi_expanded = phi_hat[valid_indices][:, None, :]  # (n_valid, 1, D)
            phi_torch = torch.tensor(phi_expanded, dtype=torch.float32, device=self._device)

            # 在 GPU 上索引 E_src/E_tgt（避免创建新 tensor）
            # 注意：这里需要先转换整个 E_src/E_tgt 到 GPU（已在 cache 中）
            # 但 cache 中只有 flat 版本，需要 reshape
            E_src_flat_torch = self._a_cache['E_src_flat_torch']
            E_tgt_flat_torch = self._a_cache['E_tgt_flat_torch']

            with torch.no_grad():
                # 在 GPU 上索引并 reshape
                E_src_subset_flat = E_src_flat_torch[valid_src_idx]  # (n_valid, d*D)
                E_src_subset = E_src_subset_flat.reshape(len(valid_src_idx), d, D)

                E_tgt_subset_flat = E_tgt_flat_torch[valid_tgt_idx]  # (n_valid, d*D)
                E_tgt_subset = E_tgt_subset_flat.reshape(len(valid_tgt_idx), d, D)

                ctx = E_src_subset * phi_torch  # (n_valid, d, D)
                # Frobenius: Σ_{r,m} E_tgt[tgt,r,m] * ctx[r,m]
                strengths = torch.sum(E_tgt_subset * ctx, dim=(1, 2))
                a_strength[valid_indices] = strengths.cpu().numpy().astype(np.float32)

            return a_strength

        # CPU fallback
        E_src = self._a_cache['E_src']
        E_tgt = self._a_cache['E_tgt']

        phi_expanded = phi_hat[valid_indices][:, None, :]       # (n_valid, 1, D)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            ctx = E_src[valid_src_idx] * phi_expanded            # (n_valid, d, D)
            a_strength[valid_indices] = np.sum(
                E_tgt[valid_tgt_idx] * ctx, axis=(1, 2)
            )

        return a_strength
