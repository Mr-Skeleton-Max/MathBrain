"""Q 模块: 稀疏 EMA 上下文状态 (V2 模式)"""

import numpy as np

from .config import MathBrainConfig


class QState:
    """稀疏 Q 状态管理

    内部维护对齐数组:
      _slot_ids  : ndarray(n,)     — 活跃槽位 ID
      _Q_values  : ndarray(n, N)   — 对应 Q 向量
      _slot_to_idx : dict           — slot_id → 数组索引

    衰减和阈值裁剪都是数组广播操作。
    新槽位 append 到末尾，周期性 compact 移除死槽位。
    """

    def __init__(self, config: MathBrainConfig = None):
        cfg = config or MathBrainConfig()
        self.N = cfg.N
        self._rho = cfg.rho.copy()
        self._eps_q = cfg.EPS_Q
        self._theta_q = cfg.THETA_Q

        # 预分配（初始容量 1024，按需扩展）
        self._capacity = 1024
        self._n = 0
        self._slot_ids = np.empty(self._capacity, dtype=np.int32)
        self._Q_values = np.empty((self._capacity, self.N), dtype=np.float32)
        self._slot_to_idx: dict = {}

    def update(self, slot_counts: dict):
        """衰减现有 Q，然后写入新的 slot 激活。

        Args:
            slot_counts: Dict[slot_id, count] — 来自 retina.encode()
        """
        if self._n > 0:
            # 向量化衰减
            self._Q_values[:self._n] *= self._rho
            # 裁剪死槽位
            alive = np.max(np.abs(self._Q_values[:self._n]), axis=1) >= self._eps_q
            if not np.all(alive):
                self._compact(alive)

        # 写入新激活
        for slot_id, count in slot_counts.items():
            idx = self._slot_to_idx.get(slot_id)
            if idx is not None:
                self._Q_values[idx] += float(count)
            else:
                self._append(slot_id, float(count))

    def reset(self):
        self._n = 0
        self._slot_to_idx.clear()

    def get_active(self):
        """返回 (active_slot_ids, Q_values) — 仅 norm > THETA_Q 的槽位"""
        if self._n == 0:
            return np.array([], dtype=np.int32), np.zeros((0, self.N), dtype=np.float32)

        norms = np.linalg.norm(self._Q_values[:self._n], axis=1)
        mask = norms > self._theta_q
        ids = self._slot_ids[:self._n][mask]
        vals = self._Q_values[:self._n][mask]
        return ids.copy(), vals.copy()

    # ---- 诊断 API ----

    def stats(self) -> dict:
        if self._n == 0:
            return {'n_active': 0, 'norm_mean': 0.0, 'norm_max': 0.0}
        norms = np.linalg.norm(self._Q_values[:self._n], axis=1)
        return {
            'n_active': int(self._n),
            'norm_mean': float(norms.mean()),
            'norm_max': float(norms.max()),
        }

    def inspect_slot(self, slot_id: int) -> dict:
        idx = self._slot_to_idx.get(slot_id)
        if idx is None:
            return {'exists': False}
        return {
            'exists': True,
            'Q': self._Q_values[idx].copy(),
            'norm': float(np.linalg.norm(self._Q_values[idx])),
        }

    # ---- 内部 ----

    def _append(self, slot_id: int, value: float):
        if self._n >= self._capacity:
            self._grow()
        idx = self._n
        self._slot_ids[idx] = slot_id
        self._Q_values[idx] = value  # 广播到 N 维
        self._slot_to_idx[slot_id] = idx
        self._n += 1

    def _grow(self):
        new_cap = self._capacity * 2
        new_ids = np.empty(new_cap, dtype=np.int32)
        new_vals = np.empty((new_cap, self.N), dtype=np.float32)
        new_ids[:self._n] = self._slot_ids[:self._n]
        new_vals[:self._n] = self._Q_values[:self._n]
        self._slot_ids = new_ids
        self._Q_values = new_vals
        self._capacity = new_cap

    def _compact(self, alive_mask: np.ndarray):
        keep_ids = self._slot_ids[:self._n][alive_mask]
        keep_vals = self._Q_values[:self._n][alive_mask]
        n_keep = len(keep_ids)
        self._slot_ids[:n_keep] = keep_ids
        self._Q_values[:n_keep] = keep_vals
        self._n = n_keep
        self._slot_to_idx = {int(self._slot_ids[i]): i for i in range(self._n)}
