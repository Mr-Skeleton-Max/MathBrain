"""B 通道: 短期记忆 (per-source 连续存储 + 惰性衰减)"""

import numpy as np

from .config import MathBrainConfig

_INIT_CAP = 256  # 每个 source 的初始 target 容量（优化：从 128 增加到 256，减少扩容次数）


class BMemory:
    """短期记忆 B 通道

    Per-source 连续存储:
      _vecs[src]  : ndarray(cap, D)   — 连接向量 (连续内存)
      _tgts[src]  : ndarray(cap,)     — 目标槽位
      _times[src] : ndarray(cap,)     — 时间戳
      _cnt[src]   : int               — 有效条目数
      _tpos[src]  : dict{tgt→pos}     — 目标位置索引

    predict 用 matmul per source，O(n_active) 次 BLAS 调用。
    """

    def __init__(self, config: MathBrainConfig = None):
        cfg = config or MathBrainConfig()
        self.D = cfg.D_COSINE
        self.K = cfg.K
        self._eta = cfg.ETA
        self._eps_prune = cfg.EPS_PRUNE
        self._neg_suppress = cfg.NEG_SUPPRESS
        self._neg_threshold = cfg.NEG_THRESHOLD
        self._decay_table = cfg.decay_table
        self._max_dt = len(self._decay_table)
        self._max_b = cfg.MAX_B_ENTRIES
        self._conservation = cfg.B_ENERGY_CONSERVATION  # 守恒开关

        # Per-source storage
        self._vecs: dict = {}    # src → ndarray(cap, D)
        self._tgts: dict = {}    # src → ndarray(cap,)
        self._times: dict = {}   # src → ndarray(cap,)
        self._cnt: dict = {}     # src → int
        self._tpos: dict = {}    # src → {tgt: pos}

        self.n_b = 0
        self.global_step = 0

    # ==================== 预测 ====================

    def predict(self, active_slots: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Per-source matmul 预测 → slot_scores (K,)"""
        scores = np.zeros(self.K, dtype=np.float32)
        if len(active_slots) == 0 or self.n_b == 0:
            return scores

        step = self.global_step
        decay_table = self._decay_table
        max_dt = self._max_dt
        _vecs = self._vecs
        _tgts = self._tgts
        _times = self._times
        _cnt = self._cnt

        for i in range(len(active_slots)):
            src = int(active_slots[i])
            cnt = _cnt.get(src, 0)
            if cnt == 0:
                continue

            vecs = _vecs[src][:cnt]
            tgts = _tgts[src][:cnt]
            times = _times[src][:cnt]

            # 惰性衰减 — 无条件执行，避免 np.any 开销
            dts = step - times
            need = (dts > 0) & (dts < max_dt)
            decay_f = np.where(need, decay_table[np.minimum(dts, max_dt - 1)], 1.0)
            vecs *= decay_f[:, np.newaxis]
            times[:] = np.where(need, step, times)

            # 单次 matmul
            dots = vecs @ phi[i]
            np.add.at(scores, tgts, dots)

        return scores / float(max(1, len(active_slots)))

    # ==================== 写入 ====================

    def write(self, target_slot_counts: dict, active_slots: np.ndarray,
              phi: np.ndarray, slot_scores: np.ndarray,
              vocab_slots: set = None):
        """纯赫布写入: ΔB = η·I(target) - λ·B (衰减在 predict 时惰性执行)"""
        self.global_step += 1
        step = self.global_step
        target_set = set(target_slot_counts.keys())
        n_src = len(active_slots)
        if n_src == 0:
            return

        # 正样本: 纯赫布生长 err = η
        tgt_list = []
        err_list = []
        for tgt in target_set:
            tgt_list.append(int(tgt))
            err_list.append(self._eta)

        # 无负抑制

        if not tgt_list:
            return

        decay_table = self._decay_table
        max_dt = self._max_dt
        _vecs = self._vecs
        _times = self._times
        _tpos = self._tpos

        n_tgt = len(tgt_list)
        tgt_arr = np.array(tgt_list, dtype=np.int32)
        err_arr = np.array(err_list, dtype=np.float32)

        # 翻转循环: for src → 批量处理所有 tgt
        for i in range(n_src):
            src = int(active_slots[i])
            phi_i = phi[i]  # (D,)
            tp = _tpos.get(src)

            # 分离已有 vs 新建
            if tp is not None:
                exist_pos = np.empty(n_tgt, dtype=np.int32)
                exist_mask = np.zeros(n_tgt, dtype=bool)
                for k in range(n_tgt):
                    p = tp.get(tgt_list[k])
                    if p is not None:
                        exist_pos[k] = p
                        exist_mask[k] = True

                # 批量更新已有条目
                n_exist = exist_mask.sum()
                if n_exist > 0:
                    epos = exist_pos[exist_mask]
                    eerr = err_arr[exist_mask]
                    # 衰减
                    dts = step - _times[src][epos]
                    need = (dts > 0) & (dts < max_dt)
                    if np.any(need):
                        _vecs[src][epos[need]] *= decay_table[dts[need], np.newaxis]
                    # delta = err * phi_i → (n_exist, D)
                    _vecs[src][epos] += eerr[:, np.newaxis] * phi_i
                    _times[src][epos] = step

                # 新条目
                if n_exist < n_tgt:
                    new_mask = ~exist_mask
                    for k in np.where(new_mask)[0]:
                        self._alloc_entry(src, tgt_list[k],
                                          err_arr[k] * phi_i, step)
            else:
                # src 完全没有条目，全部新建
                for k in range(n_tgt):
                    self._alloc_entry(src, tgt_list[k],
                                      err_arr[k] * phi_i, step)

    def _alloc_entry(self, src: int, tgt: int, vec: np.ndarray, step: int):
        if src not in self._cnt:
            cap = _INIT_CAP
            self._vecs[src] = np.zeros((cap, self.D), dtype=np.float32)
            self._tgts[src] = np.empty(cap, dtype=np.int32)
            self._times[src] = np.zeros(cap, dtype=np.int32)
            self._cnt[src] = 0
            self._tpos[src] = {}

        cnt = self._cnt[src]
        cap = len(self._tgts[src])
        if cnt >= cap:
            new_cap = cap * 2
            new_v = np.zeros((new_cap, self.D), dtype=np.float32)
            new_v[:cap] = self._vecs[src]
            self._vecs[src] = new_v
            new_t = np.empty(new_cap, dtype=np.int32)
            new_t[:cap] = self._tgts[src]
            self._tgts[src] = new_t
            new_tm = np.zeros(new_cap, dtype=np.int32)
            new_tm[:cap] = self._times[src]
            self._times[src] = new_tm

        self._vecs[src][cnt] = vec
        self._tgts[src][cnt] = tgt
        self._times[src][cnt] = step
        self._tpos[src][tgt] = cnt
        self._cnt[src] = cnt + 1
        self.n_b += 1

    # ==================== 维护 ====================

    def compact(self):
        """裁剪小范数条目"""
        threshold = self._eps_prune * 10
        total = 0
        for src in list(self._cnt.keys()):
            cnt = self._cnt[src]
            if cnt == 0:
                continue
            norms = np.linalg.norm(self._vecs[src][:cnt], axis=1)
            keep = norms > threshold
            if not np.all(keep):
                self._compact_source(src, keep)
            total += self._cnt[src]
        self.n_b = total

    def _compact_source(self, src: int, keep_mask: np.ndarray):
        cnt = self._cnt[src]
        indices = np.where(keep_mask)[0]
        n = len(indices)
        if n == cnt:
            return
        if n == 0:
            del self._vecs[src], self._tgts[src], self._times[src]
            del self._cnt[src], self._tpos[src]
            return
        self._vecs[src][:n] = self._vecs[src][indices]
        self._tgts[src][:n] = self._tgts[src][indices]
        self._times[src][:n] = self._times[src][indices]
        self._cnt[src] = n
        self._tpos[src] = {int(self._tgts[src][i]): i for i in range(n)}

    def clear(self):
        """Sleep 后清零"""
        self._vecs.clear()
        self._tgts.clear()
        self._times.clear()
        self._cnt.clear()
        self._tpos.clear()
        self.n_b = 0

    def get_entries(self) -> list:
        """供 Sleep 采样: [(src, tgt, vec), ...]"""
        threshold = self._eps_prune * 10
        result = []
        for src, cnt in self._cnt.items():
            for i in range(cnt):
                if np.linalg.norm(self._vecs[src][i]) > threshold:
                    result.append((src, int(self._tgts[src][i]),
                                   self._vecs[src][i].copy()))
        return result

    def decay_and_prune(self, decay_factor: float = 0.95):
        """周期性全局衰减 + 剪枝"""
        total = 0
        for src in list(self._cnt.keys()):
            cnt = self._cnt[src]
            if cnt == 0:
                continue
            self._vecs[src][:cnt] *= decay_factor
            norms = np.linalg.norm(self._vecs[src][:cnt], axis=1)
            keep = norms > self._eps_prune
            if not np.all(keep):
                self._compact_source(src, keep)
            total += self._cnt.get(src, 0)
        self.n_b = total

    # ---- 诊断 API ----

    def stats(self) -> dict:
        if self.n_b == 0:
            return {'n_entries': 0, 'n_src': 0, 'norm_mean': 0.0}
        all_norms = []
        conns = []
        for src, cnt in self._cnt.items():
            if cnt > 0:
                all_norms.append(np.linalg.norm(self._vecs[src][:cnt], axis=1))
                conns.append(cnt)
        norms = np.concatenate(all_norms) if all_norms else np.array([0.0])
        return {
            'n_entries': self.n_b,
            'n_src': len(conns),
            'avg_connections': float(np.mean(conns)) if conns else 0,
            'norm_mean': float(norms.mean()),
            'norm_median': float(np.median(norms)),
        }

    def inspect_connection(self, src: int, tgt: int) -> dict:
        tpos = self._tpos.get(src)
        if tpos is None or tgt not in tpos:
            return {'exists': False}
        p = tpos[tgt]
        return {
            'exists': True,
            'vec': self._vecs[src][p].copy(),
            'norm': float(np.linalg.norm(self._vecs[src][p])),
            'time': int(self._times[src][p]),
        }

    def top_connections(self, slot_id: int, k: int = 10) -> list:
        cnt = self._cnt.get(slot_id, 0)
        if cnt == 0:
            return []
        norms = np.linalg.norm(self._vecs[slot_id][:cnt], axis=1)
        top_k = np.argsort(norms)[-k:][::-1]
        return [(int(self._tgts[slot_id][i]), float(norms[i])) for i in top_k]


class BMemoryLMS(BMemory):
    """LMS 自适应写入的 B 通道

    更新规则: Δb_{s→k} = η·I(k=t)·φ_s − μ·(φ_s·b_{s→k})·φ_s − γ·b_{s→k}

    其中 μ = λ_lms / ‖φ‖² 是归一化后的惩罚强度，使参数与编码维度无关。

    - η·I(k=t)·φ_s : 赫布拉力（仅真实目标）
    - μ·(φ·b)·φ    : LMS 投影惩罚（隐式白化，消除保底相似度）
    - γ·b           : 全局权重衰减（Ridge 正则 / GC，复用惰性 decay_table）

    稳定性条件: μ·‖φ‖² = λ_lms < 2·η
    均衡: b_k = (η/μ)·(C + γ/μ·I)⁻¹·μ_k  (Wiener 滤波器 + Ridge)
    """

    def __init__(self, config: MathBrainConfig = None):
        super().__init__(config)
        cfg = config or MathBrainConfig()
        self._lambda_lms = cfg.LAMBDA_LMS

    def write(self, target_slot_counts: dict, active_slots: np.ndarray,
              phi: np.ndarray, slot_scores: np.ndarray,
              vocab_slots: set = None):
        """LMS 写入: 对所有已有条目施加投影惩罚，仅对真实目标施加拉力"""
        self.global_step += 1
        step = self.global_step
        target_set = set(target_slot_counts.keys())
        n_src = len(active_slots)
        if n_src == 0:
            return

        eta = self._eta
        lam = self._lambda_lms
        decay_table = self._decay_table
        max_dt = self._max_dt
        _vecs = self._vecs
        _times = self._times
        _cnt = self._cnt
        _tpos = self._tpos

        for i in range(n_src):
            src = int(active_slots[i])
            phi_i = phi[i]  # (D,)
            cnt = _cnt.get(src, 0)

            if cnt > 0:
                vecs = _vecs[src][:cnt]
                times = _times[src][:cnt]

                # 1. 惰性 γ 衰减 (Ridge 正则的离散化)
                dts = step - times
                need = (dts > 0) & (dts < max_dt)
                if np.any(need):
                    decay_f = decay_table[np.minimum(dts[need], max_dt - 1)]
                    vecs[need] *= decay_f[:, np.newaxis]
                    times[need] = step

                # 2. LMS 投影惩罚: 对所有已有条目
                #    归一化: μ = λ / ‖φ‖² 使惩罚强度与编码维度无关
                phi_sq = float(phi_i @ phi_i)  # ‖φ‖²
                if phi_sq > 1e-8:
                    mu = lam / phi_sq
                else:
                    mu = 0.0
                dots = vecs @ phi_i          # (cnt,) — per-entry logits
                vecs -= mu * dots[:, np.newaxis] * phi_i  # O(cnt × D)

                # 3. 赫布拉力: 仅对真实目标
                tp = _tpos.get(src)
                if tp is not None:
                    for tgt_slot in target_set:
                        tgt = int(tgt_slot)
                        pos = tp.get(tgt)
                        if pos is not None:
                            _vecs[src][pos] += eta * phi_i
                            _times[src][pos] = step
                        else:
                            self._alloc_entry(src, tgt, eta * phi_i, step)
                else:
                    for tgt_slot in target_set:
                        self._alloc_entry(src, int(tgt_slot),
                                          eta * phi_i, step)
            else:
                # src 完全没有条目，全部新建 (首次见到)
                for tgt_slot in target_set:
                    self._alloc_entry(src, int(tgt_slot),
                                      eta * phi_i, step)


class BMemoryEnergy(BMemory):
    """能量守恒写入的 B 通道

    核心思想:
      每个 (src, tgt) 连接有能量预算 1, 写入量与剩余能量成正比:
        Δb = η × max(0, 1 − ‖b_{s→t}‖) × φ̂

    其中 strength = ‖b_{s→t}‖ (连接向量的范数 = 存储能量)

    自停止机制:
      - ‖b‖ ≈ 0 (新连接): 写入量 ≈ η (全速)
      - ‖b‖ ≈ 1 (满能量): 写入量 ≈ 0 (饱和停写)
      - 非目标连接: 不写入，仅靠全局衰减自然消退

    φ 归一化: 写入和预测均使用 φ̂ = φ/‖φ‖, 使:
      - 写入方向为单位向量，能量增量可控
      - 不同编码维度下行为一致

    与 A 的协作:
      - 预测时乘法合并: score = A_score × (1 + B_score)
      - B 在 A 的基础上做重分配，不创造新能量
      - B 的写入独立于 A (纯能量守恒)
    """

    def __init__(self, config: MathBrainConfig = None):
        super().__init__(config)
        self._cached_phi_hat = None  # 缓存归一化的 φ

    def predict(self, active_slots: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """归一化 φ 后调用父类 predict，并缓存 phi_hat"""
        norms = np.linalg.norm(phi, axis=1, keepdims=True)
        phi_hat = np.where(norms > 1e-8, phi / norms, phi)
        self._cached_phi_hat = phi_hat  # 缓存供 write 使用
        return super().predict(active_slots, phi_hat)

    def write(self, target_slot_counts: dict, active_slots: np.ndarray,
              phi: np.ndarray, slot_scores: np.ndarray,
              vocab_slots: set = None, a_knowledge=None):
        """统一突触能量写入 (双向校准) - 优化版本

        gap = 1 - (φ̂·b[src][tgt] + φ̂·A[src][tgt])
        gap > 0: B 正向补偿 (A 欠冲)
        gap < 0: B 负向抑制 (A 过冲)
        总强度始终趋近 1, 压缩映射保证收敛。

        优化：
        1. 批量归一化 φ
        2. 预计算 A 的 src embeddings
        3. 提前过滤无效写入
        """
        self.global_step += 1
        step = self.global_step
        target_list = list(target_slot_counts.keys())
        n_src = len(active_slots)
        n_tgt = len(target_list)

        if n_src == 0 or n_tgt == 0:
            return

        eta = self._eta
        decay_table = self._decay_table
        max_dt = self._max_dt
        _vecs = self._vecs
        _times = self._times
        _tpos = self._tpos
        _cnt = self._cnt

        # 使用缓存的 phi_hat（如果可用）
        if self._cached_phi_hat is not None and len(self._cached_phi_hat) == n_src:
            phi_hat = self._cached_phi_hat
        else:
            phi_norms = np.linalg.norm(phi, axis=1, keepdims=True)
            phi_hat = np.where(phi_norms > 1e-8, phi / phi_norms, phi)

        # 预计算: A 的距离投影 (每个 src 一次)
        has_a = (a_knowledge is not None and a_knowledge.has_knowledge
                 and a_knowledge._slot_to_idx is not None)
        src_projs = {}  # i → 归一化后的局部投影
        if has_a:
            for i in range(n_src):
                src_projs[i] = phi_hat[i]

        # 主循环
        for i in range(n_src):
            src = int(active_slots[i])
            phi_hat_i = phi_hat[i]

            # 惰性衰减
            cnt = _cnt.get(src, 0)
            if cnt > 0:
                vecs = _vecs[src][:cnt]
                times = _times[src][:cnt]
                dts = step - times
                need = (dts > 0) & (dts < max_dt)
                if np.any(need):
                    decay_f = decay_table[np.minimum(dts[need], max_dt - 1)]
                    vecs[need] *= decay_f[:, np.newaxis]
                    times[need] = step

            tp = _tpos.get(src)
            proj_i = src_projs.get(i)

            # 计算 gaps，提前过滤
            significant_writes = []

            for tgt in target_list:
                tgt = int(tgt)

                # B 强度
                b_strength = 0.0
                if tp is not None and tgt in tp:
                    b_strength = float(phi_hat_i @ _vecs[src][tp[tgt]])

                # A 强度 (使用预分配缓存)
                a_strength = 0.0
                if has_a and proj_i is not None:
                    a_strength = a_knowledge.get_a_strength(src, tgt, proj_i)

                gap = 1.0 - (a_strength + b_strength)

                if abs(gap) > 1e-6 / eta:
                    significant_writes.append((tgt, gap))

            for tgt, gap in significant_writes:
                gain = eta * gap

                if tp is not None and tgt in tp:
                    _vecs[src][tp[tgt]] += gain * phi_hat_i
                    _times[src][tp[tgt]] = step
                else:
                    self._alloc_entry(src, tgt, gain * phi_hat_i.copy(), step)

                # 守恒机制：衰减其他连接
                if self._conservation and tp is not None and len(tp) > 1:
                    # 计算当前总强度
                    cnt = self._cnt[src]
                    if cnt > 1:
                        vecs = _vecs[src][:cnt]
                        current_strengths = phi_hat_i @ vecs.T
                        total_strength = float(np.sum(current_strengths))

                        # 衰减因子 = (total - gain) / total
                        if total_strength > 1e-8:
                            decay_factor = max(0.0, (total_strength - gain) / total_strength)
                            # 对所有非目标连接应用衰减
                            tgt_pos = tp[tgt]
                            for pos in range(cnt):
                                if pos != tgt_pos:
                                    _vecs[src][pos] *= decay_factor


class BMemoryDelta(BMemory):
    """Delta Rule 写入的 B 通道 (MathBrain.md §3.3)

    更新规则: ΔB[v,i,m] = η · e_v · φ[i,m]
    其中 e_v = δ(v,v*) - p(v), p = layernorm_softmax(slot_scores)

    - e_{v*} = 1 - p(v*): 正确 token 的正向强化，越准写入越小（自停止）
    - e_v = -p(v): 错误预测的负向抑制，与误导程度成正比
    - Σ e_v = 0: 净守恒（概率归一）

    全局衰减由 predict() 惰性 decay_table 实现。
    """

    def __init__(self, config: MathBrainConfig = None):
        super().__init__(config)
        cfg = config or MathBrainConfig()
        self._gamma_ln = cfg.GAMMA_LN
        self._K = cfg.K

    def write(self, target_slot_counts: dict, active_slots: np.ndarray,
              phi: np.ndarray, slot_scores: np.ndarray,
              vocab_slots: set = None):
        """Delta Rule 写入: e_v = δ(v,v*) - p(v)"""
        self.global_step += 1
        step = self.global_step
        target_set = set(target_slot_counts.keys())
        n_src = len(active_slots)
        if n_src == 0:
            return

        eta = self._eta
        decay_table = self._decay_table
        max_dt = self._max_dt
        _vecs = self._vecs
        _times = self._times
        _tpos = self._tpos
        _cnt = self._cnt

        # 计算概率分布 p(v) = layernorm_softmax(slot_scores)
        mu = slot_scores.mean()
        sigma = slot_scores.std() + 1e-8
        normed = self._gamma_ln * (slot_scores - mu) / sigma
        normed -= normed.max()
        exp_x = np.exp(normed)
        probs = exp_x / (exp_x.sum() + 1e-8)  # (K,)

        # 计算误差信号 e_v
        # 对于 target slots: e = 1 - p(v)
        # 对于其他有显著 p 的 slots: e = -p(v)
        # 稀疏写入: 只处理 |e| > eps 的条目
        eps_e = 1e-3

        # 收集需要写入的 (slot, error) 对
        write_slots = []
        write_errs = []

        # 正样本: target slots
        for tgt in target_set:
            tgt = int(tgt)
            e = 1.0 - probs[tgt]
            if abs(e) > eps_e:
                write_slots.append(tgt)
                write_errs.append(e)

        # 负样本: 所有 p(v) 显著的 non-target slots
        # 找 p > eps_e 的 slots (稀疏: softmax 后绝大多数 ≈ 0)
        sig_mask = probs > eps_e
        for v in np.where(sig_mask)[0]:
            v = int(v)
            if v not in target_set:
                write_slots.append(v)
                write_errs.append(-float(probs[v]))

        if not write_slots:
            return

        n_write = len(write_slots)
        err_arr = np.array(write_errs, dtype=np.float32)

        for i in range(n_src):
            src = int(active_slots[i])
            phi_i = phi[i]  # (D,)
            tp = _tpos.get(src)

            for k in range(n_write):
                tgt = write_slots[k]
                delta_vec = eta * err_arr[k] * phi_i

                if tp is not None:
                    pos = tp.get(tgt)
                    if pos is not None:
                        # 惰性衰减
                        dt = step - _times[src][pos]
                        if 0 < dt < max_dt:
                            _vecs[src][pos] *= decay_table[dt]
                        _vecs[src][pos] += delta_vec
                        _times[src][pos] = step
                    else:
                        self._alloc_entry(src, tgt, delta_vec.copy(), step)
                else:
                    self._alloc_entry(src, tgt, delta_vec.copy(), step)
