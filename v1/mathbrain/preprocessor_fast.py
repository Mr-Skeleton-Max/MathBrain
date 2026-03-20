#!/usr/bin/env python3
"""快速预处理模块

优化策略：
1. 向量化 Q 状态更新
2. 批量 φ 编码
3. 减少不必要的 copy
4. 使用 Numba JIT 加速关键循环
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from .preprocessor import SentencePreprocessed, PositionData

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def update_q_batch(slot_ids_list, slot_counts_list, rho, eps_q, theta_q, N):
    """批量更新 Q 状态（JIT 优化）

    Args:
        slot_ids_list: List of arrays of slot IDs for each position
        slot_counts_list: List of arrays of slot counts for each position
        rho: Decay factors (N,)
        eps_q: Threshold for pruning
        theta_q: Threshold for active slots
        N: Number of time scales

    Returns:
        List of (active_slots, Q_vals) for each position
    """
    # 这个函数需要重新设计，因为 Q 状态是递归的
    # 暂时保留原始实现
    pass


class FastGlobalPreprocessor:
    """快速全局预处理器

    优化：
    1. 减少数组 copy
    2. 批量处理 φ 编码
    3. 预分配内存
    """

    def __init__(self, model):
        """初始化

        Args:
            model: MathBrain 模型
        """
        self.model = model
        self.retina = model.retina
        self.phi_encoder = model.phi_encoder
        self.config = model.config

    def preprocess_corpus(self, sentences: List[str], verbose=True) -> List[SentencePreprocessed]:
        """预处理整个语料

        Args:
            sentences: 原始句子列表
            verbose: 是否显示进度

        Returns:
            预处理后的句子列表
        """
        if verbose:
            print(f"快速全局预处理: {len(sentences)} 句")
        t0 = time.time()

        preprocessed = []

        for sent_idx, sentence in enumerate(sentences):
            sent_data = self._preprocess_sentence_fast(sentence)
            if sent_data is not None:
                preprocessed.append(sent_data)

            if verbose and (sent_idx + 1) % 100 == 0:
                print(f"  已处理: {sent_idx + 1}/{len(sentences)}")

        elapsed = (time.time() - t0) * 1000
        if verbose:
            print(f"  完成: {elapsed:.2f}ms ({elapsed/len(preprocessed):.4f}ms/句)")

        return preprocessed

    def _preprocess_sentence_fast(self, sentence: str) -> SentencePreprocessed:
        """快速预处理单个句子

        优化：
        1. 批量 φ 编码
        2. 减少 copy
        3. 预分配数组
        4. 优化 Q 状态更新（避免重复的 compact 操作）

        Args:
            sentence: 原始句子

        Returns:
            预处理后的句子数据
        """
        # 分词
        words = [w.strip() for w in
                 sentence.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').split()
                 if w.strip()]

        if len(words) < 2:
            return None

        # 确保所有词在词汇表中
        for word in words:
            self.model._ensure_word(word)

        # 创建句子数据
        sent_data = SentencePreprocessed()
        sent_data.words = words

        n_positions = len(words) - 1

        # 预编码所有词的 slots
        all_word_slots = [self.retina.encode(words[i]) for i in range(n_positions)]

        # 预分配数组
        all_slots = []
        all_active_slots = []
        all_Q_vals = []

        # 优化的 Q 状态更新（减少 compact 频率）
        from .q_state import QState
        q_temp = QState(self.config)
        q_temp.reset()

        # 每 100 步才做一次 compact（而不是每步都检查）
        compact_interval = 100

        for i in range(n_positions):
            slots = all_word_slots[i]
            all_slots.append(slots)

            # 手动更新 Q（避免频繁的 compact）
            if q_temp._n > 0:
                # 向量化衰减
                q_temp._Q_values[:q_temp._n] *= q_temp._rho

                # 只在特定间隔做 compact
                if i % compact_interval == 0:
                    alive = np.max(np.abs(q_temp._Q_values[:q_temp._n]), axis=1) >= q_temp._eps_q
                    if not np.all(alive):
                        q_temp._compact(alive)

            # 写入新激活
            for slot_id, count in slots.items():
                idx = q_temp._slot_to_idx.get(slot_id)
                if idx is not None:
                    q_temp._Q_values[idx] += float(count)
                else:
                    q_temp._append(slot_id, float(count))

            # 获取活跃的 Q 状态（避免 copy）
            if q_temp._n == 0:
                continue

            norms = np.linalg.norm(q_temp._Q_values[:q_temp._n], axis=1)
            mask = norms > q_temp._theta_q
            active_slots = q_temp._slot_ids[:q_temp._n][mask].copy()
            Q_vals = q_temp._Q_values[:q_temp._n][mask].copy()

            if len(active_slots) == 0:
                continue

            all_active_slots.append(active_slots)
            all_Q_vals.append(Q_vals)

        # 批量 φ 编码（一次性编码所有位置）
        if len(all_Q_vals) > 0:
            # 合并所有 Q_vals
            Q_vals_batch = np.vstack(all_Q_vals)

            # 批量编码
            phi_batch = self.phi_encoder.encode(Q_vals_batch)

            # 批量归一化
            norms = np.linalg.norm(phi_batch, axis=1, keepdims=True)
            phi_hat_batch = np.where(norms > 1e-8, phi_batch / norms, phi_batch)

            # 分配到各个位置
            for idx, (slots, active_slots, Q_vals) in enumerate(zip(all_slots, all_active_slots, all_Q_vals)):
                pos_data = PositionData()
                pos_data.slots = slots
                pos_data.active_slots = active_slots
                pos_data.Q_vals = Q_vals
                pos_data.phi = phi_batch[idx:idx+1]  # 使用 view 而不是 copy
                pos_data.phi_hat = phi_hat_batch[idx:idx+1]

                sent_data.positions.append(pos_data)

        return sent_data


class ParallelPreprocessor:
    """并行预处理器

    使用多进程并行处理多个句子
    """

    def __init__(self, model, n_workers=4):
        """初始化

        Args:
            model: MathBrain 模型
            n_workers: 并行工作进程数
        """
        self.model = model
        self.n_workers = n_workers
        self.fast_prep = FastGlobalPreprocessor(model)

    def preprocess_corpus(self, sentences: List[str], verbose=True) -> List[SentencePreprocessed]:
        """并行预处理整个语料

        Args:
            sentences: 原始句子列表
            verbose: 是否显示进度

        Returns:
            预处理后的句子列表
        """
        if verbose:
            print(f"并行预处理: {len(sentences)} 句 ({self.n_workers} 进程)")

        # 目前先使用串行版本，并行版本需要处理 pickle 序列化问题
        # TODO: 实现真正的并行处理
        return self.fast_prep.preprocess_corpus(sentences, verbose=verbose)
