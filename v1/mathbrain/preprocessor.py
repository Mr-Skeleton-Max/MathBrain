#!/usr/bin/env python3
"""完整的预处理模块

两层预处理结构：
1. 全局预处理（整个训练过程只做一次）
   - HashRetina 编码
   - Q 状态序列（EMA 衰减）
   - φ 编码（Cosine Chaos）
   - φ 归一化

2. Sleep 周期内预处理（每个 Sleep 后，A 更新时）
   - φ 到 A 空间的投影
   - 源槽位的 A 嵌入
"""

import numpy as np
import time
from typing import List, Dict, Tuple


class SentencePreprocessed:
    """单个句子的预处理数据"""

    def __init__(self):
        self.words: List[str] = []
        self.positions: List[PositionData] = []

    def __len__(self):
        return len(self.positions)


class PositionData:
    """句子中单个位置的预处理数据"""

    def __init__(self):
        # 全局预处理（不变）
        self.slots: Dict[int, int] = {}  # HashRetina 编码
        self.active_slots: np.ndarray = None  # Q 的活跃槽位
        self.Q_vals: np.ndarray = None  # Q 状态值
        self.phi: np.ndarray = None  # φ 编码（Cosine Chaos）
        self.phi_hat: np.ndarray = None  # φ 归一化

        # Sleep 周期内预处理（A 更新后重算）
        self.dist_proj: np.ndarray = None  # φ 到 A 空间的投影
        self.src_embeddings: Dict[int, np.ndarray] = {}  # 源槽位的 A 嵌入


class GlobalPreprocessor:
    """全局预处理器

    对整个语料进行一次性预处理：
    - HashRetina 编码
    - Q 状态序列
    - φ 编码（Cosine Chaos）
    - φ 归一化
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

    def preprocess_corpus(self, sentences: List[str]) -> List[SentencePreprocessed]:
        """预处理整个语料

        Args:
            sentences: 原始句子列表

        Returns:
            预处理后的句子列表
        """
        print(f"全局预处理: {len(sentences)} 句")
        t0 = time.time()

        preprocessed = []

        for sent_idx, sentence in enumerate(sentences):
            sent_data = self._preprocess_sentence(sentence)
            if sent_data is not None:
                preprocessed.append(sent_data)

            if (sent_idx + 1) % 100 == 0:
                print(f"  已处理: {sent_idx + 1}/{len(sentences)}")

        elapsed = (time.time() - t0) * 1000
        print(f"  完成: {elapsed:.2f}ms ({elapsed/len(preprocessed):.4f}ms/句)")

        return preprocessed

    def _preprocess_sentence(self, sentence: str) -> SentencePreprocessed:
        """预处理单个句子

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

        # 模拟 Q 状态更新，记录每个位置的状态
        # 创建临时 Q 状态（不影响模型的 Q）
        from .q_state import QState
        q_temp = QState(self.config)
        q_temp.reset()

        for i in range(len(words) - 1):
            word = words[i]

            # HashRetina 编码
            slots = self.retina.encode(word)

            # 更新 Q 状态
            q_temp.update(slots)

            # 获取活跃的 Q 状态
            active_slots, Q_vals = q_temp.get_active()

            if len(active_slots) == 0:
                continue

            # φ 编码（Cosine Chaos）
            phi = self.phi_encoder.encode(Q_vals)

            # φ 归一化
            norms = np.linalg.norm(phi, axis=1, keepdims=True)
            phi_hat = np.where(norms > 1e-8, phi / norms, phi)

            # 创建位置数据
            pos_data = PositionData()
            pos_data.slots = slots
            pos_data.active_slots = active_slots.copy()
            pos_data.Q_vals = Q_vals.copy()
            pos_data.phi = phi.copy()
            pos_data.phi_hat = phi_hat.copy()

            sent_data.positions.append(pos_data)

        return sent_data


class SleepCyclePreprocessor:
    """Sleep 周期内预处理器

    在每个 Sleep 周期后，A 参数更新时重新计算：
    - φ 到 A 空间的投影
    - 源槽位的 A 嵌入
    """

    def __init__(self, model):
        """初始化

        Args:
            model: MathBrain 模型
        """
        self.model = model
        self.a = model.a

    def preprocess_for_a(self, preprocessed: List[SentencePreprocessed]):
        """为 A 预测预处理

        Args:
            preprocessed: 全局预处理后的句子列表
        """
        if not self.a.has_knowledge:
            return

        print("Sleep 周期预处理（A 嵌入）...")
        t0 = time.time()

        # 预计算所有槽位的 A 嵌入（只计算一次）
        e_v_cache = {}  # src → E[src] @ W_V
        e_q_cache = {}  # tgt → E[tgt] @ W_Q

        for slot_id, emb in self.a.E.items():
            e_v_cache[slot_id] = (emb @ self.a.W_V).astype(np.float32)
            e_q_cache[slot_id] = (emb @ self.a.W_Q).astype(np.float32)

        # 缓存到模型中（供快速预测使用）
        self.model._a_e_v_cache = e_v_cache
        self.model._a_e_q_cache = e_q_cache

        total_positions = 0

        for sent_data in preprocessed:
            for pos_data in sent_data.positions:
                # φ 到 A 空间的投影
                # dist_proj = (P.T @ phi_hat.T).T
                dist_proj = (self.a.P.T @ pos_data.phi_hat.T).T
                pos_data.dist_proj = dist_proj.astype(np.float32)

                # 源槽位的 A 嵌入（从缓存中获取）
                pos_data.src_embeddings = {}
                for src in pos_data.active_slots:
                    src = int(src)
                    if src in e_v_cache:
                        pos_data.src_embeddings[src] = e_v_cache[src]

                total_positions += 1

        elapsed = (time.time() - t0) * 1000
        print(f"  完成: {elapsed:.2f}ms ({elapsed/total_positions:.4f}ms/位置)")


class PreprocessedTrainer:
    """使用预处理数据的训练器"""

    def __init__(self, model):
        """初始化

        Args:
            model: MathBrain 模型
        """
        self.model = model

    def train_position(self, pos_data: PositionData, target_slots: Dict[int, int]):
        """训练单个位置

        Args:
            pos_data: 位置的预处理数据
            target_slots: 目标词的槽位编码
        """
        # B 预测（直接使用预处理的 phi）
        scores_B = self.model.b.predict(pos_data.active_slots, pos_data.phi)

        # A 预测（使用预处理的 A 嵌入）
        scores_A = self._predict_a_fast(pos_data)

        # 合并预测
        merged_scores = scores_B + scores_A

        # 目标槽位
        target_slot_set = set(target_slots.keys())
        neg_slots = self.model._all_vocab_slots - target_slot_set

        # 写入 B
        self.model.b.write(target_slots, pos_data.active_slots, pos_data.phi,
                          merged_scores, neg_slots, a_knowledge=self.model.a)

    def _predict_a_fast(self, pos_data: PositionData) -> np.ndarray:
        """快速 A 预测（使用预处理数据和缓存）

        Args:
            pos_data: 位置的预处理数据

        Returns:
            slot_scores (K,)
        """
        scores = np.zeros(self.model.config.K, dtype=np.float32)

        if not self.model.a.has_knowledge or len(pos_data.active_slots) == 0:
            return scores

        # 计算 context vector c（使用预处理的嵌入）
        c = np.zeros(self.model.a.d, dtype=np.float32)

        for i, src in enumerate(pos_data.active_slots):
            src = int(src)
            if src in pos_data.src_embeddings:
                e_v = pos_data.src_embeddings[src]
                c += e_v * pos_data.dist_proj[i]

        if np.linalg.norm(c) < 1e-6:
            return scores

        # l_A[tgt] = (E[tgt]·W_Q) · c（使用缓存的 e_q）
        e_q_cache = getattr(self.model, '_a_e_q_cache', {})
        if e_q_cache:
            # 使用缓存
            for tgt, e_q in e_q_cache.items():
                scores[tgt] = np.dot(e_q, c)
        else:
            # 回退到原始方法
            for tgt, emb in self.model.a.E.items():
                e_q = emb @ self.model.a.W_Q
                scores[tgt] = np.dot(e_q, c)

        return scores

    def train_sentence(self, sent_data: SentencePreprocessed):
        """训练单个句子

        Args:
            sent_data: 预处理后的句子数据
        """
        for i, pos_data in enumerate(sent_data.positions):
            # 获取目标词的槽位
            target_word = sent_data.words[i + 1]
            target_slots = self.model.retina.encode(target_word)

            # 训练这个位置
            self.train_position(pos_data, target_slots)

        self.model.wake_steps += 1

    def train_corpus(self, preprocessed: List[SentencePreprocessed]) -> float:
        """训练整个语料

        Args:
            preprocessed: 预处理后的句子列表

        Returns:
            训练时间（毫秒）
        """
        t0 = time.time()

        for sent_data in preprocessed:
            self.train_sentence(sent_data)

        elapsed = (time.time() - t0) * 1000
        return elapsed
