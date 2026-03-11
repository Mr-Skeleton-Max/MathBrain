"""MathBrain 主类: 组装所有模块"""

import numpy as np

from .config import MathBrainConfig
from .retina import HashRetinaV2
from .q_state import QState
from .phi_encoder import create_phi_encoder
from .b_memory import BMemory, BMemoryLMS, BMemoryDelta, BMemoryEnergy
from .a_knowledge_v2 import AKnowledgeV2, BMemoryV2Wrapper
from . import inference
from .inference_fast import SparseMatrixDecoder


class MathBrain:
    """MathBrain 序列预测模型

    Hash V2 (trigram, K=65536, Sigmoid 多标签)
    + Cosine Chaos 编码 (D=64, 反向缩放 α)
    + 双通道 B/A 记忆 + Wake-Sleep 学习
    """

    def __init__(self, config: MathBrainConfig = None):
        self.config = config or MathBrainConfig()
        self.retina = HashRetinaV2(self.config)
        self.q = QState(self.config)
        self.phi_encoder = create_phi_encoder(self.config)

        # B 通道
        if self.config.B_MODE == "lms":
            self.b = BMemoryLMS(self.config)
        elif self.config.B_MODE == "delta":
            self.b = BMemoryDelta(self.config)
        elif self.config.B_MODE == "energy":
            self.b = BMemoryV2Wrapper(self.config, cp_rank=self.config.CP_RANK)
        else:
            self.b = BMemory(self.config)

        # A 通道 (slice-wise 矩阵分解, d 和 D 解耦)
        self.a = AKnowledgeV2(
            D=self.config.d,  # 使用 cfg.d（根据 PHI_MODE 自动计算）
            d=self.config.CP_RANK,
            K=self.config.K,
        )

        # 词汇表管理
        self.vocab: set = set()
        self.word_to_slots: dict = {}  # word → ndarray of slot_ids
        self._all_vocab_slots: set = set()  # 所有已知词的槽位并集

        # 优化的解码器（延迟初始化）
        self._decoder = None
        self._decoder_dirty = True  # 标记是否需要重建解码器

        self.train_count = 0
        self.wake_steps = 0

    # ==================== 词汇表 ====================

    def _ensure_word(self, word: str):
        """确保词在词汇表中"""
        if word not in self.vocab:
            self.vocab.add(word)
            slots = self.retina.get_slots(word)
            self.word_to_slots[word] = slots
            self._all_vocab_slots.update(slots.tolist())
            self._decoder_dirty = True  # 标记解码器需要重建

    # ==================== 训练 ====================

    def train_sentence(self, sentence: str):
        """训练一个句子 (Wake 阶段)"""
        words = sentence.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').split()
        words = [w.strip() for w in words if w.strip()]
        if len(words) < 2:
            return

        self.q.reset()

        for i in range(len(words) - 1):
            word = words[i]
            target = words[i + 1]
            self._ensure_word(word)
            self._ensure_word(target)

            # Q 更新
            slot_counts = self.retina.encode(word)
            self.q.update(slot_counts)

            # φ 编码
            active_slots, Q_vals = self.q.get_active()
            if len(active_slots) == 0:
                continue
            phi = self.phi_encoder.encode(Q_vals)

            # B 写入 (能量守恒写入在write内部计算B和A强度)
            target_counts = self.retina.encode(target)

            # 收集负样本候选槽位 (排除目标词)
            target_slot_set = set(target_counts.keys())
            neg_slots = self._all_vocab_slots - target_slot_set

            self.b.write(target_counts, active_slots, phi, None,
                         neg_slots, a_knowledge=self.a)
            self.wake_steps += 1

        self.train_count += 1

    def train_sentence_preprocessed(self, words: list, encoded_slots: list):
        """训练一个句子（使用预处理的数据）

        Args:
            words: 分词后的词列表
            encoded_slots: 预编码的slot列表
        """
        if len(words) < 2:
            return

        self.q.reset()

        for i in range(len(words) - 1):
            # Q 更新
            slot_counts = encoded_slots[i]
            self.q.update(slot_counts)

            # φ 编码
            active_slots, Q_vals = self.q.get_active()
            if len(active_slots) == 0:
                continue
            phi = self.phi_encoder.encode(Q_vals)

            # B 写入 (能量守恒写入在write内部计算B和A强度)
            target_counts = encoded_slots[i + 1]

            # 收集负样本候选槽位
            target_slot_set = set(target_counts.keys())
            neg_slots = self._all_vocab_slots - target_slot_set

            self.b.write(target_counts, active_slots, phi, None,
                         neg_slots, a_knowledge=self.a)
            self.wake_steps += 1

        self.train_count += 1

    def train_cached_positions(self, wake_positions: list):
        """训练一个句子（使用预计算的 Wake 位置特征，批量优化）"""
        if not wake_positions:
            self.train_count += 1
            return

        # 批量预计算 A embeddings（避免每个位置重复计算）
        if self.a.has_knowledge:
            self.b._precompute_a_embeddings(self.a)

        for pos in wake_positions:
            self.b.write(
                pos['target_counts'],
                pos['active_slots'],
                pos['phi'],
                None,
                None,
                a_knowledge=self.a,
                phi_hat=pos.get('phi_hat'),
                phi_proj=pos.get('phi_proj'),
                target_slots=pos.get('target_slots'),
            )
            self.wake_steps += 1

        self.train_count += 1

    # ==================== 预测 ====================

    def predict_next(self, context: list, k: int = 10) -> list:
        """预测下一个词 → [(word, score), ...]"""
        self.q.reset()
        for w in context:
            self._ensure_word(w)
            self.q.update(self.retina.encode(w))

        active_slots, Q_vals = self.q.get_active()
        if len(active_slots) == 0:
            return []

        phi = self.phi_encoder.encode(Q_vals)
        scores_B = self.b.predict(active_slots, phi)
        scores_A = self.a.predict(active_slots, phi)
        merged = inference.merge_logits(scores_B, scores_A)

        # 使用优化的解码器
        if self._decoder_dirty or self._decoder is None:
            self._decoder = SparseMatrixDecoder(
                self.vocab, self.word_to_slots, self.config.K
            )
            self._decoder_dirty = False

        return self._decoder.decode_top_k(merged, k)

    def predict_slots(self, active_slots: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """底层槽位分数 (供诊断用)"""
        scores_B = self.b.predict(active_slots, phi)
        scores_A = self.a.predict(active_slots, phi)
        return inference.merge_logits(scores_B, scores_A)

    # ==================== Sleep ====================

    def sleep(self, **kwargs) -> dict:
        """执行 Sleep 巩固 (v2 slice-wise)"""
        # 收集活跃槽位
        active = set()
        for slots in self.word_to_slots.values():
            active.update(slots.tolist())

        print(
            f"  [诊断] Sleep dispatch: active_arg=all_known_vocab_slots, "
            f"n_active={len(active)}, vocab_words={len(self.vocab)}"
        )

        vocab_words = sorted(self.vocab)
        sleep_backend = kwargs.pop('sleep_backend', 'mlx')
        if sleep_backend == 'numpy':
            from . import sleep_v2 as sleep_module
            result = sleep_module.consolidate_v2(
                self.b, self.a, active, self.config,
                cp_rank=self.config.CP_RANK,
                lambda_wd=self.config.LAMBDA_WD,
                retina=self.retina,
                vocab_words=vocab_words,
                **kwargs,
            )
        elif sleep_backend == 'mlx':
            from . import sleep_v2_mlx as sleep_module
            result = sleep_module.consolidate_v2_mlx(
                self.b, self.a, active, self.config,
                cp_rank=self.config.CP_RANK,
                lambda_wd=self.config.LAMBDA_WD,
                retina=self.retina,
                vocab_words=vocab_words,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown sleep_backend: {sleep_backend}")

        return {
            'n_new': result.n_new,
            'n_rehearsal': result.n_rehearsal,
            'epochs': result.epochs,
            'final_loss': result.final_loss,
            'success': result.success,
        }

    # ==================== 会话管理 ====================

    def reset_session(self):
        """重置 Q 状态 (不清除记忆)"""
        self.q.reset()

    def decay_and_prune(self, decay_factor: float = 0.95):
        """周期性 B 衰减 + 剪枝"""
        self.b.decay_and_prune(decay_factor)

    # ==================== 统计 ====================

    def get_stats(self) -> dict:
        return {
            'vocab_size': len(self.vocab),
            'train_count': self.train_count,
            'wake_steps': self.wake_steps,
            'b': self.b.stats(),
            'a': self.a.stats(),
            'q': self.q.stats(),
        }
