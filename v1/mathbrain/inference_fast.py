"""优化的推理解码模块"""

import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict


class FastDecoder:
    """优化的词汇表解码器

    使用反向索引：slot → [(word_idx, weight)]
    时间复杂度：O(V) → O(K_active)
    """

    def __init__(self, vocab: set, word_to_slots: dict):
        # 构建词列表和索引
        self.word_list = sorted(vocab)  # 保持确定性顺序
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}

        # 构建反向索引：slot → [(word_idx, weight)]
        self.slot_to_words = defaultdict(list)

        for word, slots in word_to_slots.items():
            if word not in self.word_to_idx:
                continue
            word_idx = self.word_to_idx[word]
            weight = 1.0 / len(slots)  # 算术平均权重

            for slot in slots:
                self.slot_to_words[int(slot)].append((word_idx, weight))

    def decode(self, slot_scores: np.ndarray) -> dict:
        """槽位分数 → 词分数

        Args:
            slot_scores: (K,) 槽位分数数组

        Returns:
            {word: score} 字典
        """
        word_scores = np.zeros(len(self.word_list), dtype=np.float32)

        # 仅遍历非零槽位
        nonzero_slots = np.nonzero(slot_scores)[0]

        for slot in nonzero_slots:
            slot = int(slot)
            if slot in self.slot_to_words:
                score = slot_scores[slot]
                for word_idx, weight in self.slot_to_words[slot]:
                    word_scores[word_idx] += score * weight

        # 转为字典
        return {self.word_list[i]: float(word_scores[i])
                for i in range(len(self.word_list))}

    def decode_top_k(self, slot_scores: np.ndarray, k: int = 10) -> list:
        """直接返回 Top-K，避免构建完整字典

        Returns:
            [(word, score), ...] 按分数降序
        """
        word_scores = np.zeros(len(self.word_list), dtype=np.float32)

        nonzero_slots = np.nonzero(slot_scores)[0]
        for slot in nonzero_slots:
            slot = int(slot)
            if slot in self.slot_to_words:
                score = slot_scores[slot]
                for word_idx, weight in self.slot_to_words[slot]:
                    word_scores[word_idx] += score * weight

        # 使用 argpartition 找 Top-K（O(n) 而非 O(n log n)）
        if k >= len(word_scores):
            top_indices = np.argsort(word_scores)[::-1]
        else:
            # argpartition: O(n)，只需要 top-k
            top_indices = np.argpartition(word_scores, -k)[-k:]
            # 对 top-k 排序
            top_indices = top_indices[np.argsort(word_scores[top_indices])[::-1]]

        return [(self.word_list[i], float(word_scores[i]))
                for i in top_indices]


class SparseMatrixDecoder:
    """稀疏矩阵解码器（适合超大词汇表）

    使用 CSR 矩阵：M[word, slot]
    一次矩阵乘法：word_scores = M @ slot_scores
    """

    def __init__(self, vocab: set, word_to_slots: dict, K: int):
        self.word_list = sorted(vocab)
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}
        V = len(self.word_list)

        # 构建稀疏矩阵 M[word, slot]
        rows, cols, data = [], [], []

        for word, slots in word_to_slots.items():
            if word not in self.word_to_idx:
                continue
            word_idx = self.word_to_idx[word]
            weight = 1.0 / len(slots)

            for slot in slots:
                rows.append(word_idx)
                cols.append(int(slot))
                data.append(weight)

        self.M = csr_matrix((data, (rows, cols)), shape=(V, K), dtype=np.float32)

    def decode(self, slot_scores: np.ndarray) -> dict:
        """一次矩阵乘法解码"""
        word_scores = self.M @ slot_scores  # (V,)
        return {self.word_list[i]: float(word_scores[i])
                for i in range(len(self.word_list))}

    def decode_top_k(self, slot_scores: np.ndarray, k: int = 10) -> list:
        """Top-K 解码"""
        word_scores = self.M @ slot_scores

        if k >= len(word_scores):
            top_indices = np.argsort(word_scores)[::-1]
        else:
            top_indices = np.argpartition(word_scores, -k)[-k:]
            top_indices = top_indices[np.argsort(word_scores[top_indices])[::-1]]

        return [(self.word_list[i], float(word_scores[i]))
                for i in top_indices]


# 向后兼容的函数接口
def decode_words_fast(slot_scores: np.ndarray, decoder) -> dict:
    """使用 FastDecoder 或 SparseMatrixDecoder"""
    return decoder.decode(slot_scores)


def top_k_words_fast(slot_scores: np.ndarray, decoder, k: int = 10) -> list:
    """使用优化的 Top-K 解码"""
    return decoder.decode_top_k(slot_scores, k)
