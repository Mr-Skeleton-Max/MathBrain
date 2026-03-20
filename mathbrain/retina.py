"""HashRetina: n-gram → slot 映射

全向量化实现 — encode 返回 (slot_ids, counts) 稀疏表示。
缓存已见词的编码结果。
"""

import hashlib
from typing import Dict, Tuple

import numpy as np

from .config import MathBrainConfig


def _ngram_hash(ngram: str, k: int) -> int:
    h = hashlib.md5(ngram.encode()).hexdigest()
    return int(h[:8], 16) % k


class HashRetina:
    """多尺度 N-gram 哈希视网膜

    word → Dict[slot_id, count]
    不同 n-gram 尺度通过 hash 前缀隔离。
    """

    def __init__(self, config: MathBrainConfig):
        self.k = config.K
        self.scales = config.NGRAM_SCALES
        self._cache: Dict[str, Dict[int, int]] = {}

    def encode(self, word: str) -> Dict[int, int]:
        """word → {slot_id: count}"""
        if word in self._cache:
            return self._cache[word]

        padded = f"^{word.lower()}$"
        slot_counts: Dict[int, int] = {}

        for n in self.scales:
            if len(padded) < n:
                sid = _ngram_hash(f"{n}:{padded}", self.k)
                slot_counts[sid] = slot_counts.get(sid, 0) + 1
                continue
            ngram_counts: Dict[str, int] = {}
            for i in range(len(padded) - n + 1):
                ng = padded[i:i + n]
                ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
            for ngram, count in ngram_counts.items():
                sid = _ngram_hash(f"{n}:{ngram}", self.k)
                slot_counts[sid] = slot_counts.get(sid, 0) + count

        self._cache[word] = slot_counts
        return slot_counts

    def get_slots(self, word: str) -> np.ndarray:
        """word → sorted unique slot IDs"""
        return np.array(sorted(self.encode(word).keys()), dtype=np.int32)
