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


class IdentityRetina:
    """Identity Retina: 1 word = 1 slot.

    Each unique word maps to a unique slot ID.
    No hash collisions, no n-gram decomposition.
    """

    def __init__(self, config: MathBrainConfig):
        self.k = config.K
        self._word_to_slot: Dict[str, int] = {}
        self._next_slot = 0

    def encode(self, word: str) -> Dict[int, int]:
        """word → {slot_id: 1}"""
        if word not in self._word_to_slot:
            self._word_to_slot[word] = self._next_slot
            self._next_slot += 1
        return {self._word_to_slot[word]: 1}

    def get_slots(self, word: str) -> np.ndarray:
        """word → single-element slot array"""
        enc = self.encode(word)
        return np.array(sorted(enc.keys()), dtype=np.int32)


class BPERetina:
    """BPE Retina: 1 BPE token = 1 slot.

    Uses the global BPE tokenizer from data module.
    Each BPE token ID maps directly to a slot ID.
    """

    def __init__(self, config: MathBrainConfig):
        self.k = config.K
        self._cache: Dict[str, Dict[int, int]] = {}

    def encode(self, word: str) -> Dict[int, int]:
        """word → {slot_id: 1}  (BPE token ID = slot ID)"""
        if word in self._cache:
            return self._cache[word]
        # For BPE, each "word" is already a single BPE token string
        # The slot ID is the token's position in the vocabulary
        # We use a simple hash to get a consistent slot assignment
        slot_id = hash(word) % (2**31)
        result = {slot_id: 1}
        self._cache[word] = result
        return result

    def get_slots(self, word: str) -> np.ndarray:
        """word → single-element slot array"""
        enc = self.encode(word)
        return np.array(sorted(enc.keys()), dtype=np.int32)
