"""HashRetinaV2: n-gram → 槽位映射 (支持多尺度)"""

import hashlib
from collections import defaultdict
import numpy as np

from .config import MathBrainConfig


def _ngram_hash(ngram: str, k: int) -> int:
    h = hashlib.md5(ngram.encode()).hexdigest()
    return int(h[:8], 16) % k


class HashRetinaV2:
    """多尺度 N-gram 频次哈希视网膜

    支持多种 n-gram 尺度同时激活 (如 1,3,5)。
    不同尺度通过 hash 前缀 "{n}:" 隔离到不同的 slot 区域。
    """

    def __init__(self, config: MathBrainConfig = None):
        cfg = config or MathBrainConfig()
        self.k = cfg.K
        self.ngram_size = cfg.NGRAM_SIZE        # 用于 analyze/compare 等诊断
        self.scales = cfg.NGRAM_SCALES           # 多尺度列表
        self._cache_counts: dict = {}
        self._cache_slots: dict = {}

    def encode(self, word: str) -> dict:
        """词 → Dict[slot_id, count] (多尺度 n-gram 频次)"""
        if word in self._cache_counts:
            return self._cache_counts[word]

        padded = f"^{word.lower()}$"
        slot_counts: dict = defaultdict(int)

        for n in self.scales:
            if len(padded) < n:
                # 短于 n-gram 的词: 整体作为单个 slot
                slot_counts[_ngram_hash(f"{n}:{padded}", self.k)] += 1
                continue
            ngram_counts: dict = defaultdict(int)
            for i in range(len(padded) - n + 1):
                ngram_counts[padded[i:i + n]] += 1
            for ngram, count in ngram_counts.items():
                # 用 "{n}:" 前缀隔离不同尺度的 hash 空间
                slot_counts[_ngram_hash(f"{n}:{ngram}", self.k)] += count

        result = dict(slot_counts)
        self._cache_counts[word] = result
        return result

    def get_slots(self, word: str) -> np.ndarray:
        """词 → slot_id 数组 (去重)"""
        if word in self._cache_slots:
            return self._cache_slots[word]
        slots = np.array(list(self.encode(word).keys()), dtype=np.int32)
        self._cache_slots[word] = slots
        return slots

    def jaccard(self, w1: str, w2: str) -> float:
        s1 = set(self.encode(w1).keys())
        s2 = set(self.encode(w2).keys())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    # ---- 诊断 API ----

    def analyze(self, word: str) -> dict:
        """展示词的多尺度 n-gram 分解、槽位映射"""
        padded = f"^{word.lower()}$"
        by_scale = {}
        for n in self.scales:
            if len(padded) < n:
                by_scale[n] = [padded]  # 整体作为单个 token
                continue
            ngs = [padded[i:i + n] for i in range(len(padded) - n + 1)]
            by_scale[n] = ngs
        slot_counts = self.encode(word)
        return {
            'word': word,
            'padded': padded,
            'ngrams_by_scale': by_scale,
            'slot_counts': slot_counts,
            'n_slots': len(slot_counts),
        }

    def compare(self, w1: str, w2: str) -> dict:
        """比较两词的槽位重叠"""
        s1 = set(self.encode(w1).keys())
        s2 = set(self.encode(w2).keys())
        shared = s1 & s2
        return {
            'jaccard': self.jaccard(w1, w2),
            'shared_slots': len(shared),
            'total_slots': len(s1 | s2),
            'w1_only': len(s1 - s2),
            'w2_only': len(s2 - s1),
        }
