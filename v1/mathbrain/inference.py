"""推理合并: logits 融合 + 解码"""

import numpy as np

from .config import MathBrainConfig


def merge_logits(scores_B: np.ndarray, scores_A: np.ndarray,
                 multiplicative: bool = False) -> np.ndarray:
    """B/A 分数合并

    加法模式 (A 无知识): l = l_B + l_A
    乘法模式 (A 有知识): l = l_A × (1 + l_B)
      B 在 A 的基础上做重分配，不创造新能量
    """
    if multiplicative:
        return scores_A * (1.0 + scores_B)
    return scores_B + scores_A


def layernorm_softmax(logits: np.ndarray, gamma_ln: float) -> np.ndarray:
    """LayerNorm → Softmax

    LayerNorm: x̂ = γ * (x - μ) / (σ + ε)
    Softmax: p = exp(x̂) / Σexp(x̂)
    """
    mu = logits.mean()
    sigma = logits.std() + 1e-8
    normed = gamma_ln * (logits - mu) / sigma
    normed -= normed.max()
    exp_x = np.exp(normed)
    return exp_x / (exp_x.sum() + 1e-8)


def decode_words(slot_scores: np.ndarray, vocab: set,
                 word_to_slots: dict) -> dict:
    """槽位分数 → 词分数 (算术平均)

    Score_w = (1/|S_w|) × Σ y[j]
    """
    word_scores = {}
    for word in vocab:
        slots = word_to_slots.get(word)
        if slots is not None and len(slots) > 0:
            word_scores[word] = float(np.sum(slot_scores[slots]) / len(slots))
        else:
            word_scores[word] = 0.0
    return word_scores


def top_k_words(word_scores: dict, k: int = 10) -> list:
    """Top-K 词排序"""
    return sorted(word_scores.items(), key=lambda x: -x[1])[:k]
