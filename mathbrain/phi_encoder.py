"""φ 编码器。

当前实验路径仅保留 chaos / fourier，其中 dream sleep 实验只使用 chaos。
"""

import numpy as np

from .config import MathBrainConfig


class PhiEncoderRandom:
    """随机高斯投影 + Cosine 混沌编码器

    φ = cos(E_proj · (α ⊙ Q))

    其中:
      E_proj: 随机高斯投影矩阵 (D, N)，N(0, σ²)，固定种子 42
      α: 反向缩放参数 (1-ρ)/mean(1-ρ)，快尺度大、慢尺度小
      Q: EMA 状态 (n_active, N)
    输出维度: D (可配置，默认 8)
    """

    def __init__(self, config: MathBrainConfig = None):
        cfg = config or MathBrainConfig()
        self.D = cfg.D_PHI  # 投影维度
        self.N = cfg.N      # EMA 尺度数

        # 固定种子生成随机高斯投影矩阵 N(0, σ²)
        sigma = cfg.PHI_SIGMA
        rng = np.random.RandomState(42)
        self.E_proj = rng.randn(self.D, self.N).astype(np.float32) * sigma

        # 反向缩放参数: α_k = (1-ρ_k) / mean(1-ρ) × alpha_base
        # alpha_base 是频率基值，确保 phi 的频率范围合适
        alpha_base = (np.pi / (2 * sigma * np.sqrt(self.D))) * 63.5
        self.alpha = (cfg.INVERSE_WEIGHT * alpha_base).astype(np.float32)  # (N,)

    def encode(self, Q_values: np.ndarray) -> np.ndarray:
        """Q_values (n, N) → φ (n, D)

        1. scaled_Q = α ⊙ Q              — 反向缩放 (n, N)
        2. proj = E_proj @ scaled_Q^T    — 投影 (D, n)
        3. φ = cos(proj^T)               — Cosine 折叠 (n, D)
        """
        # (n, N) ⊙ (N,) → (n, N)
        scaled_Q = Q_values * self.alpha

        # (D, N) @ (N, n) → (D, n) → (n, D)
        proj = self.E_proj @ scaled_Q.T
        phi = np.cos(proj.T)

        return phi.astype(np.float32)


class PhiEncoderFourier:
    """确定性傅里叶特征编码器

    φ = [cos(α_1·w⊙Q), sin(α_1·w⊙Q), ..., cos(α_F·w⊙Q), sin(α_F·w⊙Q)]

    其中:
      α_f: F 个几何间距频率基值
      w = (1-ρ)/mean(1-ρ): 反向缩放权重 (N,)
      Q: EMA 状态 (n_active, N)
    输出维度: D = F × N × 2
    """

    def __init__(self, config: MathBrainConfig = None):
        cfg = config or MathBrainConfig()
        self.D = cfg.D_COSINE               # F * N * 2
        self._alpha_grid = cfg.ALPHA_GRID    # (F,)
        self._inv_weight = cfg.INVERSE_WEIGHT  # (N,)

    def encode(self, Q_values: np.ndarray) -> np.ndarray:
        """Q_values (n, N) → φ (n, F*N*2)

        1. scaled_Q = w * Q                     — 反向缩放 (n, N)
        2. theta = α_f * scaled_Q for each f    — 多频率 (n, F*N)
        3. φ = [cos(theta), sin(theta)]          — 拼接 (n, F*N*2)
        """
        # (n, N)
        scaled_Q = self._inv_weight * Q_values

        # (F, 1, 1) * (1, n, N) → (F, n, N)
        theta = self._alpha_grid[:, None, None] * scaled_Q[None, :, :]

        # (F, n, N) → (n, F*N)
        n_active = Q_values.shape[0]
        theta_flat = theta.transpose(1, 0, 2).reshape(n_active, -1)

        # cos + sin 拼接 → (n, F*N*2)
        phi = np.concatenate([np.cos(theta_flat), np.sin(theta_flat)], axis=-1)

        return phi.astype(np.float32)


# 默认别名保留到 chaos 工厂路径；random 模式已移除实验支持
PhiEncoder = PhiEncoderRandom


def create_phi_encoder(config: MathBrainConfig = None):
    """根据配置创建 φ 编码器

    Args:
        config: 配置对象

    Returns:
        PhiEncoder 实例
    """
    cfg = config or MathBrainConfig()

    if cfg.PHI_MODE == "fourier":
        return PhiEncoderFourier(cfg)
    elif cfg.PHI_MODE == "random":
        raise ValueError('PHI_MODE=random has been removed from the experiment path; use chaos')
    elif cfg.PHI_MODE == "chaos":
        from .phi_encoder_chaos import PhiEncoderCosineChaos
        return PhiEncoderCosineChaos(
            cfg, D=cfg.D_PHI,
            n_folds=cfg.CHAOS_N_FOLDS,
            alpha=cfg.CHAOS_ALPHA)
    else:
        raise ValueError(f"Unknown PHI_MODE: {cfg.PHI_MODE}")
