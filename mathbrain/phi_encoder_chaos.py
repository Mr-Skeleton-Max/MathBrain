"""Cosine Chaos φ 编码器

基于 Edge of Chaos 理论的递归位置编码:
  M_t = cos(α · (W_shift · M_{t-1} + E))

两种模式:
  1. 有 P (legacy): E = P @ (α_scale ⊙ Q), P ∈ R^{D×N} 随机高斯
  2. 无 P (CHAOS_NO_P=True): E = (α_scale ⊙ Q).T, D=N, 满秩

无 P 模式下:
  D = N, L = N (完整循环), α = α*(N) = 2 - 0.399/N^0.322
  Lyapunov 指数 λ ≈ 0 (临界点)
"""

import numpy as np

from .config import MathBrainConfig

try:
    import mlx.core as mx
    HAS_MLX = True
except Exception:
    HAS_MLX = False
    mx = None


def critical_alpha(N: int) -> float:
    """计算 L=N 时的 Lyapunov 临界 α*。

    经验公式: α*(N) = 2 - 0.399 / N^0.322
    数值验证误差 < 0.015 (N=4..32)
    """
    return 2.0 - 0.399 / (N ** 0.322)


class PhiEncoderCosineChaos:
    """Cosine Chaos 编码器

    M_0 = 0
    M_t = cos(α · (W_shift · M_{t-1} + E))  for t=1..L
    φ = M_L

    参数:
      D: 输出维度
      n_folds: 折叠次数 L
      alpha: 混沌参数 α
      no_P: 去掉 P 投影矩阵，D=N
    """

    def __init__(self, config: MathBrainConfig = None, D=8, n_folds=3,
                 alpha=2.0, no_P=False):
        cfg = config or MathBrainConfig()
        self.N = cfg.N
        self.no_P = no_P

        if no_P:
            # 无 P 模式: D=N, L=N, α=α*(N)
            self.D = self.N
            self.n_folds = self.N
            self.alpha = critical_alpha(self.N)
            self.P = None

            # alpha_scale: 保留 INVERSE_WEIGHT 均衡各尺度贡献
            # alpha_base 让 |E_steady| ≈ π (cos 有效灵敏区)
            mean_one_minus_rho = float(np.mean(1 - cfg.rho))
            alpha_base = np.pi * mean_one_minus_rho
            self.alpha_scale = (cfg.INVERSE_WEIGHT * alpha_base).astype(np.float32)
        else:
            # 有 P 模式
            self.D = D
            self.n_folds = D if n_folds < 0 else n_folds  # -1 → L=D (完整平移一圈)
            self.alpha = alpha
            sigma = cfg.PHI_SIGMA
            rng = np.random.RandomState(42)
            self.P = rng.randn(D, self.N).astype(np.float32) * sigma
            alpha_base = (np.pi / (2 * sigma * np.sqrt(D))) * 63.5
            self.alpha_scale = (cfg.INVERSE_WEIGHT * alpha_base).astype(np.float32)

        if HAS_MLX:
            if self.P is not None:
                self._P_mx = mx.array(self.P)
            else:
                self._P_mx = None
            self._alpha_scale_mx = mx.array(self.alpha_scale)
            self._mlx_warm = False
        else:
            self._P_mx = None
            self._alpha_scale_mx = None
            self._mlx_warm = False

    def _warmup_mlx(self):
        if not HAS_MLX or self._mlx_warm:
            return
        q_mx = mx.array(np.ones((2, self.N), dtype=np.float32))
        scaled_q = q_mx * self._alpha_scale_mx
        if self._P_mx is not None:
            e_mx = self._P_mx @ scaled_q.T
        else:
            e_mx = scaled_q.T
        m_mx = mx.zeros((self.D, 2), dtype=mx.float32)
        for _ in range(self.n_folds):
            shifted = mx.concatenate([m_mx[-1:], m_mx[:-1]], axis=0)
            m_mx = mx.cos(self.alpha * (shifted + e_mx))
        phi_mx = m_mx.T
        norms = mx.sqrt(mx.sum(phi_mx * phi_mx, axis=1, keepdims=True))
        phi_hat = phi_mx / (norms + 1e-8)
        mx.eval(phi_hat)
        self._mlx_warm = True

    def encode(self, Q_values: np.ndarray) -> np.ndarray:
        """Q_values (n, N) → φ (n, D)"""
        scaled_Q = Q_values * self.alpha_scale  # (n, N)

        if self.P is not None:
            E = (self.P @ scaled_Q.T).astype(np.float32, copy=False)  # (D, n)
        else:
            E = scaled_Q.T.astype(np.float32, copy=False)  # (N, n), D=N

        n = Q_values.shape[0]
        M = np.zeros((self.D, n), dtype=np.float32)
        buf = np.empty((self.D, n), dtype=np.float32)

        for _ in range(self.n_folds):
            buf[0] = M[-1]
            buf[1:] = M[:-1]
            np.add(buf, E, out=buf)
            buf *= self.alpha
            np.cos(buf, out=M)

        return M.T

    def encode_normalized(self, Q_values: np.ndarray) -> np.ndarray:
        if Q_values.shape[0] == 0:
            return np.zeros((0, self.D), dtype=np.float32)
        phi = self.encode(Q_values)
        norms = np.linalg.norm(phi, axis=1, keepdims=True)
        return (phi / (norms + 1e-8)).astype(np.float32, copy=False)

    def encode_normalized_mlx(self, Q_values: np.ndarray) -> np.ndarray:
        if not HAS_MLX or Q_values.shape[0] == 0:
            return self.encode_normalized(Q_values)

        self._warmup_mlx()
        q_mx = mx.array(Q_values.astype(np.float32, copy=False))
        scaled_q = q_mx * self._alpha_scale_mx

        if self._P_mx is not None:
            e_mx = self._P_mx @ scaled_q.T
        else:
            e_mx = scaled_q.T

        n = int(Q_values.shape[0])
        m_mx = mx.zeros((self.D, n), dtype=mx.float32)
        for _ in range(self.n_folds):
            shifted = mx.concatenate([m_mx[-1:], m_mx[:-1]], axis=0)
            m_mx = mx.cos(self.alpha * (shifted + e_mx))

        phi_mx = m_mx.T
        norms = mx.sqrt(mx.sum(phi_mx * phi_mx, axis=1, keepdims=True))
        phi_hat = phi_mx / (norms + 1e-8)
        mx.eval(phi_hat)
        return np.array(phi_hat, copy=False)


def create_phi_encoder_chaos(config: MathBrainConfig = None, D=8, n_folds=3,
                             alpha=2.0):
    cfg = config or MathBrainConfig()
    if cfg.CHAOS_NO_P:
        return PhiEncoderCosineChaos(cfg, no_P=True)
    return PhiEncoderCosineChaos(cfg, D=D, n_folds=n_folds, alpha=alpha)
