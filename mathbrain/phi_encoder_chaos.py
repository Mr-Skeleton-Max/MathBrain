"""Cosine Chaos φ 编码器

基于 Edge of Chaos 理论的递归位置编码:
  M_t = cos(α · (W_shift · M_{t-1} + E))

其中:
  - E = P @ (α_scale ⊙ Q): 初始投影
  - W_shift: 循环位移矩阵
  - α: 混沌参数 (α=2 为临界点)
  - L: 折叠次数

理论依据:
  Lyapunov 指数 λ = log(α/2)
  α < 2: 有序区 (λ<0)
  α = 2: 临界点 (λ=0, 最大计算能力)
  α > 2: 混沌区 (λ>0)
"""

import numpy as np

from .config import MathBrainConfig

try:
    import mlx.core as mx
    HAS_MLX = True
except Exception:
    HAS_MLX = False
    mx = None


class PhiEncoderCosineChaos:
    """Cosine Chaos 编码器

    M_0 = 0
    M_t = cos(α · (W_shift · M_{t-1} + E))  for t=1..L
    φ = M_L

    参数:
      D: 输出维度
      n_folds: 折叠次数 L
      alpha: 混沌参数 α (推荐 2.0 为临界点)
    """

    def __init__(self, config: MathBrainConfig = None, D=8, n_folds=3, alpha=2.0):
        cfg = config or MathBrainConfig()
        self.D = D
        self.N = cfg.N
        self.n_folds = n_folds
        self.alpha = alpha

        sigma = cfg.PHI_SIGMA
        rng = np.random.RandomState(42)
        self.P = rng.randn(D, self.N).astype(np.float32) * sigma

        alpha_base = (np.pi / (2 * sigma * np.sqrt(D))) * 63.5
        self.alpha_scale = (cfg.INVERSE_WEIGHT * alpha_base).astype(np.float32)

        if HAS_MLX:
            self._P_mx = mx.array(self.P)
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
        e_mx = self._P_mx @ scaled_q.T
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
        scaled_Q = Q_values * self.alpha_scale
        E = (self.P @ scaled_Q.T).astype(np.float32, copy=False)

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
        e_mx = self._P_mx @ scaled_q.T

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


def create_phi_encoder_chaos(config: MathBrainConfig = None, D=8, n_folds=3, alpha=2.0):
    return PhiEncoderCosineChaos(config, D=D, n_folds=n_folds, alpha=alpha)
