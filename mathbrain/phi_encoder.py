"""CosineChaos φ 编码器

递归位置编码:
  M_0 = 0
  M_t = cos(α · (shift(M_{t-1}) + E))   for t = 1..L
  φ = M_L

E = P @ (α_scale ⊙ Q),  P ∈ R^{D×N} 是 Butterfly 正交矩阵。

全向量化: encode() 接受 (n, N) batch Q_values, 内部无 Python for-loop
(仅 n_folds 次循环, 通常 L=D=8~32, 不可避免)。
"""

import numpy as np

from .config import MathBrainConfig


def _butterfly_matrix(D: int, N: int, sigma: float) -> np.ndarray:
    """确定性 Butterfly 正交混合矩阵 * sigma → (D, N)

    log2(D) 层稀疏 2×2 旋转的乘积。
    角度用黄金比例序列确保无对齐。
    """
    d = D
    if d & (d - 1) != 0:
        # 非 2 的幂 → fallback QR
        rng = np.random.RandomState(42)
        Q, _ = np.linalg.qr(rng.randn(d, d))
        return (Q[:, :N] * sigma).astype(np.float32)

    log_d = int(np.log2(d))
    phi_golden = (1 + np.sqrt(5)) / 2

    result = np.eye(d)
    for layer in range(log_d):
        stride = 1 << layer
        B = np.eye(d)
        for i in range(0, d, 2 * stride):
            for j in range(stride):
                i1, i2 = i + j, i + j + stride
                theta = np.pi * ((layer * d + i1) * phi_golden % 1)
                c, s = np.cos(theta), np.sin(theta)
                B[i1, i1] = c;  B[i1, i2] = -s
                B[i2, i1] = s;  B[i2, i2] = c
        result = B @ result

    return (result[:, :N] * sigma).astype(np.float32)


class CosineChaosEncoder:
    """CosineChaos φ 编码器

    Attributes:
        D: 输出维度
        N: EMA 尺度数
        P: Butterfly 投影矩阵 (D, N)
        alpha: 混沌参数
        alpha_scale: 反向缩放向量 (N,)
        n_folds: 递归折叠次数
    """

    def __init__(self, config: MathBrainConfig):
        self.N = config.N
        self.D = config.D_PHI
        self.n_folds = config.D_PHI if config.CHAOS_N_FOLDS < 0 else config.CHAOS_N_FOLDS
        self.alpha = config.CHAOS_ALPHA

        # Butterfly P 矩阵
        self.P = _butterfly_matrix(self.D, self.N, config.PHI_SIGMA)

        # 反向缩放: alpha_base 让 |E_steady| ≈ π
        alpha_base = (np.pi / (2 * config.PHI_SIGMA * np.sqrt(self.D))) * 63.5
        self.alpha_scale = (config.inverse_weight * alpha_base).astype(np.float32)

    def encode(self, Q_values: np.ndarray) -> np.ndarray:
        """Q_values (n, N) → φ (n, D)

        全向量化: matmul + in-place cos, 无 per-sample 循环。
        仅有 n_folds 次循环 (L=D, 不可避免的递归)。
        """
        scaled_Q = Q_values * self.alpha_scale           # (n, N)
        E = (self.P @ scaled_Q.T).astype(np.float32)     # (D, n)

        n = Q_values.shape[0]
        M = np.zeros((self.D, n), dtype=np.float32)
        buf = np.empty((self.D, n), dtype=np.float32)

        for _ in range(self.n_folds):
            buf[0] = M[-1]
            buf[1:] = M[:-1]
            np.add(buf, E, out=buf)
            buf *= self.alpha
            np.cos(buf, out=M)

        return M.T  # (n, D)

    def encode_normalized(self, Q_values: np.ndarray) -> np.ndarray:
        """Q_values (n, N) → φ_hat (n, D), L2 归一化"""
        if Q_values.shape[0] == 0:
            return np.zeros((0, self.D), dtype=np.float32)
        phi = self.encode(Q_values)
        norms = np.linalg.norm(phi, axis=1, keepdims=True)
        return (phi / (norms + 1e-8)).astype(np.float32)
