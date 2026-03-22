"""MathBrain 配置 — 仅保留 A 通道必要参数"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class MathBrainConfig:
    # Hash Retina
    K: int = 65536                           # slot 总数
    NGRAM_SCALES: Tuple[int, ...] = (3,)     # n-gram 尺度
    RETINA_MODE: str = 'hash'                # 'hash' | 'identity'

    # EMA
    N: int = 4                               # EMA 尺度数
    RHO: Tuple[float, ...] = (0.3, 0.6, 0.85, 0.95)
    EPS_Q: float = 1e-6                      # slot 存活阈值
    THETA_Q: float = 0.005                   # position 活跃阈值

    # CosineChaos φ 编码器
    D_PHI: int = 8                           # φ 输出维度
    PHI_SIGMA: float = 0.5                   # Butterfly P 初始化缩放
    CHAOS_N_FOLDS: int = -1                  # 折叠次数 (-1 = D_PHI)
    CHAOS_ALPHA: float = 2.1                 # 混沌参数 α

    # Bilinear 分类器
    CP_RANK: int = 384                       # 低秩分解秩

    # MLP φ (替代 CosineChaos)
    MLP_PHI_HIDDEN: int = 0                  # 0=用 chaos, >0=MLP 隐藏层维度

    # MLP ctx (聚合后非线性)
    MLP_CTX_HIDDEN: int = 0                  # 0=关, >0=ctx MLP 隐藏层维度

    # Fourier φ (替代 CosineChaos)
    FOURIER_PHI_K: int = 0                   # 0=关, >0=Fourier PE 频率数 K

    # Slot Transformer decoder
    TRANSFORMER_D_MODEL: int = 0             # 0=用原始bilinear, >0=Transformer
    TRANSFORMER_NHEAD: int = 4
    TRANSFORMER_LAYERS: int = 2
    TRANSFORMER_FFN: int = 256
    TRANSFORMER_DROPOUT: float = 0.1
    TRANSFORMER_PE_MODE: str = 'fourier'     # 'fourier' | 'linear'
    TRANSFORMER_Q_TRANSFORM: str = 'none'    # 'none' | 'log' | 'norm'
    TRANSFORMER_TIE_WEIGHTS: bool = False     # True = E_src as lm_head (weight tying)
    TRANSFORMER_PRED_MODE: str = 'ce'         # 'ce' | 'innovation' | 'hybrid'
    INNOVATION_WEIGHT: float = 0.1            # λ for hybrid mode: CE + λ·MSE
    VICREG_WEIGHT: float = 0.0               # 0=off, >0=anti-collapse regularization

    # Training
    SLEEP_LR: float = 0.01
    SLEEP_MAX_EPOCHS: int = 1000
    SLEEP_BATCH_SIZE: int = 512

    def __post_init__(self):
        self._rho = np.array(self.RHO, dtype=np.float64)
        self._inverse_weight = (
            (1 - self._rho) / np.mean(1 - self._rho)
        ).astype(np.float32)

    @property
    def rho(self) -> np.ndarray:
        return self._rho.astype(np.float32)

    @property
    def inverse_weight(self) -> np.ndarray:
        return self._inverse_weight
