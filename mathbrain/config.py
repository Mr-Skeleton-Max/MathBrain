"""MathBrain 超参数集中管理"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MathBrainConfig:
    # Hash
    K: int = 65536
    NGRAM_SIZE: int = 3
    NGRAM_SCALES: tuple = (3,)   # 多尺度 n-gram: (1,3,5) 或 (3,) 保持向后兼容

    # EMA
    N: int = 4
    RHO: tuple = (0.3, 0.6, 0.85, 0.95)

    # φ 编码: 支持多种模式
    PHI_MODE: str = "chaos"      # 仅使用 chaos 编码
    D_PHI: int = 8               # φ 维度（random/chaos 模式使用，默认 8）
    PHI_SIGMA: float = 0.5       # E_proj 初始化标准差（random/chaos 模式）

    # Cosine Chaos 编码器参数 (chaos 模式)
    CHAOS_N_FOLDS: int = 3       # 折叠次数 L (推荐 1-3)
    CHAOS_ALPHA: float = 2.0     # 混沌参数 α (2.0 为临界点)
    CHAOS_NO_P: bool = False     # True: 去掉 P 投影，D=N, L=N, α=α*(N)

    # 确定性傅里叶特征 (Deterministic Fourier Features)
    # D_COSINE = d = F_ALPHA * N * 2 (cos + sin)
    F_ALPHA: int = 32            # 频率基值个数
    ALPHA_MIN: float = 0.5       # α 最小频率
    ALPHA_MAX: float = 200.0     # α 最大频率

    # B channel
    B_MODE: str = "energy"    # "hebbian" | "lms" | "delta" | "energy"
    ETA: float = 0.5
    LAMBDA_B: float = 1e-4
    LAMBDA_LMS: float = 0.2
    EPS_PRUNE: float = 1e-4
    MAX_B_ENTRIES: int = 500000
    NEG_SUPPRESS: float = 0.1
    NEG_THRESHOLD: float = 0.1
    B_ENERGY_CONSERVATION: bool = False

    # B 堆栈式裁剪（Wake阶段动态控制）
    B_STACK_CAPACITY: int = 500000
    ENABLE_STACK_PRUNE: bool = False

    # Q state
    EPS_Q: float = 1e-6
    THETA_Q: float = 0.005

    # A / Sleep
    # D_COSINE = phi 维度 (F_ALPHA * N * 2), d = D_COSINE
    # CP_RANK = A 模块内部秩 (与 D 解耦)
    d: int = 256                 # 会被 _build_derived 覆盖
    CP_RANK: int = 384           # slice-wise 分解秩（提升至 384）
    LAMBDA_WD: float = 1e-4      # AdamW weight decay (替代正交约束)
    SLEEP_GLOBAL_DECAY: float = 0.85  # Sleep 前对全部 A 边强度做全局衰减；0 表示关闭衰减，(0,1] 表示保留比例
    GAMMA_DECAY: float = 0.9
    GAMMA_ADAPTIVE: bool = True
    GAMMA_INITIAL: float = 0.99
    GAMMA_MIN: float = 0.9
    GAMMA_DECAY_RATE: float = 0.9
    LAMBDA_ORTHO: float = 0.3
    CONF_THRESHOLD: float = 0.01
    SLEEP_LR: float = 0.01
    SLEEP_MAX_EPOCHS: int = 1000
    SLEEP_MIN_EPOCHS: int = 500
    SLEEP_PATIENCE: int = 200
    SLEEP_REL_TOL: float = 0.001
    SLEEP_BATCH_SIZE: int = 5000  # Mini-batch Sleep 批量大小（稀疏抽取）
    SLEEP_MINIBATCH_MEMORY_LIMIT_GB: float = 20.0
    SLEEP_MINIBATCH_UTILIZATION: float = 0.9
    SLEEP_LOSS: str = "mse"          # "mse" 或 "huber"
    SLEEP_HUBER_DELTA: float = 1.0    # Huber loss 的 δ 阈值
    SLEEP_SOLVER: str = "adamw"       # "adamw" | "als" | "dream"
    SLEEP_ALS_ITERS: int = 6
    SLEEP_ALS_RIDGE: float = 1e-3
    SLEEP_ALS_PROX: float = 1.0
    SLEEP_DREAM_SAMPLES: int = 512
    SLEEP_DREAM_ACTIVE: int = 8
    SLEEP_DREAM_TOPK: int = 0  # 0 表示使用全部 involved slots 做完整分布蒸馏
    SLEEP_DREAM_PHASES: int = 8
    SLEEP_DREAM_MAX_PROTO: int = 4
    SLEEP_DREAM_BATCH_SIZE: int = 128
    SLEEP_DREAM_EPOCHS: int = 200
    SLEEP_DREAM_TEMPERATURE: float = 1.5
    SLEEP_DREAM_PROX: float = 1e-3
    SLEEP_DREAM_LOGIT_WEIGHT: float = 1.0
    SLEEP_DREAM_KL_WEIGHT: float = 0.25
    SLEEP_DREAM_EPISODE_LEN: int = 12
    SLEEP_DREAM_SEQ_MULTIPLIER: float = 1.0
    SLEEP_DREAM_PROBE_COUNT: int = 3
    SLEEP_DREAM_UNIFORM_MIX: float = 0.25
    SLEEP_DREAM_SEQUENCE_SOURCE: str = "train"
    SLEEP_DREAM_WARMSTART_ALS_ITERS: int = 0
    SLEEP_EMA_LAMBDA: float = 0.1
    SLEEP_EMA_ADAPTIVE: bool = False
    SLEEP_EMA_LAMBDA_MIN: float = 0.05
    SLEEP_EMA_LAMBDA_MAX: float = 0.3
    SLEEP_USE_SATURATION_GATE: bool = False
    SLEEP_ETA: float = 0.1

    # Decay table size
    DECAY_TABLE_SIZE: int = 10000

    def __post_init__(self):
        self._rho = np.array(self.RHO, dtype=np.float32)
        self._build_derived()

    def _build_derived(self):
        """Compute derived constants from base params."""
        # 反向缩放权重: 快尺度大α, 慢尺度小α
        self.INVERSE_WEIGHT = (
            (1 - self._rho) / np.mean(1 - self._rho)
        ).astype(np.float32)  # (N,)

        # 傅立叶特征维度
        self.D_COSINE = self.F_ALPHA * self.N * 2

        # 根据 PHI_MODE 设置 d
        if self.PHI_MODE == "fourier":
            self.d = self.D_COSINE
        elif self.PHI_MODE in ("random", "chaos"):
            if self.CHAOS_NO_P:
                self.D_PHI = self.N  # 无 P 模式: D = N
            self.d = self.D_PHI
        else:
            raise ValueError(f"Unknown PHI_MODE: {self.PHI_MODE}")

        # 频率网格 (几何间距, 对数空间) - 用于傅立叶模式
        self.ALPHA_GRID = np.geomspace(
            self.ALPHA_MIN, self.ALPHA_MAX, self.F_ALPHA
        ).astype(np.float32)  # (F,)

        # Decay table for lazy B decay
        self.decay_table = np.array(
            [(1 - self.LAMBDA_B) ** i for i in range(self.DECAY_TABLE_SIZE)],
            dtype=np.float32
        )

        # Inference LayerNorm gain
        self.GAMMA_LN = 3 * np.log(self.K)

    @property
    def rho(self) -> np.ndarray:
        return self._rho
