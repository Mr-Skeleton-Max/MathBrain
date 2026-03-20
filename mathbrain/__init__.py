"""MathBrain — 稀疏 EMA + CosineChaos 序列预测

GPU-optimized pipeline:
  - HashRetina hashing (CPU parallel)
  - Dynamic EMA per batch (GPU)
  - CUDA warp shuffle phi encoder
  - Triton fused bilinear
  - E_word matmul fusion
"""

from .config import MathBrainConfig
from .retina import HashRetina
from .phi_encoder import CosineChaosEncoder
from .trainer import MathBrainTrainer

__all__ = [
    'MathBrainConfig',
    'HashRetina',
    'CosineChaosEncoder',
    'MathBrainTrainer',
]
