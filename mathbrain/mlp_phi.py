"""Learnable φ encoders — alternatives to fixed cosine chaos map

MLPPhiEncoder:  Q → MLP → L2 normalize → φ
FourierPhiEncoder:  Q → sin/cos multi-freq → linear proj → L2 normalize → φ
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPhiEncoder(nn.Module):
    """Learnable MLP φ encoder with L2 normalization."""

    def __init__(self, N_in: int, D_out: int, hidden: int = 64):
        super().__init__()
        self.N_in = N_in
        self.D_out = D_out
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(N_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, D_out),
        )
        # Small init to match chaos output scale
        with torch.no_grad():
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)

    def forward(self, Q_values: torch.Tensor) -> torch.Tensor:
        """Q_values (n, N_in) → φ̂ (n, D_out), L2 normalized."""
        h = self.net(Q_values)
        return F.normalize(h, dim=1, eps=1e-8)

    def encode_normalized(self, Q_values: torch.Tensor) -> torch.Tensor:
        """API compatible with GPUPhiEncoder."""
        return self.forward(Q_values)


class FourierPhiEncoder(nn.Module):
    """Fourier feature φ encoder — Q → sin/cos expansion → project → normalize.

    固定频率 sin/cos 展开 (无可学习参数), 然后线性投影到 D_PHI 维.
    等价于带 sine 激活的单层网络.
    """

    def __init__(self, N_ema: int, D_out: int, K_freq: int = 8):
        super().__init__()
        self.N_ema = N_ema
        self.D_out = D_out
        self.K_freq = K_freq

        # Fixed frequencies: π, 2π, 4π, ..., 2^(K-1)π
        freqs = (2.0 ** torch.arange(K_freq).float()) * math.pi
        self.register_buffer('freqs', freqs)

        fourier_dim = N_ema * 2 * K_freq  # sin + cos per scale per freq
        # Linear projection to D_out (only learnable part)
        self.proj = nn.Linear(fourier_dim, D_out, bias=False)

    def forward(self, Q_values: torch.Tensor) -> torch.Tensor:
        """Q_values (n, N_ema) → φ̂ (n, D_out), L2 normalized."""
        # Fourier expansion: (n, N_ema) → (n, N_ema, K) → (n, N_ema, 2K)
        x = Q_values.unsqueeze(-1) * self.freqs          # (n, N, K)
        fe = torch.cat([x.sin(), x.cos()], dim=-1)       # (n, N, 2K)
        fe = fe.reshape(Q_values.shape[0], -1)            # (n, N*2K)

        # Project to D_out and normalize
        h = self.proj(fe)                                 # (n, D_out)
        return F.normalize(h, dim=1, eps=1e-8)

    def encode_normalized(self, Q_values: torch.Tensor) -> torch.Tensor:
        """API compatible with GPUPhiEncoder."""
        return self.forward(Q_values)

