"""Configuration for MathBrain EMA Slot Transformer"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class MathBrainConfig:
    # ── Tokenizer / Retina ──
    vocab_size: int = 8000
    retina_mode: str = 'bpe'  # 'bpe' or 'identity'

    # ── EMA Encoding ──
    # N is the number of timescales (e.g., N=8)
    N: int = 8
    # Rho values can be configured manually or generated via half-lives
    rho: Tuple[float, ...] = (0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999)
    eps_q: float = 1e-6       # Dead slot threshold (if Q < eps_q, slot is ignored)
    max_active_slots: int = 64 # Hard cap on active slots fed to the decoder
                               # Without this, long-timescale rho≈1 keeps ALL slots
                               # active indefinitely → k grows to vocab_size → OOM.
    

    # ── Decoder (Slot Transformer) ──
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    tie_weights: bool = False # Tie embedding and output projection

    def __post_init__(self):
        # Validate and convert rho to a float32 numpy array for fast access
        self.rho_array = np.array(self.rho, dtype=np.float32)
        if len(self.rho_array) != self.N:
            raise ValueError(f"Length of rho ({len(self.rho_array)}) must match N ({self.N})")

    @classmethod
    def from_half_lives(cls, h_min: float, h_max: float, N: int, **kwargs):
        """Generate Config using log-uniformly spaced half-lives for rho."""
        half_lives = np.logspace(np.log10(h_min), np.log10(h_max), N)
        rho_vals = tuple(float(0.5 ** (1.0 / h)) for h in half_lives)
        return cls(N=N, rho=rho_vals, **kwargs)
