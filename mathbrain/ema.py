"""
Per-Slot EMA Scanner.

Computes rhos (decay rates) only. C_init is computed vectorially in model.py
using the same decay matrix as attention — zero Python loops.
"""

import torch
import torch.nn as nn


class EMAScanner(nn.Module):
    """
    EMA decay rate generator. The actual C_init computation is vectorized
    in model.py using decay matrix + same_token mask — no sequential loop.
    """
    def __init__(self, N: int, base: float = 2.0, max_scale: float = 10.0):
        super().__init__()
        self.N = N
        log_base = torch.log(torch.tensor(base))
        scales = torch.logspace(0, torch.log10(torch.tensor(max_scale)), N)
        rhos = torch.exp(-log_base / scales)
        self.register_buffer("rhos", rhos)
