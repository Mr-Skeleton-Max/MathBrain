#!/usr/bin/env python3
"""精确拆分 phi_encode 每一步的耗时 — CUDA warp shuffle 版"""
import sys, time
import torch
sys.path.insert(0, '/root/autodl-tmp')

from MathBrain.config import MathBrainConfig
from MathBrain.gpu_phi_encoder import GPUPhiEncoder

cfg = MathBrainConfig(
    N=8, RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
    D_PHI=32, CP_RANK=32)

print("Compiling CUDA kernel...")
phi = GPUPhiEncoder(cfg, device='cuda')
print("Done.\n")

for n in [10000, 100000, 800000]:
    Q = torch.randn(n, 8, device='cuda')

    # Warmup
    for _ in range(5):
        phi.encode_normalized(Q)
    torch.cuda.synchronize()

    N_RUNS = 20
    elapsed = 0.0

    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = phi.encode_normalized(Q)
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0

    ms = elapsed / N_RUNS * 1000
    print(f"n={n:>8,}:  {ms:.3f}ms  (output shape: {tuple(out.shape)})")

print("\nDone.")
