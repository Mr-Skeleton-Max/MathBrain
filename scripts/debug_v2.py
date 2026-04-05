"""Debug Split-K forward by comparing against V1 output on the same input."""
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mathbrain.triton_flash_ema import (
    flash_ema_forward, flash_ema_forward_splitk,
    precompute_boundaries, _build_unique_index,
)

torch.manual_seed(42)
B, L, V, d, H, N = 2, 64, 1024, 128, 4, 16
hd = d // H

# Create test data
x = torch.randint(0, V, (B, L), device='cuda')
Q = torch.randn(B, H, L, hd, device='cuda')
unique_tensor, K_max = _build_unique_index(x, 'cuda')
K_max = max(K_max, 16)

E_k = torch.randn(B, K_max, H * hd, device='cuda') * 0.1
E_v = torch.randn(B, K_max, H * hd, device='cuda') * 0.1
W_pe = torch.randn(N, d, device='cuda') * 0.1
rhos = torch.exp(-torch.log(torch.tensor(2.0)) / torch.logspace(0, 2, N)).cuda()

C_bounds = precompute_boundaries(x, unique_tensor, rhos, 64, c_init=None)

# ── V1: Original fused kernel ──
print("=== V1 (original fused kernel) ===")
O_v1, LSE_v1, _, _, _ = flash_ema_forward(
    Q, x, E_k, E_v, W_pe, rhos, V, unique_tensor, K_max, C_bounds, use_silu=True
)
print(f"  O_v1 shape: {O_v1.shape}, mean: {O_v1.mean():.6f}, std: {O_v1.std():.6f}")
print(f"  O_v1 abs max: {O_v1.abs().max():.6f}")
print(f"  LSE_v1 mean: {LSE_v1.mean():.4f}")

# ── Split-K: k-chunks parallel ──
print("\n=== Split-K (k-chunks parallel) ===")
O_sk, LSE_sk, _, _, _ = flash_ema_forward_splitk(
    Q, x, E_k, E_v, W_pe, rhos, V, unique_tensor, K_max, C_bounds, use_silu=True
)
print(f"  O_sk shape: {O_sk.shape}, mean: {O_sk.mean():.6f}, std: {O_sk.std():.6f}")
print(f"  O_sk abs max: {O_sk.abs().max():.6f}")
print(f"  LSE_sk mean: {LSE_sk.mean():.4f}")

# ── Compare ──
print("\n=== Comparison ===")
diff_O = (O_v1 - O_sk).abs()
diff_LSE = (LSE_v1 - LSE_sk).abs()
print(f"  O diff:   max={diff_O.max():.8f}, mean={diff_O.mean():.8f}")
print(f"  LSE diff: max={diff_LSE.max():.8f}, mean={diff_LSE.mean():.8f}")

if diff_O.max() < 1e-3:
    print("\n  PASS: V1 and Split-K outputs match within tolerance")
else:
    print(f"\n  FAIL: V1 and Split-K differ significantly")
    worst = diff_O.argmax()
    idx = []
    for s in O_v1.shape[::-1]:
        idx.append(worst % s)
        worst = worst // s
    idx = tuple(reversed(idx))
    print(f"  Worst at {idx}: V1={O_v1[idx]:.6f}, SK={O_sk[idx]:.6f}")
