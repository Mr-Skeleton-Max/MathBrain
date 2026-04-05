"""Debug V2 forward by comparing against V1 output on the same input."""
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mathbrain.triton_flash_ema import (
    flash_ema_forward, flash_ema_forward_v2, compute_c_all,
    precompute_boundaries, _build_unique_index, _pad_n_dim,
    HAS_TRITON
)

torch.manual_seed(42)
B, L, V, d, H, N = 2, 64, 1024, 128, 4, 16
hd = d // H
K_max_raw = 80

# Create test data
x = torch.randint(0, V, (B, L), device='cuda')
Q = torch.randn(B, H, L, hd, device='cuda')
unique_tensor, K_max = _build_unique_index(x, 'cuda')
K_max = max(K_max, 16)

E_k = torch.randn(B, K_max, H * hd, device='cuda') * 0.1
E_v = torch.randn(B, K_max, H * hd, device='cuda') * 0.1
W_pe = torch.randn(N, d, device='cuda') * 0.1
rhos = torch.exp(-torch.log(torch.tensor(2.0)) / torch.logspace(0, 2, N)).cuda()

# ── V1: Original fused kernel ──
print("=== V1 (original fused kernel) ===")
BLOCK_N = max(N, 16)
rhos_p, W_pe_p, _, _ = _pad_n_dim(rhos, W_pe.contiguous(), None, None, BLOCK_N)
C_bounds = precompute_boundaries(x, unique_tensor, rhos, 64, c_init=None)
O_v1, LSE_v1, _, _, _ = flash_ema_forward(
    Q, x, E_k, E_v, W_pe, rhos, V, unique_tensor, K_max, C_bounds, use_silu=True
)
print(f"  O_v1 shape: {O_v1.shape}, mean: {O_v1.mean():.6f}, std: {O_v1.std():.6f}")
print(f"  O_v1 abs max: {O_v1.abs().max():.6f}")
print(f"  LSE_v1 mean: {LSE_v1.mean():.4f}")

# ── V2: Decomposed (cuBLAS gate + attn kernel) ──
print("\n=== V2 (decomposed cuBLAS + attn) ===")
c_all = compute_c_all(x, unique_tensor, rhos, K_max, c_init=None)
print(f"  c_all shape: {c_all.shape}, mean: {c_all.mean():.6f}, std: {c_all.std():.6f}")
print(f"  c_all abs max: {c_all.abs().max():.6f}")
print(f"  c_all[0,0,:5,:4]:\n{c_all[0, 0, :5, :4]}")

O_v2, LSE_v2, P_gate, _, _ = flash_ema_forward_v2(
    Q, E_k, E_v, W_pe, V, unique_tensor, K_max, c_all, use_silu=True
)
print(f"  O_v2 shape: {O_v2.shape}, mean: {O_v2.mean():.6f}, std: {O_v2.std():.6f}")
print(f"  O_v2 abs max: {O_v2.abs().max():.6f}")
print(f"  LSE_v2 mean: {LSE_v2.mean():.4f}")

# ── Compare ──
print("\n=== Comparison ===")
diff_O = (O_v1 - O_v2).abs()
diff_LSE = (LSE_v1 - LSE_v2).abs()
print(f"  O diff:   max={diff_O.max():.8f}, mean={diff_O.mean():.8f}")
print(f"  LSE diff: max={diff_LSE.max():.8f}, mean={diff_LSE.mean():.8f}")

if diff_O.max() < 1e-3:
    print("\n  PASS: V1 and V2 outputs match within tolerance")
else:
    print(f"\n  FAIL: V1 and V2 differ significantly")
    # Find worst position
    worst = diff_O.argmax()
    idx = []
    for s in O_v1.shape[::-1]:
        idx.append(worst % s)
        worst = worst // s
    idx = tuple(reversed(idx))
    print(f"  Worst at {idx}: V1={O_v1[idx]:.6f}, V2={O_v2[idx]:.6f}")

    # Debug: check P_gate sample
    print(f"\n  P_gate shape: {P_gate.shape}")
    print(f"  P_gate mean: {P_gate.mean():.6f}, std: {P_gate.std():.6f}")
    print(f"  P_gate[0, 0, :5, 0, :4]:\n{P_gate[0, 0, :5, 0, :4]}")

    # Also manually compute gate for comparison
    print("\n  Manual gate check (first timestep, first k-slots, head 0):")
    c0 = c_all[0, 0, :5, :]  # [5, N]
    w0 = W_pe[:, :hd]  # [N, hd] for head 0
    manual_gate = torch.nn.functional.silu(c0 @ w0)  # [5, hd]
    print(f"  Manual: {manual_gate[:, :4]}")
    print(f"  P_gate: {P_gate[0, 0, :5, 0, :4]}")
