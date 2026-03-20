#!/usr/bin/env python3
"""Benchmark: GPU EMA — Sequential vs Triton Parallel Scan

Parallel scan: O(log T) depth instead of O(T)
EMA 递推: Q[t] = ρ·Q[t-1] + x[t]  →  (a₁,b₁)⊕(a₂,b₂) = (a₁a₂, a₂b₁+b₂)
"""
import sys, time, math
import numpy as np
import torch
import triton
import triton.language as tl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════
# Triton Parallel Scan Kernel
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _combine(a_l, b_l, a_r, b_r):
    """结合律算子: (a₁,b₁) ⊕ (a₂,b₂) = (a₁·a₂, a₂·b₁ + b₂)"""
    return a_l * a_r, a_r * b_l + b_r


@triton.jit
def ema_scan_kernel(
    X_ptr, Q_ptr, RHO_ptr,
    stride_xb, stride_xt, stride_xs,
    stride_qb, stride_qt, stride_qs, stride_qn,
    T, S, N,
    BLOCK_T: tl.constexpr,
):
    """每个 program 处理一个 (b, s, n) 的全 T 序列"""
    pid = tl.program_id(0)
    n = pid % N
    pid_sn = pid // N
    s = pid_sn % S
    b = pid_sn // S

    rho = tl.load(RHO_ptr + n)

    t_offs = tl.arange(0, BLOCK_T)
    mask = t_offs < T

    # Load x[b, :, s]
    x_vals = tl.load(X_ptr + b * stride_xb + t_offs * stride_xt + s * stride_xs,
                     mask=mask, other=0.0)

    # Scan pairs: (a=ρ, b=x[t]) for valid, (a=1, b=0) for padding (identity)
    a_vals = tl.where(mask, rho, 1.0)
    b_vals = x_vals

    # Parallel associative scan: O(log T) depth
    _, q_vals = tl.associative_scan((a_vals, b_vals), 0, _combine)

    # Store Q[b, :, s, n]
    tl.store(Q_ptr + b * stride_qb + t_offs * stride_qt + s * stride_qs + n * stride_qn,
             q_vals, mask=mask)


# ═══════════════════════════════════════════════════════════════
# Python: Chunked parallel scan for long sequences
# ═══════════════════════════════════════════════════════════════

def gpu_ema_sequential(x_gpu, rho_gpu):
    """GPU sequential loop — baseline"""
    B, T, S = x_gpu.shape
    N = rho_gpu.shape[0]
    Q = torch.zeros(B, T, S, N, device=x_gpu.device, dtype=torch.float32)
    rr = rho_gpu.view(1, 1, N)
    Q[:, 0] = x_gpu[:, 0, :, None]
    for t in range(1, T):
        Q[:, t] = Q[:, t-1] * rr + x_gpu[:, t, :, None]
    return Q


def gpu_ema_triton(x_gpu, rho_gpu, chunk_size=4096):
    """Triton parallel scan EMA

    T <= chunk_size: 单个 kernel, O(log T) steps
    T > chunk_size:  分 chunk, 每 chunk parallel scan + carry forward
    """
    B, T, S = x_gpu.shape
    N = rho_gpu.shape[0]
    Q = torch.empty(B, T, S, N, device=x_gpu.device, dtype=torch.float32)

    if T <= chunk_size:
        # Single kernel
        BLOCK_T = triton.next_power_of_2(T)
        grid = (B * S * N,)
        ema_scan_kernel[grid](
            x_gpu, Q, rho_gpu,
            x_gpu.stride(0), x_gpu.stride(1), x_gpu.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            T, S, N,
            BLOCK_T=BLOCK_T,
        )
    else:
        # Chunked: parallel scan per chunk + carry forward between chunks
        BLOCK_T = triton.next_power_of_2(chunk_size)
        grid = (B * S * N,)
        carry = torch.zeros(B, S, N, device=x_gpu.device, dtype=torch.float32)

        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            chunk_len = t_end - t_start

            # 如果有 carry, 在 CPU 端把 carry 注入 x 的第一步
            # x_mod[0] = rho * carry + x[0], 其余不变
            if t_start > 0:
                # 修改 x 的 chunk: x[:, t_start, :] += rho * carry (in-place)
                # 但不能改原始 x, 所以 clone 第一行
                x_first = x_gpu[:, t_start, :].clone()  # (B, S)
                # carry (B, S, N) → 需要把 rho*carry 加到 x 的每个 rho 通道
                # 但 x 只有 (B, S), scan 公式里 b[0] = rho*carry + x[0]
                # 这需要在 kernel 里做。但我们用 Python 方式：
                # 先做 scan 得到 Q_chunk (不含 carry)
                # 然后加上 carry 的衰减：Q[t] += carry * rho^(t+1)
                pass

            # Parallel scan on this chunk (不含 carry)
            x_chunk = x_gpu[:, t_start:t_end].contiguous()
            Q_chunk_tmp = torch.empty(B, chunk_len, S, N,
                                      device=x_gpu.device, dtype=torch.float32)
            BT = triton.next_power_of_2(chunk_len)
            ema_scan_kernel[grid](
                x_chunk, Q_chunk_tmp, rho_gpu,
                x_chunk.stride(0), x_chunk.stride(1), x_chunk.stride(2),
                Q_chunk_tmp.stride(0), Q_chunk_tmp.stride(1),
                Q_chunk_tmp.stride(2), Q_chunk_tmp.stride(3),
                chunk_len, S, N,
                BLOCK_T=BT,
            )

            if t_start > 0:
                # 加上 carry 的衰减: Q_adjusted[t] = Q_scan[t] + carry * rho^(t+1)
                # rho_powers (chunk_len, N): rho^1, rho^2, ..., rho^chunk_len
                t_indices = torch.arange(1, chunk_len + 1, device=x_gpu.device,
                                         dtype=torch.float32)  # (chunk_len,)
                rho_powers = rho_gpu.unsqueeze(0).pow(
                    t_indices.unsqueeze(1))  # (chunk_len, N)
                # carry: (B, S, N) → (B, 1, S, N) * (1, chunk_len, 1, N)
                Q_chunk_tmp += carry.unsqueeze(1) * rho_powers.unsqueeze(0).unsqueeze(2)

            # Write to Q
            Q[:, t_start:t_end] = Q_chunk_tmp

            # Update carry = Q[:, t_end-1, :, :]
            carry = Q[:, t_end - 1, :, :].clone()

    return Q


# ═══════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda')
    rho = np.array([0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999], np.float32)
    N = len(rho)
    rho_gpu = torch.from_numpy(rho).to(device)

    configs = [
        ("Short (sentence)",    500,   50,  100),    # B, T, S
        ("Medium (paragraph)",  100,  500,  100),
        ("Long (document)",      10, 5000,  100),
        ("Very long (book)",      2, 50000,  50),
    ]

    for name, B, T, S in configs:
        print(f"\n{'='*70}")
        print(f"{name}: B={B}, T={T}, S={S}, N={N}")
        data_mb = B * T * S * 4 / 1e6
        out_mb = B * T * S * N * 4 / 1e6
        print(f"  Input: {data_mb:.1f}MB, Output: {out_mb:.1f}MB")

        x = np.zeros((B, T, S), dtype=np.float32)
        mask_np = np.random.random((B, T, S)) < 0.1
        x[mask_np] = np.random.randint(1, 5, size=mask_np.sum()).astype(np.float32)
        x_gpu = torch.from_numpy(x).to(device)

        # ── GPU Sequential (ground truth) ──
        Q_seq = gpu_ema_sequential(x_gpu, rho_gpu)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        Q_seq = gpu_ema_sequential(x_gpu, rho_gpu)
        torch.cuda.synchronize()
        t_seq = (time.perf_counter() - t0) * 1000
        print(f"  GPU Sequential:    {t_seq:.2f}ms  ({T} steps)")

        # ── Triton Parallel Scan ──
        Q_tri = gpu_ema_triton(x_gpu, rho_gpu, chunk_size=4096)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        Q_tri = gpu_ema_triton(x_gpu, rho_gpu, chunk_size=4096)
        torch.cuda.synchronize()
        t_tri = (time.perf_counter() - t0) * 1000

        if T <= 4096:
            label = f"log₂({T})={math.log2(T):.0f} steps"
        else:
            n_chunks = (T + 4095) // 4096
            label = f"{n_chunks} chunks"

        print(f"  Triton Scan:       {t_tri:.2f}ms  ({label})")

        # Correctness
        max_diff = (Q_seq - Q_tri).abs().max().item()
        rel_max = max_diff / (Q_seq.abs().max().item() + 1e-8)
        print(f"  Max abs diff:      {max_diff:.6f}")
        print(f"  Max rel diff:      {rel_max:.2e}")
        print(f"  Speedup:           {t_seq/t_tri:.1f}x")


if __name__ == '__main__':
    main()
