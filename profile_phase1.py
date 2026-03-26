#!/usr/bin/env python3
"""
profile_phase1.py — Isolate Phase 1 kernel costs with variant kernels.

Creates 5 kernel variants to measure exactly where 2.10ms is spent:
  [K0] bare_scan:      FMA loop only, NO writes at all → pure serial compute cost
  [K1] scan+qmax:      FMA + amax reduction + Q_max write
  [K2] scan+carry:     FMA + carry write (only at end)
  [K3] scan+qquery:    FMA + token match + conditional q_query write
  [K4] full_phase1:    all outputs (= production kernel)

Also tests:
  [P1] one_hot alloc cost
  [P2] torch.topk on Q_max
  [P3] ema_chunk sensitivity (C=16,32,64,128)

Usage:
    python profile_phase1.py --vocab 8000 --batch-size 32 --seq-len 256
"""
import argparse
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    @triton.jit
    def k0_bare_scan(
        X_ptr, DUMMY_ptr, RHO_ptr, INIT_ptr,
        stride_xb, stride_xt, stride_xs,
        stride_ib, stride_is,
        T: tl.constexpr, S, N,
        HAS_INIT: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """K0: bare FMA scan, minimal writes. Measures pure serial compute cost."""
        pid = tl.program_id(0)
        s = pid % S; b = pid // S
        n_offs = tl.arange(0, BLOCK_N); n_mask = n_offs < N
        rho = tl.load(RHO_ptr + n_offs, mask=n_mask, other=1.0)
        if HAS_INIT:
            q = tl.load(INIT_ptr + b * stride_ib + s * stride_is + n_offs, mask=n_mask, other=0.0)
        else:
            q = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in range(T):
            x = tl.load(X_ptr + b * stride_xb + t * stride_xt + s * stride_xs)
            q = q * rho + x
        # Store one float to prevent DCE (negligible overhead)
        tl.store(DUMMY_ptr + pid, tl.sum(q))

    @triton.jit
    def k1_scan_qmax(
        X_ptr, QMAX_ptr, RHO_ptr, INIT_ptr,
        stride_xb, stride_xt, stride_xs,
        stride_mb, stride_mt,
        stride_ib, stride_is,
        T: tl.constexpr, S, N,
        HAS_INIT: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """K1: FMA scan + amax reduction + Q_max write."""
        pid = tl.program_id(0)
        s = pid % S; b = pid // S
        n_offs = tl.arange(0, BLOCK_N); n_mask = n_offs < N
        rho = tl.load(RHO_ptr + n_offs, mask=n_mask, other=1.0)
        if HAS_INIT:
            q = tl.load(INIT_ptr + b * stride_ib + s * stride_is + n_offs, mask=n_mask, other=0.0)
        else:
            q = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in range(T):
            x = tl.load(X_ptr + b * stride_xb + t * stride_xt + s * stride_xs)
            q = q * rho + x
            abs_q = tl.where(n_mask, tl.abs(q), 0.0)
            qm = tl.max(abs_q, axis=0)
            tl.store(QMAX_ptr + b * stride_mb + t * stride_mt + s, qm)

    @triton.jit
    def k2_scan_carry(
        X_ptr, CARRY_ptr, RHO_ptr, INIT_ptr,
        stride_xb, stride_xt, stride_xs,
        stride_cb, stride_cs,
        stride_ib, stride_is,
        T: tl.constexpr, S, N,
        HAS_INIT: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """K2: FMA scan + carry write at end only."""
        pid = tl.program_id(0)
        s = pid % S; b = pid // S
        n_offs = tl.arange(0, BLOCK_N); n_mask = n_offs < N
        rho = tl.load(RHO_ptr + n_offs, mask=n_mask, other=1.0)
        if HAS_INIT:
            q = tl.load(INIT_ptr + b * stride_ib + s * stride_is + n_offs, mask=n_mask, other=0.0)
        else:
            q = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in range(T):
            x = tl.load(X_ptr + b * stride_xb + t * stride_xt + s * stride_xs)
            q = q * rho + x
        tl.store(CARRY_ptr + b * stride_cb + s * stride_cs + n_offs, q, mask=n_mask)

    @triton.jit
    def k3_scan_qquery(
        X_ptr, TOKENS_ptr, QQUERY_ptr, RHO_ptr, INIT_ptr,
        stride_xb, stride_xt, stride_xs,
        stride_tb,
        stride_ib, stride_is,
        T: tl.constexpr, S, N,
        HAS_INIT: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """K3: FMA scan + token match + conditional q_query write."""
        pid = tl.program_id(0)
        s = pid % S; b = pid // S
        n_offs = tl.arange(0, BLOCK_N); n_mask = n_offs < N
        rho = tl.load(RHO_ptr + n_offs, mask=n_mask, other=1.0)
        if HAS_INIT:
            q = tl.load(INIT_ptr + b * stride_ib + s * stride_is + n_offs, mask=n_mask, other=0.0)
        else:
            q = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in range(T):
            x = tl.load(X_ptr + b * stride_xb + t * stride_xt + s * stride_xs)
            q = q * rho + x
            tok = tl.load(TOKENS_ptr + b * stride_tb + t)
            if s == tok:
                bt = b * T + t
                tl.store(QQUERY_ptr + bt * N + n_offs, q, mask=n_mask)

except ImportError:
    pass


def cuda_time(fn, n_warmup=3, n_repeat=10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sum(times)/len(times), min(times), max(times)

def fmt(m, lo=None, hi=None):
    if lo is not None:
        return f"{m:8.2f} ms  (best {lo:.2f}, worst {hi:.2f})"
    return f"{m:8.2f} ms"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab',       type=int, default=8000)
    p.add_argument('--batch-size',  type=int, default=32)
    p.add_argument('--seq-len',     type=int, default=256)
    p.add_argument('--ema-chunk',   type=int, default=32)
    p.add_argument('--N',           type=int, default=64)
    p.add_argument('--h-min',       type=float, default=1)
    p.add_argument('--h-max',       type=float, default=1000)
    p.add_argument('--n-warmup',    type=int, default=3)
    p.add_argument('--n-repeat',    type=int, default=10)
    args = p.parse_args()

    device = torch.device('cuda')
    B, T, S, N = args.batch_size, args.seq_len, args.vocab, args.N
    C = args.ema_chunk
    W, R = args.n_warmup, args.n_repeat

    from mathbrain.config import MathBrainConfig
    from mathbrain.encoder import compute_ema_v4

    cfg = MathBrainConfig.from_half_lives(
        h_min=args.h_min, h_max=args.h_max, vocab_size=S, retina_mode='bpe',
        N=N, d_model=64, n_layers=2, n_heads=8, d_ff=256)

    rho = torch.tensor(cfg.rho_array, device=device)
    inputs = torch.randint(0, S, (B, T), device=device)
    carry = torch.zeros(B, S, N, device=device)

    sub = inputs[:, :C]
    x_sub = F.one_hot(sub, num_classes=S).float()
    tokens = sub.contiguous()

    BLOCK_N = triton.next_power_of_2(N)
    grid = (B * S,)

    Q_max  = torch.empty(B, C, S, device=device)
    carry_out = torch.empty(B, S, N, device=device)
    q_query = torch.empty(B * C, N, device=device)
    dummy = torch.empty(B * S, device=device)

    print(f"\nDevice: {device}  |  B={B} T={T} S={S} N={N} C={C}")
    print(f"Grid: {B*S} programs  |  BLOCK_N={BLOCK_N}")
    print(f"Warmup={W}  Repeat={R}")

    # ═══════════════════════════════════════════════════════════════
    # PART 1: Variant kernel breakdown
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PART 1: Phase 1 Kernel Variants  (tc={C})")
    print(f"{'='*60}")

    # [K0] bare scan
    def run_k0():
        k0_bare_scan[grid](
            x_sub, dummy, rho, carry,
            x_sub.stride(0), x_sub.stride(1), x_sub.stride(2),
            carry.stride(0), carry.stride(1),
            T=C, S=S, N=N, HAS_INIT=True, BLOCK_N=BLOCK_N)
    m,lo,hi = cuda_time(run_k0, W, R)
    print(f"  [K0] bare scan (no writes)   {fmt(m,lo,hi)}")
    print(f"       reads: x={B*C*S*4/1e6:.0f}MB + init={B*S*N*4/1e6:.0f}MB + rho=0MB")

    # [K1] scan + Q_max
    def run_k1():
        k1_scan_qmax[grid](
            x_sub, Q_max, rho, carry,
            x_sub.stride(0), x_sub.stride(1), x_sub.stride(2),
            Q_max.stride(0), Q_max.stride(1),
            carry.stride(0), carry.stride(1),
            T=C, S=S, N=N, HAS_INIT=True, BLOCK_N=BLOCK_N)
    m1,lo,hi = cuda_time(run_k1, W, R)
    print(f"  [K1] scan + Q_max write      {fmt(m1,lo,hi)}")
    print(f"       + writes: Q_max={B*C*S*4/1e6:.0f}MB")

    # [K2] scan + carry
    def run_k2():
        k2_scan_carry[grid](
            x_sub, carry_out, rho, carry,
            x_sub.stride(0), x_sub.stride(1), x_sub.stride(2),
            carry_out.stride(0), carry_out.stride(1),
            carry.stride(0), carry.stride(1),
            T=C, S=S, N=N, HAS_INIT=True, BLOCK_N=BLOCK_N)
    m2,lo,hi = cuda_time(run_k2, W, R)
    print(f"  [K2] scan + carry write      {fmt(m2,lo,hi)}")
    print(f"       + writes: carry={B*S*N*4/1e6:.0f}MB")

    # [K3] scan + q_query
    def run_k3():
        k3_scan_qquery[grid](
            x_sub, tokens, q_query, rho, carry,
            x_sub.stride(0), x_sub.stride(1), x_sub.stride(2),
            tokens.stride(0),
            carry.stride(0), carry.stride(1),
            T=C, S=S, N=N, HAS_INIT=True, BLOCK_N=BLOCK_N)
    m3,lo,hi = cuda_time(run_k3, W, R)
    print(f"  [K3] scan + q_query write    {fmt(m3,lo,hi)}")
    print(f"       + reads: tokens={B*C*4/1e6:.2f}MB  + cond store")

    # [K4] full phase1
    def run_k4():
        compute_ema_v4(x_sub, rho, sub, init_state=carry)
    m4,lo,hi = cuda_time(run_k4, W, R)
    print(f"  [K4] full phase1 (prod)      {fmt(m4,lo,hi)}")

    # Incremental costs
    print(f"\n  --- Incremental cost analysis ---")
    print(f"  Base serial FMA:        {m:.3f} ms  (K0)")
    print(f"  + amax + Q_max write:  +{m1-m:.3f} ms  (K1-K0)")
    print(f"  + carry write:         +{m2-m:.3f} ms  (K2-K0)")
    print(f"  + token match+qquery:  +{m3-m:.3f} ms  (K3-K0)")

    # ═══════════════════════════════════════════════════════════════
    # PART 2: ema_chunk sensitivity
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PART 2: ema_chunk Sensitivity  (total T={T})")
    print(f"{'='*60}")

    from mathbrain.trainer import extract_topk_indices
    from mathbrain.encoder import recompute_active_q

    for chunk in [16, 32, 64, 128]:
        if chunk > T:
            continue
        n_iters = (T + chunk - 1) // chunk

        def full_pipeline(C_=chunk):
            c = torch.zeros(B, S, N, device=device)
            with torch.no_grad():
                for t0 in range(0, T, C_):
                    t1 = min(t0+C_, T); tc = t1-t0
                    sub_in = inputs[:, t0:t1]
                    xs = F.one_hot(sub_in, num_classes=S).float()
                    prev_c = c
                    Qm, qq, c = compute_ema_v4(xs, rho, sub_in, init_state=c)
                    c = c.detach()
                    Qmf = Qm.reshape(B*tc, S)
                    si, pm = extract_topk_indices(Qmf, cfg.eps_q, cfg.max_active_slots)
                    recompute_active_q(xs, rho, si, init_state=prev_c)
                    del Qm, xs
        m,lo,hi = cuda_time(full_pipeline, W, R)
        print(f"  C={chunk:3d}  iters={n_iters:2d}  total={fmt(m,lo,hi)}  per_tok={m*1000/(B*T):.0f}µs")

    # ═══════════════════════════════════════════════════════════════
    # PART 3: one_hot cost
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  PART 3: one_hot / topk / alloc costs")
    print(f"{'='*60}")

    m,lo,hi = cuda_time(lambda: F.one_hot(sub, num_classes=S).float(), W, R)
    print(f"  one_hot (B,{C},{S})        {fmt(m,lo,hi)}")

    Qm_flat = Q_max.reshape(B*C, S)
    m,lo,hi = cuda_time(lambda: torch.topk(Qm_flat, 32, dim=1, sorted=False), W, R)
    print(f"  topk(k=32) on ({B*C},{S})  {fmt(m,lo,hi)}")

    m,lo,hi = cuda_time(lambda: torch.empty(B, C, S, device=device), W, R)
    print(f"  alloc Q_max ({B},{C},{S})   {fmt(m,lo,hi)}")

    m,lo,hi = cuda_time(lambda: torch.empty(B, S, N, device=device), W, R)
    print(f"  alloc carry ({B},{S},{N})   {fmt(m,lo,hi)}")

    m,lo,hi = cuda_time(lambda: torch.empty(B, C, S, N, device=device), W, R)
    print(f"  alloc Q ({B},{C},{S},{N})   {fmt(m,lo,hi)}")

    print()

if __name__ == '__main__':
    main()
