"""V5b EMA Encoder — Fused lean kernel: no one_hot, Q_max at end.

Two optimizations over V5:
  1. Eliminate one_hot: kernel reads token indices directly, checks token[b,t]==s
     Saves 33MB per chunk allocation + memory traffic.
  2. Fuse Q_max: computed at loop end (one amax per program, not per step).
     Saves separate carry.abs().amax() PyTorch call (0.10ms × 8 = 0.80ms).

Architecture:
  Phase 1 kernel: grid=(B*S), reads token indices, writes carry(B,S,N) + Q_max(B,S).
  Phase 3 kernel: flat pointwise recompute for Q_active + q_query.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True

    @triton.jit
    def ema_fused_kernel(
        TOKENS_ptr, CARRY_ptr, QMAX_ptr, RHO_ptr, INIT_ptr,
        stride_tb, stride_tt,
        stride_cb, stride_cs,
        stride_qb,
        stride_ib, stride_is,
        T: tl.constexpr, S, N,
        HAS_INIT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        V5b fused kernel: reads token indices (not one_hot), writes carry + Q_max.
        No one_hot tensor needed. Q_max computed once at loop end.
        """
        pid = tl.program_id(0)
        s = pid % S
        b = pid // S

        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        rho = tl.load(RHO_ptr + n_offs, mask=n_mask, other=1.0)

        if HAS_INIT:
            q_state = tl.load(
                INIT_ptr + b * stride_ib + s * stride_is + n_offs,
                mask=n_mask, other=0.0)
        else:
            q_state = tl.zeros([BLOCK_N], dtype=tl.float32)

        for t in range(T):
            # Read token index instead of one_hot vector
            tok = tl.load(TOKENS_ptr + b * stride_tb + t * stride_tt)
            x_val = tl.where(s == tok, 1.0, 0.0)
            q_state = q_state * rho + x_val

        # Write carry
        tl.store(
            CARRY_ptr + b * stride_cb + s * stride_cs + n_offs,
            q_state, mask=n_mask)

        # Write Q_max (once at end, not per step)
        abs_q = tl.where(n_mask, tl.abs(q_state), 0.0)
        qmax_val = tl.max(abs_q, axis=0)
        tl.store(QMAX_ptr + b * stride_qb + s, qmax_val)

    @triton.jit
    def ema_pointwise_kernel(
        TOKENS_ptr, INIT_ptr, SLOTS_ptr, Q_OUT_ptr, RHO_ptr,
        stride_tb, stride_tt,
        stride_ib, stride_is,
        S, N, k,
        T_SUB: tl.constexpr,
        HAS_INIT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Flat pointwise EMA recomputation using token indices (no one_hot).
        Grid: (num_items,). pid maps to one (bt, j) pair.
        """
        pid = tl.program_id(0)
        bt = pid // k
        j = pid % k

        b = bt // T_SUB
        t_target = bt % T_SUB
        s = tl.load(SLOTS_ptr + bt * k + j)

        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        rho = tl.load(RHO_ptr + n_offs, mask=n_mask, other=1.0)

        if HAS_INIT:
            q_state = tl.load(
                INIT_ptr + b * stride_ib + s * stride_is + n_offs,
                mask=n_mask, other=0.0)
        else:
            q_state = tl.zeros([BLOCK_N], dtype=tl.float32)

        for t in range(T_SUB):
            if t <= t_target:
                tok = tl.load(TOKENS_ptr + b * stride_tb + t * stride_tt)
                x_val = tl.where(s == tok, 1.0, 0.0)
                q_state = q_state * rho + x_val

        tl.store(Q_OUT_ptr + pid * N + n_offs, q_state, mask=n_mask)

    # ── Legacy V3 kernel ──
    @triton.jit
    def ema_scan_kernel(
        X_ptr, Q_ptr, QMAX_ptr, RHO_ptr, INIT_ptr,
        stride_xb, stride_xt, stride_xs,
        stride_qb, stride_qt, stride_qs, stride_qn,
        stride_mb, stride_mt,
        stride_ib, stride_is,
        T, S, N,
        HAS_INIT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        s = pid % S
        b = pid // S
        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        rho = tl.load(RHO_ptr + n_offs, mask=n_mask, other=1.0)
        if HAS_INIT:
            q_state = tl.load(
                INIT_ptr + b * stride_ib + s * stride_is + n_offs,
                mask=n_mask, other=0.0)
        else:
            q_state = tl.zeros([BLOCK_N], dtype=tl.float32)
        for t in range(T):
            x_val = tl.load(X_ptr + b * stride_xb + t * stride_xt + s * stride_xs)
            q_state = q_state * rho + x_val
            Q_base = Q_ptr + b * stride_qb + t * stride_qt + s * stride_qs
            tl.store(Q_base + n_offs * stride_qn, q_state, mask=n_mask)
            abs_q = tl.where(n_mask, tl.abs(q_state), 0.0)
            q_max_val = tl.max(abs_q, axis=0)
            tl.store(QMAX_ptr + b * stride_mb + t * stride_mt + s, q_max_val)

except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════

def compute_ema(
    x: torch.Tensor, rho: torch.Tensor,
    chunk_size: int = 4096, init_state: torch.Tensor = None,
) -> tuple:
    """Legacy V3 API. Returns (Q, Q_max, carry)."""
    B, T, S = x.shape
    N = rho.shape[0]
    Q     = torch.empty(B, T, S, N, device=x.device, dtype=torch.float32)
    Q_max = torch.empty(B, T, S,    device=x.device, dtype=torch.float32)
    if not HAS_TRITON or x.device.type != 'cuda':
        q = (init_state.clone() if init_state is not None
             else torch.zeros(B, S, N, device=x.device, dtype=torch.float32))
        rho_v = rho.view(1, 1, N)
        for t in range(T):
            q = q * rho_v + x[:, t, :].unsqueeze(-1)
            Q[:, t] = q
            Q_max[:, t] = q.abs().amax(dim=-1)
        return Q, Q_max, q
    BLOCK_N = triton.next_power_of_2(N)
    has_init = init_state is not None
    init_c = init_state.contiguous() if has_init else x
    ema_scan_kernel[(B * S,)](
        x, Q, Q_max, rho, init_c,
        x.stride(0), x.stride(1), x.stride(2),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        Q_max.stride(0), Q_max.stride(1),
        init_c.stride(0) if has_init else 0,
        init_c.stride(1) if has_init else 0,
        T, S, N, HAS_INIT=has_init, BLOCK_N=BLOCK_N)
    return Q, Q_max, Q[:, -1, :, :].contiguous()


def compute_ema_v5(
    tokens: torch.Tensor, rho: torch.Tensor, S: int,
    init_state: torch.Tensor = None,
) -> tuple:
    """
    V5b fused EMA: reads token indices directly, writes carry + Q_max.
    No one_hot tensor needed.

    Args:
        tokens:     (B, T) int64 token indices
        rho:        (N,) decay rates
        S:          vocab size
        init_state: (B, S, N) carry from previous chunk

    Returns: (carry, Q_max) — both NEW tensors.
        carry: (B, S, N)
        Q_max: (B, S)
    """
    assert tokens.is_contiguous()
    B, T = tokens.shape
    N = rho.shape[0]

    carry = torch.empty(B, S, N, device=tokens.device, dtype=torch.float32)
    Q_max = torch.empty(B, S, device=tokens.device, dtype=torch.float32)

    if not HAS_TRITON or tokens.device.type != 'cuda':
        q = (init_state.clone() if init_state is not None
             else torch.zeros(B, S, N, device=tokens.device, dtype=torch.float32))
        rho_v = rho.view(1, 1, N)
        for t in range(T):
            x = torch.zeros(B, S, device=tokens.device)
            x.scatter_(1, tokens[:, t:t+1], 1.0)
            q = q * rho_v + x.unsqueeze(-1)
        carry[:] = q
        Q_max[:] = q.abs().amax(dim=-1)
        return carry, Q_max

    BLOCK_N = triton.next_power_of_2(N)
    has_init = init_state is not None
    init_c = init_state.contiguous() if has_init else tokens

    ema_fused_kernel[(B * S,)](
        tokens, carry, Q_max, rho, init_c,
        tokens.stride(0), tokens.stride(1),
        carry.stride(0), carry.stride(1),
        Q_max.stride(0),
        init_c.stride(0) if has_init else 0,
        init_c.stride(1) if has_init else 0,
        T=T, S=S, N=N, HAS_INIT=has_init, BLOCK_N=BLOCK_N)

    return carry, Q_max


def recompute_slots(
    tokens: torch.Tensor, rho: torch.Tensor, S: int,
    slot_indices: torch.Tensor, init_state: torch.Tensor = None,
) -> torch.Tensor:
    """
    Flat pointwise Q recomputation using token indices (no one_hot).

    Args:
        tokens:       (B, T) int64 token indices
        rho:          (N,) decay rates
        S:            vocab size
        slot_indices: (BT, k) int64 — topk slot indices
        init_state:   (B, S, N) carry from BEFORE this sub-chunk

    Returns:
        Q_out: (BT, k, N)
    """
    B, T = tokens.shape
    N = rho.shape[0]

    was_2d = slot_indices.dim() == 2
    if was_2d:
        BT, k = slot_indices.shape
        total = BT * k
    else:
        total = slot_indices.shape[0]
        k = 1

    if not HAS_TRITON or tokens.device.type != 'cuda':
        q = (init_state.clone() if init_state is not None
             else torch.zeros(B, S, N, device=tokens.device, dtype=torch.float32))
        rho_v = rho.view(1, 1, N)
        Q_out = torch.empty(total, N, device=tokens.device, dtype=torch.float32)
        for t in range(T):
            x = torch.zeros(B, S, device=tokens.device)
            x.scatter_(1, tokens[:, t:t+1], 1.0)
            q = q * rho_v + x.unsqueeze(-1)
            for pid in range(total):
                bt = pid // k
                b_i, t_pos = bt // T, bt % T
                if t == t_pos:
                    s = slot_indices.view(-1)[pid].item()
                    Q_out[pid] = q[b_i, s]
        return Q_out.view(BT, k, N) if was_2d else Q_out

    Q_out = torch.empty(total, N, device=tokens.device, dtype=torch.float32)
    BLOCK_N = triton.next_power_of_2(N)
    has_init = init_state is not None
    init_c = init_state.contiguous() if has_init else tokens

    slots_flat = slot_indices.reshape(-1).contiguous()

    ema_pointwise_kernel[(total,)](
        tokens, init_c, slots_flat, Q_out, rho,
        tokens.stride(0), tokens.stride(1),
        init_c.stride(0) if has_init else 0,
        init_c.stride(1) if has_init else 0,
        S, N, k,
        T_SUB=T, HAS_INIT=has_init, BLOCK_N=BLOCK_N)

    return Q_out.view(BT, k, N) if was_2d else Q_out
