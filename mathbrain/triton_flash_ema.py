"""
Flash-EMA Triton Kernels — Clean Rewrite with Variant C Strict Causality
=========================================
Forward + Backward kernels for O(L) memory EMA-gated cross-attention.

Architecture (matching MultiplicativeTrueEMATransformer):
    C_decayed = C_t * rho
    P_gate_K = SiLU(C_decayed @ W_pe)           (Variant C strict causal gating)
    K_t = E_k * P_gate_K, V_t = E_v * P_gate_K   (multiplicative modulation)
    O_t = softmax(Q_t @ K_t^T / sqrt(d)) @ V_t    (standard attention)

All EMA state computation happens in SRAM registers. Only Q, O, LSE touch HBM.
"""
import math
import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─────────────────────────────────────────────────
# 1. Boundary Precomputation Kernel
# ─────────────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _c_boundary_scan(
        x_ptr, unique_ptr, rhos_ptr, c_bounds_ptr, c_init_ptr,
        doc_ids_ptr, has_doc_ids: tl.constexpr,
        B, L, K_max, N: tl.constexpr, Br: tl.constexpr,
        num_chunks,
        BLOCK_K: tl.constexpr,
    ):
        """Precompute EMA state C at the start of each Br-sized time chunk.
        Resets C at document boundaries when has_doc_ids is True."""
        pid_b = tl.program_id(0)
        pid_k = tl.program_id(1)

        k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_off < K_max
        uk = tl.load(unique_ptr + pid_b * K_max + k_off, mask=k_mask, other=-1)

        n_off = tl.arange(0, N)
        rhos = tl.load(rhos_ptr + n_off)

        # Initialize C from c_init (cross-chunk EMA history) instead of zeros
        c_init_off = pid_b * K_max * N + k_off[:, None] * N + n_off[None, :]
        C = tl.load(c_init_ptr + c_init_off, mask=k_mask[:, None], other=0.0)

        prev_doc = -1
        if has_doc_ids:
            prev_doc = tl.load(doc_ids_ptr + pid_b * L + 0)

        for chunk in range(num_chunks):
            # Store boundary BEFORE processing this chunk
            out_off = (pid_b * num_chunks * K_max * N
                       + chunk * K_max * N
                       + k_off[:, None] * N + n_off[None, :])
            tl.store(c_bounds_ptr + out_off, C, mask=k_mask[:, None])

            t_start = chunk * Br
            for i in range(Br):
                t = t_start + i
                mt = t < L
                xt = tl.load(x_ptr + pid_b * L + t, mask=mt, other=-1)

                # Reset C at document boundary
                if has_doc_ids:
                    cur_doc = tl.load(doc_ids_ptr + pid_b * L + t, mask=mt, other=-1)
                    boundary = (cur_doc != prev_doc) & (t > 0) & mt
                    keep = 1.0 - boundary.to(tl.float32)
                    C = C * keep  # zero all slots at boundary
                    prev_doc = cur_doc

                match = ((xt == uk) & mt).to(tl.float32)
                C = C * rhos[None, :] + match[:, None]


def precompute_boundaries(x, unique_tensor, rhos, Br, c_init=None, doc_ids=None):
    B, L = x.shape
    K_max = unique_tensor.shape[1]
    N = rhos.shape[0]
    BLOCK_N = max(N, 16)  # tl.dot minimum
    num_chunks = max(1, triton.cdiv(L, Br)) if HAS_TRITON else 1
    c_bounds = torch.empty(B, num_chunks, K_max, BLOCK_N, device=x.device, dtype=torch.float32)

    if HAS_TRITON:
        if c_init is None:
            c_init = torch.zeros(B, K_max, BLOCK_N, device=x.device, dtype=torch.float32)
        else:
            if c_init.shape[-1] < BLOCK_N:
                pad = BLOCK_N - c_init.shape[-1]
                c_init = torch.cat([c_init, torch.zeros(*c_init.shape[:2], pad, device=x.device, dtype=c_init.dtype)], dim=-1)

        rhos_p = rhos
        if N < BLOCK_N:
            rhos_p = torch.cat([rhos, torch.zeros(BLOCK_N - N, device=x.device, dtype=rhos.dtype)])

        has_doc_ids = doc_ids is not None
        if not has_doc_ids:
            doc_ids = torch.zeros(B, L, dtype=torch.int32, device=x.device)

        BLOCK_K = max(16, min(64, triton.next_power_of_2(K_max)))
        grid = (B, triton.cdiv(K_max, BLOCK_K))
        _c_boundary_scan[grid](
            x, unique_tensor, rhos_p, c_bounds, c_init,
            doc_ids, has_doc_ids,
            B, L, K_max, BLOCK_N, Br, num_chunks,
            BLOCK_K=BLOCK_K,
        )
    return c_bounds


# ─────────────────────────────────────────────────
# 1b. Full EMA Scan — stores c_decayed at EVERY timestep
#     (for the decomposed gate projection approach)
# ─────────────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _c_full_scan(
        x_ptr, unique_ptr, rhos_ptr, c_all_ptr, c_init_ptr,
        B, L, K_max, N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute c_decayed (after decay, before match update) at every timestep.
        Output: c_all[B, L, K_max, N] — the value used for gate projection."""
        pid_b = tl.program_id(0)
        pid_k = tl.program_id(1)

        k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_off < K_max
        uk = tl.load(unique_ptr + pid_b * K_max + k_off, mask=k_mask, other=-1)

        n_off = tl.arange(0, N)
        rhos = tl.load(rhos_ptr + n_off)

        c_init_off = pid_b * K_max * N + k_off[:, None] * N + n_off[None, :]
        C = tl.load(c_init_ptr + c_init_off, mask=k_mask[:, None], other=0.0)

        for t in range(L):
            xt = tl.load(x_ptr + pid_b * L + t, mask=True, other=-1)
            match = (xt == uk).to(tl.float32)

            # Decay first (Variant C: gate sees decayed state BEFORE match)
            c_decayed = C * rhos[None, :]

            # Store c_decayed at this timestep
            out_off = (pid_b * L * K_max * N
                       + t * K_max * N
                       + k_off[:, None] * N + n_off[None, :])
            tl.store(c_all_ptr + out_off, c_decayed, mask=k_mask[:, None])

            # Update: match AFTER storing (Variant C)
            C = c_decayed + match[:, None]


def compute_c_all(x, unique_tensor, rhos, K_max, c_init=None):
    """Compute c_decayed at every timestep for all vocab slots. Shared across layers."""
    B, L = x.shape
    N = rhos.shape[0]
    device = x.device

    c_all = torch.empty(B, L, K_max, N, device=device, dtype=torch.float32)

    if c_init is None:
        c_init = torch.zeros(B, K_max, N, device=device, dtype=torch.float32)

    BLOCK_K = max(16, min(64, triton.next_power_of_2(K_max)))
    grid = (B, triton.cdiv(K_max, BLOCK_K))
    _c_full_scan[grid](
        x, unique_tensor, rhos, c_all, c_init,
        B, L, K_max, N,
        BLOCK_K=BLOCK_K,
    )
    return c_all


# ─────────────────────────────────────────────────
# 1c. Attention-only kernel — uses precomputed P_gate
# ─────────────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _attn_with_gate(
        Q, E_k, E_v, P_gate, unique_ptr,
        Out, M_ptr, L_ptr, LSE_out,
        stride_qb, stride_qh, stride_ql, stride_qd,
        B, H, L, hd: tl.constexpr, K_max, V_total,
        num_k_chunks,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Attention kernel using precomputed P_gate. No tl.dot needed."""
        pid_bh = tl.program_id(0)
        pid_tc = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        t_start = pid_tc * BLOCK_M
        offs_d = tl.arange(0, hd)

        for kc in range(num_k_chunks):
            k_start = kc * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K_max

            ek_off = b * K_max * H * hd + offs_k[:, None] * H * hd + h * hd + offs_d[None, :]
            e_k = tl.load(E_k + ek_off, mask=k_mask[:, None], other=0.0)
            e_v = tl.load(E_v + ek_off, mask=k_mask[:, None], other=0.0)

            uk = tl.load(unique_ptr + b * K_max + offs_k, mask=k_mask, other=-1)
            valid_k = (uk != -1) & k_mask

            for t_step in range(BLOCK_M):
                t_global = t_start + t_step
                mt = t_global < L

                # Load precomputed gate for this (b, t, k_chunk, head)
                pg_off = (b * L * K_max * H * hd
                          + t_global * K_max * H * hd
                          + offs_k[:, None] * H * hd
                          + h * hd + offs_d[None, :])
                p_gate = tl.load(P_gate + pg_off, mask=k_mask[:, None] & mt, other=0.0)

                k_dyn = e_k * p_gate
                v_dyn = e_v * p_gate

                q_ptrs = Q + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd
                q_t = tl.load(q_ptrs, mask=mt, other=0.0)

                s_t = tl.sum(q_t[None, :] * k_dyn, axis=1) / math.sqrt(hd)
                valid = valid_k & mt
                s_t = tl.where(valid, s_t, float('-inf'))

                m_ptr = M_ptr + b * H * L + h * L + t_global
                l_ptr = L_ptr + b * H * L + h * L + t_global
                o_ptrs = Out + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd

                if kc == 0:
                    m_old = -float('inf')
                    l_old = (V_total - K_max).to(tl.float32)
                    o_old = tl.zeros((hd,), dtype=tl.float32)
                else:
                    m_old = tl.load(m_ptr, mask=mt, other=float('-inf'))
                    l_old = tl.load(l_ptr, mask=mt, other=0.0)
                    o_old = tl.load(o_ptrs, mask=mt, other=0.0)

                m_chunk = tl.max(s_t)
                m_new = tl.maximum(m_old, m_chunk)
                alpha = tl.exp(m_old - m_new)
                exp_s = tl.where(valid, tl.exp(s_t - m_new), 0.0)

                l_new = l_old * alpha + tl.sum(exp_s)
                o_new = o_old * alpha + tl.sum(exp_s[:, None] * v_dyn, axis=0)

                tl.store(m_ptr, m_new, mask=mt)
                tl.store(l_ptr, l_new, mask=mt)
                tl.store(o_ptrs, o_new, mask=mt)

                if kc == num_k_chunks - 1:
                    lse_val = m_new + tl.log(l_new)
                    lse_ptr = LSE_out + b * H * L + h * L + t_global
                    tl.store(lse_ptr, lse_val, mask=mt)


def flash_ema_forward_v2(Q, E_k_active, E_v_active, W_pe, V_total, unique_tensor, K_max, c_all, use_silu=True, Br=64):
    """Decomposed forward: cuBLAS gate projection + Triton attention kernel."""
    B, H, L, hd = Q.shape
    D = H * hd
    device = Q.device
    N = rhos.shape[0]

    E_k = E_k_active.view(B, K_max, H, hd).contiguous()
    E_v = E_v_active.view(B, K_max, H, hd).contiguous()

    # ── Stage 1: Gate projection via cuBLAS (one large matmul per time chunk) ──
    # c_all: [B, L, K_max, N], W_pe: [N, D] (transposed from pe_proj.weight)
    # P_gate = SiLU(c_all @ W_pe) → [B, L, K_max, D]
    # Process in time chunks of Br to limit memory
    P_gate = torch.empty(B, L, K_max, D, device=device, dtype=torch.float32)

    num_chunks = max(1, triton.cdiv(L, Br))
    for tc in range(num_chunks):
        t_start = tc * Br
        t_end = min(t_start + Br, L)
        t_len = t_end - t_start

        # [B * t_len * K_max, N] × [N, D] → [B * t_len * K_max, D]
        c_chunk = c_all[:, t_start:t_end, :, :].reshape(-1, N)
        pre_act = c_chunk @ W_pe  # cuBLAS handles this efficiently
        if use_silu:
            import torch.nn.functional as F
            pg = F.silu(pre_act)
        else:
            pg = pre_act
        P_gate[:, t_start:t_end, :, :] = pg.view(B, t_len, K_max, D)

    # Reshape P_gate to [B, L, K_max, H, hd] for per-head access in kernel
    P_gate = P_gate.view(B, L, K_max, H, hd).contiguous()

    # ── Stage 2: Attention kernel (no gate computation, just loads P_gate) ──
    O = torch.zeros_like(Q)
    M_val = torch.full((B, H, L), float('-inf'), device=device, dtype=torch.float32)
    L_val = torch.full((B, H, L), float(V_total - K_max), device=device, dtype=torch.float32)
    LSE = torch.empty((B, H, L), device=device, dtype=torch.float32)

    BLOCK_K = max(16, min(64, triton.next_power_of_2(K_max)))
    num_k_chunks = triton.cdiv(K_max, BLOCK_K)

    grid = (B * H, num_chunks)
    _attn_with_gate[grid](
        Q, E_k, E_v, P_gate, unique_tensor,
        O, M_val, L_val, LSE,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        B, H, L, hd, K_max, V_total,
        num_k_chunks,
        BLOCK_M=Br, BLOCK_K=BLOCK_K,
    )

    O = O / L_val.unsqueeze(-1)
    return O, LSE, P_gate, E_k, E_v


# ─────────────────────────────────────────────────
# 2. Original Forward Kernel (kept for backward compatibility)
# ─────────────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _flash_ema_fwd(
        Q, x_labels, E_k, E_v, W_pe, rhos_ptr, C_bounds, unique_ptr,
        Out, M_ptr, L_ptr, LSE_out,
        doc_ids_ptr, has_doc_ids: tl.constexpr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        B, H, L, hd: tl.constexpr, K_max, V_total,
        num_chunks, num_k_chunks,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, USE_SILU: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_tc = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        t_start = pid_tc * BLOCK_M
        offs_d = tl.arange(0, hd)

        # Load initial prev_doc for boundary detection
        prev_doc = -1
        if has_doc_ids:
            prev_doc = tl.load(doc_ids_ptr + b * L + t_start, mask=t_start < L, other=-1)

        for kc in range(num_k_chunks):
            k_start = kc * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K_max

            offs_n = tl.arange(0, BLOCK_N)
            c_off = (b * num_chunks * K_max * BLOCK_N
                     + pid_tc * K_max * BLOCK_N
                     + offs_k[:, None] * BLOCK_N + offs_n[None, :])
            c_local = tl.load(C_bounds + c_off, mask=k_mask[:, None], other=0.0)

            ek_off = b * K_max * H * hd + offs_k[:, None] * H * hd + h * hd + offs_d[None, :]
            e_k = tl.load(E_k + ek_off, mask=k_mask[:, None], other=0.0)
            e_v = tl.load(E_v + ek_off, mask=k_mask[:, None], other=0.0)

            wpe_off = offs_n[:, None] * H * hd + h * hd + offs_d[None, :]
            w_pe = tl.load(W_pe + wpe_off)
            w_pe_tc = w_pe.to(tl.bfloat16)  # hoist cast outside t-loop (invariant)

            uk = tl.load(unique_ptr + b * K_max + offs_k, mask=k_mask, other=-1)
            valid_k = (uk != -1) & k_mask  # hoist validity check (invariant across t)
            rhos_v = tl.load(rhos_ptr + offs_n)

            # Reset prev_doc tracking for each k-chunk (same time range)
            if has_doc_ids:
                prev_doc = tl.load(doc_ids_ptr + b * L + t_start, mask=t_start < L, other=-1)

            for t_step in range(BLOCK_M):
                t_global = t_start + t_step
                mt = t_global < L

                x_base = x_labels + b * L + t_global
                x_val = tl.load(x_base, mask=mt, other=-1)
                match = ((x_val == uk) & mt).to(tl.float32)

                # Document boundary: reset C
                if has_doc_ids:
                    cur_doc = tl.load(doc_ids_ptr + b * L + t_global, mask=mt, other=-1)
                    boundary = (cur_doc != prev_doc) & (t_step > 0) & mt
                    keep = 1.0 - boundary.to(tl.float32)
                    c_local = c_local * keep
                    prev_doc = cur_doc

                c_decayed = c_local * rhos_v[None, :]

                # Tensor Core matmul
                c_decayed_tc = c_decayed.to(tl.bfloat16)
                pre_act = tl.dot(c_decayed_tc, w_pe_tc, out_dtype=tl.float32)

                if USE_SILU:
                    sig = tl.sigmoid(pre_act)
                    p_gate = pre_act * sig
                else:
                    p_gate = pre_act

                k_dyn = e_k * p_gate                             
                v_dyn = e_v * p_gate                             

                q_ptrs = Q + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd
                q_t = tl.load(q_ptrs, mask=mt, other=0.0)  

                # Score
                s_t = tl.sum(q_t[None, :] * k_dyn, axis=1) / math.sqrt(hd)
                valid = valid_k & mt
                s_t = tl.where(valid, s_t, float('-inf'))

                # L2 Cache accumulators!
                m_ptr = M_ptr + b * H * L + h * L + t_global
                l_ptr = L_ptr + b * H * L + h * L + t_global
                o_ptrs = Out + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd

                # Initial values for M and L handle the V_total scaling dynamically
                if kc == 0:
                    m_old = -float('inf')
                    l_old = (V_total - K_max).to(tl.float32)
                    o_old = tl.zeros((hd,), dtype=tl.float32)
                else:
                    m_old = tl.load(m_ptr, mask=mt, other=float('-inf'))
                    l_old = tl.load(l_ptr, mask=mt, other=0.0)
                    o_old = tl.load(o_ptrs, mask=mt, other=0.0)

                m_chunk = tl.max(s_t)
                m_new = tl.maximum(m_old, m_chunk)
                alpha = tl.exp(m_old - m_new)
                exp_s = tl.where(valid, tl.exp(s_t - m_new), 0.0)

                l_new = l_old * alpha + tl.sum(exp_s)
                o_new = o_old * alpha + tl.sum(exp_s[:, None] * v_dyn, axis=0)

                tl.store(m_ptr, m_new, mask=mt)
                tl.store(l_ptr, l_new, mask=mt)
                tl.store(o_ptrs, o_new, mask=mt)
                
                if kc == num_k_chunks - 1:
                    lse_val = m_new + tl.log(l_new)
                    lse_ptr = LSE_out + b * H * L + h * L + t_global
                    tl.store(lse_ptr, lse_val, mask=mt)

                c_local = c_decayed + match[:, None]


# ─────────────────────────────────────────────────
# 3. Backward Kernels — Split into two passes to
#    reduce register pressure (255 regs → ~180 each)
# ─────────────────────────────────────────────────
if HAS_TRITON:
    @triton.jit
    def _flash_ema_bwd_dq_de(
        Q, x_labels, E_k, E_v, W_pe, rhos_ptr, C_bounds, unique_ptr,
        dO, D_vals, LSE,
        dQ_out, dE_k_out, dE_v_out,
        doc_ids_ptr, has_doc_ids: tl.constexpr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        B, H, L, hd: tl.constexpr, K_max, V_total,
        num_chunks, num_k_chunks,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, USE_SILU: tl.constexpr,
    ):
        """Pass 1: Computes dQ, dE_k, dE_v. Skips dW_pe entirely."""
        pid_bh = tl.program_id(0)
        pid_tc = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        t_start = pid_tc * BLOCK_M
        offs_d = tl.arange(0, hd)
        offs_n = tl.arange(0, BLOCK_N)

        wpe_off = offs_n[:, None] * H * hd + h * hd + offs_d[None, :]
        w_pe = tl.load(W_pe + wpe_off)
        w_pe_tc = w_pe.to(tl.bfloat16)
        rhos_v = tl.load(rhos_ptr + offs_n)

        for kc in range(num_k_chunks):
            k_start = kc * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K_max

            ek_off = b * K_max * H * hd + offs_k[:, None] * H * hd + h * hd + offs_d[None, :]
            e_k = tl.load(E_k + ek_off, mask=k_mask[:, None], other=0.0)
            e_v = tl.load(E_v + ek_off, mask=k_mask[:, None], other=0.0)

            uk = tl.load(unique_ptr + b * K_max + offs_k, mask=k_mask, other=-1)
            valid_k = (uk != -1) & k_mask

            c_off = (b * num_chunks * K_max * BLOCK_N
                     + pid_tc * K_max * BLOCK_N
                     + offs_k[:, None] * BLOCK_N + offs_n[None, :])
            c_local = tl.load(C_bounds + c_off, mask=k_mask[:, None], other=0.0)

            dE_k_acc = tl.zeros([BLOCK_K, hd], dtype=tl.float32)
            dE_v_acc = tl.zeros([BLOCK_K, hd], dtype=tl.float32)

            prev_doc = -1
            if has_doc_ids:
                prev_doc = tl.load(doc_ids_ptr + b * L + t_start, mask=t_start < L, other=-1)

            for t_step in range(BLOCK_M):
                t_global = t_start + t_step
                mt = t_global < L

                x_val = tl.load(x_labels + b * L + t_global, mask=mt, other=-1)
                match = ((x_val == uk) & mt).to(tl.float32)

                if has_doc_ids:
                    cur_doc = tl.load(doc_ids_ptr + b * L + t_global, mask=mt, other=-1)
                    boundary = (cur_doc != prev_doc) & (t_step > 0) & mt
                    keep = 1.0 - boundary.to(tl.float32)
                    c_local = c_local * keep
                    prev_doc = cur_doc

                c_decayed = c_local * rhos_v[None, :]
                c_decayed_tc = c_decayed.to(tl.bfloat16)
                pre_act = tl.dot(c_decayed_tc, w_pe_tc, out_dtype=tl.float32)

                if USE_SILU:
                    p_gate = pre_act * tl.sigmoid(pre_act)
                else:
                    p_gate = pre_act

                k_dyn = e_k * p_gate
                v_dyn = e_v * p_gate

                q_ptrs = Q + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd
                q_t = tl.load(q_ptrs, mask=mt, other=0.0)
                do_ptrs = dO + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd
                do_t = tl.load(do_ptrs, mask=mt, other=0.0)
                D_t = tl.load(D_vals + b * H * L + h * L + t_global, mask=mt, other=0.0)
                lse_t = tl.load(LSE + b * H * L + h * L + t_global, mask=mt, other=0.0)

                s_t = tl.sum(q_t[None, :] * k_dyn, axis=1) / math.sqrt(hd)
                valid = valid_k & mt
                s_t = tl.where(valid, s_t, float('-inf'))

                p_t = tl.exp(s_t - lse_t)
                p_t = tl.where(valid, p_t, 0.0)

                dp_t = tl.sum(do_t[None, :] * v_dyn, axis=1)
                ds_t = p_t * (dp_t - D_t)

                # dQ accumulation
                dq_t = tl.sum(ds_t[:, None] * k_dyn, axis=0) / math.sqrt(hd)
                dq_ptr = dQ_out + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd
                dq_old = tl.load(dq_ptr, mask=mt, other=0.0)
                tl.store(dq_ptr, dq_old + dq_t, mask=mt)

                # dE_k, dE_v accumulation (fused, no separate dk_dyn/dv_dyn storage)
                dE_k_acc += (ds_t[:, None] * q_t[None, :]) * (p_gate / math.sqrt(hd))
                dE_v_acc += (p_t[:, None] * do_t[None, :]) * p_gate

                c_local = c_decayed + match[:, None]

            # Write to partitioned buffer [B, H, K_max, hd] — only time chunks contend
            de_off = b * H * K_max * hd + h * K_max * hd + offs_k[:, None] * hd + offs_d[None, :]
            tl.atomic_add(dE_k_out + de_off, dE_k_acc, mask=k_mask[:, None])
            tl.atomic_add(dE_v_out + de_off, dE_v_acc, mask=k_mask[:, None])


    @triton.jit
    def _flash_ema_bwd_dw(
        Q, x_labels, E_k, E_v, W_pe, rhos_ptr, C_bounds, unique_ptr,
        dO, D_vals, LSE,
        dW_pe_out,
        doc_ids_ptr, has_doc_ids: tl.constexpr,
        stride_qb, stride_qh, stride_ql, stride_qd,
        B, H, L, hd: tl.constexpr, K_max, V_total,
        num_chunks, num_k_chunks,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr, USE_SILU: tl.constexpr,
    ):
        """Pass 2: Computes dW_pe only. Skips dQ, dE_k, dE_v."""
        pid_bh = tl.program_id(0)
        pid_tc = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        t_start = pid_tc * BLOCK_M
        offs_d = tl.arange(0, hd)
        offs_n = tl.arange(0, BLOCK_N)

        wpe_off = offs_n[:, None] * H * hd + h * hd + offs_d[None, :]
        w_pe = tl.load(W_pe + wpe_off)
        w_pe_tc = w_pe.to(tl.bfloat16)
        rhos_v = tl.load(rhos_ptr + offs_n)

        dW_pe_acc = tl.zeros([BLOCK_N, hd], dtype=tl.float32)

        for kc in range(num_k_chunks):
            k_start = kc * BLOCK_K
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K_max

            ek_off = b * K_max * H * hd + offs_k[:, None] * H * hd + h * hd + offs_d[None, :]
            e_k = tl.load(E_k + ek_off, mask=k_mask[:, None], other=0.0)
            e_v = tl.load(E_v + ek_off, mask=k_mask[:, None], other=0.0)

            uk = tl.load(unique_ptr + b * K_max + offs_k, mask=k_mask, other=-1)
            valid_k = (uk != -1) & k_mask

            c_off = (b * num_chunks * K_max * BLOCK_N
                     + pid_tc * K_max * BLOCK_N
                     + offs_k[:, None] * BLOCK_N + offs_n[None, :])
            c_local = tl.load(C_bounds + c_off, mask=k_mask[:, None], other=0.0)

            prev_doc = -1
            if has_doc_ids:
                prev_doc = tl.load(doc_ids_ptr + b * L + t_start, mask=t_start < L, other=-1)

            for t_step in range(BLOCK_M):
                t_global = t_start + t_step
                mt = t_global < L

                x_val = tl.load(x_labels + b * L + t_global, mask=mt, other=-1)
                match = ((x_val == uk) & mt).to(tl.float32)

                if has_doc_ids:
                    cur_doc = tl.load(doc_ids_ptr + b * L + t_global, mask=mt, other=-1)
                    boundary = (cur_doc != prev_doc) & (t_step > 0) & mt
                    keep = 1.0 - boundary.to(tl.float32)
                    c_local = c_local * keep
                    prev_doc = cur_doc

                c_decayed = c_local * rhos_v[None, :]
                c_decayed_tc = c_decayed.to(tl.bfloat16)
                pre_act = tl.dot(c_decayed_tc, w_pe_tc, out_dtype=tl.float32)

                if USE_SILU:
                    sig = tl.sigmoid(pre_act)
                    p_gate = pre_act * sig
                else:
                    sig = tl.zeros_like(pre_act)
                    p_gate = pre_act

                k_dyn = e_k * p_gate
                v_dyn = e_v * p_gate

                q_ptrs = Q + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd
                q_t = tl.load(q_ptrs, mask=mt, other=0.0)
                do_ptrs = dO + b * stride_qb + h * stride_qh + t_global * stride_ql + offs_d * stride_qd
                do_t = tl.load(do_ptrs, mask=mt, other=0.0)
                D_t = tl.load(D_vals + b * H * L + h * L + t_global, mask=mt, other=0.0)
                lse_t = tl.load(LSE + b * H * L + h * L + t_global, mask=mt, other=0.0)

                s_t = tl.sum(q_t[None, :] * k_dyn, axis=1) / math.sqrt(hd)
                valid = valid_k & mt
                s_t = tl.where(valid, s_t, float('-inf'))

                p_t = tl.exp(s_t - lse_t)
                p_t = tl.where(valid, p_t, 0.0)

                dp_t = tl.sum(do_t[None, :] * v_dyn, axis=1)
                ds_t = p_t * (dp_t - D_t)

                # Compute gradient chain → dW_pe
                dk_dyn = ds_t[:, None] * q_t[None, :] / math.sqrt(hd)
                dv_dyn = p_t[:, None] * do_t[None, :]
                dP_gate = dk_dyn * e_k + dv_dyn * e_v

                if USE_SILU:
                    gate_grad = sig * (1.0 + pre_act * (1.0 - sig))
                else:
                    gate_grad = 1.0
                dx = dP_gate * gate_grad

                dx_tc = dx.to(tl.bfloat16)
                c_tc = c_decayed.to(tl.bfloat16)
                dW_pe_acc += tl.dot(tl.trans(c_tc), dx_tc, out_dtype=tl.float32)

                c_local = c_decayed + match[:, None]

        # Write to per-block partition — zero atomic contention
        block_id = pid_bh * num_chunks + pid_tc
        dw_off = block_id * BLOCK_N * hd + offs_n[:, None] * hd + offs_d[None, :]
        tl.store(dW_pe_out + dw_off, dW_pe_acc)


# ─────────────────────────────────────────────────
# 4. Python Wrappers
# ─────────────────────────────────────────────────
def _build_unique_index(x, device):
    """
    Build compacted unique token tensor on GPU.
    Returns:
       unique_compacted: [B, K_max] — valid IDs packed to front, -1 padding at end
       K_max: actual max unique count across batch (NOT L)
    """
    B, L = x.shape

    # 1. Sort to align duplicates adjacently
    x_sorted, _ = torch.sort(x, dim=1)

    # 2. Mark first occurrences
    is_first = torch.ones(B, L, dtype=torch.bool, device=device)
    is_first[:, 1:] = x_sorted[:, 1:] != x_sorted[:, :-1]

    # 3. Count actual unique tokens per batch element
    unique_counts = is_first.sum(dim=1)  # [B]
    K_max = max(unique_counts.max().item(), 16)  # tl.dot requires BLOCK_K >= 16

    # 4. Compact: stable-sort on inverse mask pushes unique entries to front
    perm = (~is_first).long().argsort(dim=1, stable=True)
    # Take up to K_max columns (pad with -1 if fewer unique tokens)
    x_compacted = torch.full((B, K_max), -1, dtype=x.dtype, device=device)
    n_take = min(perm.shape[1], K_max)
    gathered = x_sorted.gather(1, perm)[:, :n_take]
    x_compacted[:, :n_take] = gathered

    # 5. Mark per-batch padding (batch elements with fewer uniques than K_max)
    pad_mask = torch.arange(K_max, device=device).unsqueeze(0) >= unique_counts.unsqueeze(1)
    x_compacted.masked_fill_(pad_mask, -1)

    return x_compacted, K_max


def _pad_n_dim(rhos, W_pe, C_bounds, c_init, BLOCK_N):
    """Pad the N dimension to BLOCK_N >= 16 for tl.dot tensor core minimum.
    Only pads tensors whose last dim is smaller than BLOCK_N (avoids double-padding)."""
    device = rhos.device
    if rhos.shape[0] < BLOCK_N:
        pad = BLOCK_N - rhos.shape[0]
        rhos = torch.cat([rhos, torch.zeros(pad, device=device, dtype=rhos.dtype)])
    if W_pe.shape[0] < BLOCK_N:
        pad = BLOCK_N - W_pe.shape[0]
        W_pe = torch.cat([W_pe, torch.zeros(pad, W_pe.shape[1], device=device, dtype=W_pe.dtype)])
    if C_bounds is not None and C_bounds.shape[-1] < BLOCK_N:
        pad = BLOCK_N - C_bounds.shape[-1]
        C_bounds = torch.cat([C_bounds, torch.zeros(*C_bounds.shape[:3], pad, device=device, dtype=C_bounds.dtype)], dim=-1)
    if c_init is not None and c_init.shape[-1] < BLOCK_N:
        pad = BLOCK_N - c_init.shape[-1]
        c_init = torch.cat([c_init, torch.zeros(*c_init.shape[:2], pad, device=device, dtype=c_init.dtype)], dim=-1)
    return rhos, W_pe, C_bounds, c_init


def flash_ema_forward(Q, x, E_k_active, E_v_active, W_pe, rhos, V_total, unique_tensor, K_max, C_bounds, use_silu=True, Br=64):
    B, H, L, hd = Q.shape
    device = Q.device

    E_k = E_k_active.view(B, K_max, H, hd).contiguous()
    E_v = E_v_active.view(B, K_max, H, hd).contiguous()

    O = torch.zeros_like(Q)
    M_val = torch.full((B, H, L), float('-inf'), device=device, dtype=torch.float32)
    L_val = torch.full((B, H, L), float(V_total - K_max), device=device, dtype=torch.float32)
    LSE = torch.empty((B, H, L), device=device, dtype=torch.float32)

    if not HAS_TRITON or device.type != 'cuda':
        raise NotImplementedError("triton_flash_ema natively requires Triton CUDA. Fallback logic should intercept!")

    num_chunks = max(1, triton.cdiv(L, Br))
    BLOCK_K = max(16, min(64, triton.next_power_of_2(K_max)))
    num_k_chunks = triton.cdiv(K_max, BLOCK_K)
    N = rhos.shape[0]
    BLOCK_N = max(N, 16)

    rhos_p, W_pe_p, C_bounds_p, _ = _pad_n_dim(rhos, W_pe.contiguous(), C_bounds, None, BLOCK_N)

    # doc_ids not needed: data layer guarantees single-document chunks
    doc_ids_dummy = torch.zeros(B, L, dtype=torch.int32, device=device)

    grid = (B * H, num_chunks)
    _flash_ema_fwd[grid](
        Q, x, E_k, E_v, W_pe_p, rhos_p, C_bounds_p, unique_tensor,
        O, M_val, L_val, LSE,
        doc_ids_dummy, False,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        B, H, L, hd, K_max, V_total,
        num_chunks, num_k_chunks,
        BLOCK_M=Br, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, USE_SILU=use_silu,
    )

    O = O / L_val.unsqueeze(-1)

    return O, LSE, C_bounds, E_k, E_v


class FlashEMAFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, Q, x, E_k_active, E_v_active, W_pe, rhos, V_total, unique_tensor, K_max, C_bounds, use_silu=True, Br=64):
        O, LSE, C_bounds, E_k, E_v = \
            flash_ema_forward(Q, x, E_k_active, E_v_active, W_pe, rhos, V_total, unique_tensor, K_max, C_bounds, use_silu, Br)
        ctx.save_for_backward(Q, x, E_k, E_v, W_pe, rhos, C_bounds, unique_tensor, O, LSE)
        ctx.K_max = K_max
        ctx.V_total = V_total
        ctx.use_silu = use_silu
        ctx.Br = Br
        return O

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dO):
        Q, x, E_k, E_v, W_pe, rhos, C_bounds, unique_tensor, O, LSE = ctx.saved_tensors
        B, H, L, hd = Q.shape
        K_max = ctx.K_max
        Br = ctx.Br
        N = rhos.shape[0]
        BLOCK_N = max(N, 16)

        if not HAS_TRITON or Q.device.type != 'cuda':
            raise NotImplementedError("triton_flash_ema natively requires Triton CUDA.")

        num_chunks = max(1, triton.cdiv(L, Br))
        BLOCK_K = max(16, min(64, triton.next_power_of_2(K_max)))
        num_k_chunks = triton.cdiv(K_max, BLOCK_K)

        dO = dO.contiguous()
        D_vals = (dO * O).sum(dim=-1).contiguous()

        rhos_p, W_pe_p, _, _ = _pad_n_dim(rhos, W_pe.contiguous(), None, None, BLOCK_N)

        dQ_out = torch.zeros_like(Q)

        # ── Partitioned gradient accumulation (eliminates atomic contention) ──
        # Instead of all (B*H*num_chunks) thread blocks atomic_add'ing to shared
        # dE_k/dE_v/dW_pe, each head writes to its own partition.
        # dE_k/dE_v: partition by H (num_chunks reduce within kernel per k-chunk)
        # dW_pe: partition by (B*H*num_chunks), reduce after kernel
        dE_k_partitioned = torch.zeros(B, H, K_max, hd, device=Q.device, dtype=Q.dtype)
        dE_v_partitioned = torch.zeros(B, H, K_max, hd, device=Q.device, dtype=Q.dtype)
        # dW_pe: each (b,h,tc) block gets its own slot → no atomics needed
        n_blocks = B * H * num_chunks
        dW_pe_partitioned = torch.zeros(n_blocks, BLOCK_N, W_pe.shape[1], device=Q.device, dtype=Q.dtype)

        # doc_ids not needed: data layer guarantees single-document chunks
        doc_ids_dummy = torch.zeros(B, L, dtype=torch.int32, device=Q.device)

        grid = (B * H, num_chunks)
        common_args = (
            Q, x, E_k, E_v, W_pe_p, rhos_p, C_bounds, unique_tensor,
        )
        common_kwargs = dict(
            doc_ids_ptr=doc_ids_dummy, has_doc_ids=False,
            stride_qb=Q.stride(0), stride_qh=Q.stride(1),
            stride_ql=Q.stride(2), stride_qd=Q.stride(3),
            B=B, H=H, L=L, hd=hd, K_max=K_max, V_total=ctx.V_total,
            num_chunks=num_chunks, num_k_chunks=num_k_chunks,
            BLOCK_M=Br, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, USE_SILU=ctx.use_silu,
        )

        # Pass 1: dQ + dE_k + dE_v (partitioned by head, atomic only across time chunks)
        _flash_ema_bwd_dq_de[grid](
            *common_args,
            dO, D_vals, LSE,
            dQ_out, dE_k_partitioned, dE_v_partitioned,
            **common_kwargs,
        )

        # Pass 2: dW_pe only — each block writes to its own partition, zero atomics
        _flash_ema_bwd_dw[grid](
            *common_args,
            dO, D_vals, LSE,
            dW_pe_partitioned,
            **common_kwargs,
        )

        # ── Reduce partitioned gradients ──
        # dE_k/dE_v: sum across heads → [B, K_max, H*hd]
        # The kernel writes to [B, H, K_max, hd] with atomic across time chunks only.
        # We need to reshape to match E_k's layout [B, K_max, H, hd] → [B, K_max, H*hd]
        dE_k_active = dE_k_partitioned.permute(0, 2, 1, 3).contiguous().view(B, K_max, H * hd)
        dE_v_active = dE_v_partitioned.permute(0, 2, 1, 3).contiguous().view(B, K_max, H * hd)

        # dW_pe: sum across all blocks → [BLOCK_N, hd]
        dW_pe_reduced = dW_pe_partitioned.sum(dim=0)
        dW_pe_final = dW_pe_reduced[:N, :]

        # Arguments: Q, x, E_k_active, E_v_active, W_pe, rhos, V_total, unique_tensor, K_max, C_bounds, use_silu, Br
        return dQ_out, None, dE_k_active, dE_v_active, dW_pe_final, None, None, None, None, None, None, None


class FlashEMAFunctionV2(torch.autograd.Function):
    """V2: cuBLAS gate projection in forward, old Triton kernels for backward."""
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, Q, x, E_k_active, E_v_active, W_pe, rhos, V_total, unique_tensor, K_max, c_all, C_bounds, use_silu=True, Br=64):
        O, LSE, P_gate, E_k, E_v = \
            flash_ema_forward_v2(Q, E_k_active, E_v_active, W_pe, V_total, unique_tensor, K_max, c_all, use_silu, Br)
        # Save tensors needed by backward (still uses old Triton kernels)
        ctx.save_for_backward(Q, x, E_k, E_v, W_pe, rhos, C_bounds, unique_tensor, O, LSE)
        ctx.K_max = K_max
        ctx.V_total = V_total
        ctx.use_silu = use_silu
        ctx.Br = Br
        return O

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dO):
        # Reuse the exact same backward as FlashEMAFunction
        Q, x, E_k, E_v, W_pe, rhos, C_bounds, unique_tensor, O, LSE = ctx.saved_tensors
        B, H, L, hd = Q.shape
        K_max = ctx.K_max
        Br = ctx.Br
        N = rhos.shape[0]
        BLOCK_N = max(N, 16)

        num_chunks = max(1, triton.cdiv(L, Br))
        BLOCK_K = max(16, min(64, triton.next_power_of_2(K_max)))
        num_k_chunks = triton.cdiv(K_max, BLOCK_K)

        dO = dO.contiguous()
        D_vals = (dO * O).sum(dim=-1).contiguous()

        rhos_p, W_pe_p, _, _ = _pad_n_dim(rhos, W_pe.contiguous(), None, None, BLOCK_N)

        dQ_out = torch.zeros_like(Q)
        dE_k_partitioned = torch.zeros(B, H, K_max, hd, device=Q.device, dtype=Q.dtype)
        dE_v_partitioned = torch.zeros(B, H, K_max, hd, device=Q.device, dtype=Q.dtype)
        n_blocks = B * H * num_chunks
        dW_pe_partitioned = torch.zeros(n_blocks, BLOCK_N, W_pe.shape[1], device=Q.device, dtype=Q.dtype)

        doc_ids_dummy = torch.zeros(B, L, dtype=torch.int32, device=Q.device)

        grid = (B * H, num_chunks)
        common_args = (Q, x, E_k, E_v, W_pe_p, rhos_p, C_bounds, unique_tensor)
        common_kwargs = dict(
            doc_ids_ptr=doc_ids_dummy, has_doc_ids=False,
            stride_qb=Q.stride(0), stride_qh=Q.stride(1),
            stride_ql=Q.stride(2), stride_qd=Q.stride(3),
            B=B, H=H, L=L, hd=hd, K_max=K_max, V_total=ctx.V_total,
            num_chunks=num_chunks, num_k_chunks=num_k_chunks,
            BLOCK_M=Br, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N, USE_SILU=ctx.use_silu,
        )

        _flash_ema_bwd_dq_de[grid](*common_args, dO, D_vals, LSE,
                                    dQ_out, dE_k_partitioned, dE_v_partitioned, **common_kwargs)
        _flash_ema_bwd_dw[grid](*common_args, dO, D_vals, LSE,
                                dW_pe_partitioned, **common_kwargs)

        dE_k_active = dE_k_partitioned.permute(0, 2, 1, 3).contiguous().view(B, K_max, H * hd)
        dE_v_active = dE_v_partitioned.permute(0, 2, 1, 3).contiguous().view(B, K_max, H * hd)
        dW_pe_reduced = dW_pe_partitioned.sum(dim=0)
        dW_pe_final = dW_pe_reduced[:N, :]

        # Arguments: Q, x, E_k_active, E_v_active, W_pe, rhos, V_total, unique_tensor, K_max, c_all, C_bounds, use_silu, Br
        return dQ_out, None, dE_k_active, dE_v_active, dW_pe_final, None, None, None, None, None, None, None, None
