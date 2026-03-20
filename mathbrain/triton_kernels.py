"""Triton 加速 kernels + autograd 集成

Forward 3 kernels:
  K1: fused_bilinear   — E_src gather + φ mul + scatter → ctx (Triton)
  K2: cuBLAS matmul    — ctx @ E_tgt.T → slot_scores
  K3: fused_word_proj  — slot_scores → word_logits (Triton sparse gather)

Backward 优化:
  K3_bwd: dense matmul d_logits @ wp_dense → d_scores (cuBLAS, 无 atomic)
  K2_bwd: cuBLAS (automatic)
  K1_bwd: atomic scatter (Triton, L2 cache 高效)
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    # ─── K1 Forward: Fused Bilinear ─────────────────────────

    @triton.jit
    def _bilinear_fwd_kernel(
        E_src_ptr, flat_active_ptr, flat_phi_ptr,
        batch_starts_ptr, batch_counts_ptr, ctx_ptr,
        D: tl.constexpr, dD: tl.constexpr,
        BLOCK_F: tl.constexpr, MAX_ACTIVE: tl.constexpr,
    ):
        bid = tl.program_id(0)
        fid = tl.program_id(1)
        feat_off = fid * BLOCK_F + tl.arange(0, BLOCK_F)
        feat_mask = feat_off < dD
        d_idx = feat_off % D
        start = tl.load(batch_starts_ptr + bid)
        count = tl.load(batch_counts_ptr + bid)
        acc = tl.zeros([BLOCK_F], dtype=tl.float32)
        for m in range(MAX_ACTIVE):
            if m < count:
                pos = start + m
                slot = tl.load(flat_active_ptr + pos)
                e = tl.load(E_src_ptr + slot * dD + feat_off,
                            mask=feat_mask, other=0.0)
                p = tl.load(flat_phi_ptr + pos * D + d_idx,
                            mask=feat_mask, other=0.0)
                acc += e * p
        if count > 0:
            acc = acc / count.to(tl.float32)
        tl.store(ctx_ptr + bid * dD + feat_off, acc, mask=feat_mask)

    # ─── K1 Backward: atomic scatter (L2 高效) ──────────────

    @triton.jit
    def _bilinear_bwd_kernel(
        d_ctx_ptr,          # (B, dD)
        flat_active_ptr, flat_phi_ptr,
        batch_starts_ptr, batch_counts_ptr,
        d_E_src_ptr,        # (S, dD)
        D: tl.constexpr, dD: tl.constexpr,
        BLOCK_F: tl.constexpr, MAX_ACTIVE: tl.constexpr,
    ):
        bid = tl.program_id(0)
        fid = tl.program_id(1)
        feat_off = fid * BLOCK_F + tl.arange(0, BLOCK_F)
        feat_mask = feat_off < dD
        d_idx = feat_off % D
        start = tl.load(batch_starts_ptr + bid)
        count = tl.load(batch_counts_ptr + bid)
        if count == 0:
            return
        d_ctx_val = tl.load(d_ctx_ptr + bid * dD + feat_off,
                            mask=feat_mask, other=0.0)
        d_ctx_val = d_ctx_val / count.to(tl.float32)
        for m in range(MAX_ACTIVE):
            if m < count:
                pos = start + m
                slot = tl.load(flat_active_ptr + pos)
                p = tl.load(flat_phi_ptr + pos * D + d_idx,
                            mask=feat_mask, other=0.0)
                grad = d_ctx_val * p
                tl.atomic_add(d_E_src_ptr + slot * dD + feat_off, grad,
                              mask=feat_mask)

    # ─── K3 Forward: Fused Word Projection ──────────────────

    @triton.jit
    def _word_proj_fwd_kernel(
        slot_scores_ptr, wp_offsets_ptr, wp_indices_ptr, wp_weights_ptr,
        out_ptr, S, V,
        MAX_SLOTS: tl.constexpr, BLOCK_W: tl.constexpr,
    ):
        bid = tl.program_id(0)
        wblock = tl.program_id(1)
        w_off = wblock * BLOCK_W + tl.arange(0, BLOCK_W)
        w_mask = w_off < V
        starts = tl.load(wp_offsets_ptr + w_off, mask=w_mask, other=0)
        ends = tl.load(wp_offsets_ptr + w_off + 1, mask=w_mask, other=0)
        counts = ends - starts
        acc = tl.zeros([BLOCK_W], dtype=tl.float32)
        for k in range(MAX_SLOTS):
            k_mask = (k < counts) & w_mask
            pos = starts + k
            idx = tl.load(wp_indices_ptr + pos, mask=k_mask, other=0)
            wt = tl.load(wp_weights_ptr + pos, mask=k_mask, other=0.0)
            score = tl.load(slot_scores_ptr + bid * S + idx,
                            mask=k_mask, other=0.0)
            acc += wt * score
        tl.store(out_ptr + bid * V + w_off, acc, mask=w_mask)


# =====================================================================
# Utility
# =====================================================================

def build_wp_dense(wp_offsets, wp_indices, wp_weights, S, V, device):
    """Build dense (V, S) matrix for matmul backward. One-time cost."""
    import numpy as np
    wp_off = wp_offsets.cpu().numpy()
    wp_idx = wp_indices.cpu().numpy()
    wp_wt = wp_weights.cpu().numpy()
    dense = np.zeros((V, S), dtype=np.float32)
    for w in range(V):
        lo, hi = wp_off[w], wp_off[w + 1]
        for k in range(lo, hi):
            dense[w, wp_idx[k]] = wp_wt[k]
    return torch.from_numpy(dense).to(device)


# =====================================================================
# Autograd Functions
# =====================================================================

class TritonBilinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E_src_weight, flat_active, flat_phi_normed,
                batch_starts, counts, B, dD, D, S):
        output = torch.zeros(B, dD, device=E_src_weight.device,
                             dtype=torch.float32)
        BLOCK_F = 128
        grid = (B, triton.cdiv(dD, BLOCK_F))
        _bilinear_fwd_kernel[grid](
            E_src_weight, flat_active, flat_phi_normed,
            batch_starts, counts, output,
            D=D, dD=dD, BLOCK_F=BLOCK_F, MAX_ACTIVE=1024,
        )
        ctx.save_for_backward(flat_active, flat_phi_normed,
                              batch_starts, counts)
        ctx.B = B; ctx.dD = dD; ctx.D = D; ctx.S = S
        return output

    @staticmethod
    def backward(ctx, d_output):
        flat_active, flat_phi_normed, batch_starts, counts = ctx.saved_tensors
        B, dD, D, S = ctx.B, ctx.dD, ctx.D, ctx.S
        d_E_src = torch.zeros(S, dD, device=d_output.device,
                              dtype=torch.float32)
        BLOCK_F = 128
        grid = (B, triton.cdiv(dD, BLOCK_F))
        _bilinear_bwd_kernel[grid](
            d_output, flat_active, flat_phi_normed,
            batch_starts, counts, d_E_src,
            D=D, dD=dD, BLOCK_F=BLOCK_F, MAX_ACTIVE=1024,
        )
        return d_E_src, None, None, None, None, None, None, None, None


class TritonWordProjFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slot_scores, wp_offsets, wp_indices, wp_weights,
                B, S, V, wp_dense):
        out = torch.zeros(B, V, device=slot_scores.device,
                          dtype=torch.float32)
        BLOCK_W = 128
        grid = (B, triton.cdiv(V, BLOCK_W))
        _word_proj_fwd_kernel[grid](
            slot_scores, wp_offsets, wp_indices, wp_weights, out,
            S, V, MAX_SLOTS=16, BLOCK_W=BLOCK_W,
        )
        ctx.save_for_backward(wp_dense)
        ctx.B = B; ctx.S = S; ctx.V = V
        return out

    @staticmethod
    def backward(ctx, d_logits):
        wp_dense, = ctx.saved_tensors
        # Dense matmul: (B, V) @ (V, S) → (B, S), 无 atomic
        d_scores = d_logits @ wp_dense
        return d_scores, None, None, None, None, None, None, None


# =====================================================================
# High-level API
# =====================================================================

def triton_bilinear(E_src_weight, flat_active, flat_phi_normed,
                    batch_starts, counts, B, dD, D, S):
    """Differentiable fused bilinear with Triton."""
    return TritonBilinearFn.apply(
        E_src_weight, flat_active, flat_phi_normed,
        batch_starts, counts, B, dD, D, S)


def triton_word_proj(slot_scores, wp_offsets, wp_indices, wp_weights,
                     B, S, V, wp_dense=None):
    """Forward: Triton sparse kernel. Backward: dense matmul (no atomic)."""
    if wp_dense is not None:
        return TritonWordProjFn.apply(
            slot_scores, wp_offsets, wp_indices, wp_weights,
            B, S, V, wp_dense)
    else:
        raise ValueError("wp_dense required — use build_wp_dense()")
