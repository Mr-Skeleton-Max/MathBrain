#!/usr/bin/env python3
"""Triton 全链路 forward benchmark

全部 Triton kernel:
  K1: fused_bilinear  — E_src lookup + φ mul + scatter → ctx
  K2: cuBLAS matmul   — ctx @ E_tgt.T → slot_scores (保留, 已最优)
  K3: fused_wp_ce     — word_proj + cross_entropy 融合: 寄存器内 gather+sum+softmax+nll

用法: python experiments/bench_triton.py --corpus datasets/tinystories_2000.txt
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MathBrain.config import MathBrainConfig
from MathBrain.retina import HashRetina
from MathBrain.phi_encoder import CosineChaosEncoder
from MathBrain.data import preprocess_corpus, WordProjection

import triton
import triton.language as tl


# =====================================================================
# K1: Fused Bilinear (E_src lookup + mul + scatter → ctx)
# =====================================================================

@triton.jit
def fused_bilinear_kernel(
    E_src_ptr, flat_active_ptr, flat_phi_ptr,
    batch_starts_ptr, batch_counts_ptr,
    ctx_ptr,
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
            e_vals = tl.load(E_src_ptr + slot * dD + feat_off,
                             mask=feat_mask, other=0.0)
            phi_vals = tl.load(flat_phi_ptr + pos * D + d_idx,
                               mask=feat_mask, other=0.0)
            acc += e_vals * phi_vals
    if count > 0:
        acc = acc / count.to(tl.float32)
    tl.store(ctx_ptr + bid * dD + feat_off, acc, mask=feat_mask)


# =====================================================================
# K2: Fused Word Projection (standalone, for comparison)
# =====================================================================

@triton.jit
def fused_word_proj_kernel(
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
        score = tl.load(slot_scores_ptr + bid * S + idx, mask=k_mask, other=0.0)
        acc += wt * score
    tl.store(out_ptr + bid * V + w_off, acc, mask=w_mask)


# =====================================================================
# K3: Fused Word Projection + Cross Entropy (寄存器内完成)
# 每个 program 处理一个 batch element — 遍历所有 V words 计算 logits,
# 同时在寄存器内做 softmax + nll, 只输出一个 scalar loss
# =====================================================================

@triton.jit
def fused_wp_ce_kernel(
    slot_scores_ptr,    # (B, S)
    wp_offsets_ptr,     # (V+1,)
    wp_indices_ptr,     # (nnz,)
    wp_weights_ptr,     # (nnz,)
    targets_ptr,        # (B,)
    loss_ptr,           # (B,) per-sample loss
    S, V,
    MAX_SLOTS: tl.constexpr,
    BLOCK_W: tl.constexpr,     # words per tile
):
    """一个 program = 一个 batch element, 全 V words 在寄存器内遍历"""
    bid = tl.program_id(0)
    target = tl.load(targets_ptr + bid)

    # Pass 1: 遍历所有 words, 算 logits, 找 max (for numerical stability)
    max_logit = tl.full([], float('-inf'), dtype=tl.float32)
    target_logit = tl.zeros([], dtype=tl.float32)

    n_blocks = tl.cdiv(V, BLOCK_W)
    for wb in range(n_blocks):
        w_off = wb * BLOCK_W + tl.arange(0, BLOCK_W)
        w_mask = w_off < V
        starts = tl.load(wp_offsets_ptr + w_off, mask=w_mask, other=0)
        ends = tl.load(wp_offsets_ptr + w_off + 1, mask=w_mask, other=0)
        counts = ends - starts

        logits = tl.zeros([BLOCK_W], dtype=tl.float32)
        for k in range(MAX_SLOTS):
            k_mask = (k < counts) & w_mask
            pos = starts + k
            idx = tl.load(wp_indices_ptr + pos, mask=k_mask, other=0)
            wt = tl.load(wp_weights_ptr + pos, mask=k_mask, other=0.0)
            score = tl.load(slot_scores_ptr + bid * S + idx,
                            mask=k_mask, other=0.0)
            logits += wt * score

        # 设置 non-word 位置为 -inf
        logits = tl.where(w_mask, logits, float('-inf'))
        block_max = tl.max(logits)
        max_logit = tl.maximum(max_logit, block_max)

        # 存 target logit
        is_target = (w_off == target) & w_mask
        target_logit += tl.sum(tl.where(is_target, logits, 0.0))

    # Pass 2: 遍历所有 words, 算 sum(exp(logit - max))
    sum_exp = tl.zeros([], dtype=tl.float32)
    for wb in range(n_blocks):
        w_off = wb * BLOCK_W + tl.arange(0, BLOCK_W)
        w_mask = w_off < V
        starts = tl.load(wp_offsets_ptr + w_off, mask=w_mask, other=0)
        ends = tl.load(wp_offsets_ptr + w_off + 1, mask=w_mask, other=0)
        counts = ends - starts

        logits = tl.zeros([BLOCK_W], dtype=tl.float32)
        for k in range(MAX_SLOTS):
            k_mask = (k < counts) & w_mask
            pos = starts + k
            idx = tl.load(wp_indices_ptr + pos, mask=k_mask, other=0)
            wt = tl.load(wp_weights_ptr + pos, mask=k_mask, other=0.0)
            score = tl.load(slot_scores_ptr + bid * S + idx,
                            mask=k_mask, other=0.0)
            logits += wt * score

        logits = tl.where(w_mask, logits - max_logit, float('-inf'))
        sum_exp += tl.sum(tl.exp(logits))

    # CE loss: -target_logit + max + log(sum_exp)
    loss = -(target_logit - max_logit) + tl.log(sum_exp)
    tl.store(loss_ptr + bid, loss)


# =====================================================================
# Python wrappers
# =====================================================================

def triton_bilinear(E_src_weight, flat_active, flat_phi_normed,
                    batch_starts, batch_counts, B, dD, D):
    ctx = torch.zeros(B, dD, device=flat_active.device, dtype=torch.float32)
    BLOCK_F = 128
    grid = (B, triton.cdiv(dD, BLOCK_F))
    fused_bilinear_kernel[grid](
        E_src_weight, flat_active, flat_phi_normed,
        batch_starts, batch_counts, ctx,
        D=D, dD=dD, BLOCK_F=BLOCK_F, MAX_ACTIVE=1024,
    )
    return ctx


def triton_word_proj(slot_scores, wp_offsets, wp_indices, wp_weights, B, S, V):
    out = torch.zeros(B, V, device=slot_scores.device, dtype=torch.float32)
    BLOCK_W = 128
    grid = (B, triton.cdiv(V, BLOCK_W))
    fused_word_proj_kernel[grid](
        slot_scores, wp_offsets, wp_indices, wp_weights, out,
        S, V, MAX_SLOTS=16, BLOCK_W=BLOCK_W,
    )
    return out


def triton_wp_ce(slot_scores, wp_offsets, wp_indices, wp_weights, targets,
                 B, S, V):
    loss = torch.zeros(B, device=slot_scores.device, dtype=torch.float32)
    grid = (B,)
    fused_wp_ce_kernel[grid](
        slot_scores, wp_offsets, wp_indices, wp_weights, targets, loss,
        S, V, MAX_SLOTS=16, BLOCK_W=128,
    )
    return loss.mean()


# =====================================================================
# Eager baselines
# =====================================================================

def eager_bilinear(E_src_weight, flat_active, flat_phi, batch_idx, counts,
                   B, dD, D):
    d = dD // D
    src = E_src_weight[flat_active]
    phi_n = flat_phi / flat_phi.norm(dim=1, keepdim=True).clamp(min=1e-8)
    weighted = (src.view(-1, d, D) * phi_n.unsqueeze(1)).view(-1, dD)
    ctx = torch.zeros(B, dD, device=flat_active.device)
    ctx.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, dD), weighted)
    ctx /= counts.float().unsqueeze(1).clamp(min=1.0)
    return ctx


# =====================================================================
# Main
# =====================================================================

def parse_rho(s):
    return tuple(float(x) for x in s.split(','))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='datasets/tinystories_2000.txt')
    parser.add_argument('--rho', type=parse_rho,
                        default='0.3,0.75,0.93,0.98,0.995,0.999,0.9995,0.9999')
    parser.add_argument('--D', type=int, default=32)
    parser.add_argument('--cp-rank', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda')
    N = len(args.rho)
    cfg = MathBrainConfig(N=N, RHO=args.rho, D_PHI=args.D, CP_RANK=args.cp_rank)

    corpus = [l.strip() for l in open(args.corpus) if l.strip()]
    print(f"Corpus: {len(corpus)} sentences, Device: {device}")

    retina = HashRetina(cfg)
    phi_enc = CosineChaosEncoder(cfg)
    result = preprocess_corpus(corpus, retina, phi_enc, cfg)

    S, V, D = result.S, result.V, result.D
    d = args.cp_rank
    dD = d * D
    B = args.batch_size
    n_pos = result.n_pos
    avg_active = int(result.offsets[-1]) // n_pos
    n_batches = max(1, (n_pos + B - 1) // B)

    print(f"S={S}, V={V}, D={D}, d={d}, dD={dD}")
    print(f"n_pos={n_pos:,}, avg_active={avg_active}, n_batches={n_batches}")
    print()

    # ── Data on GPU ──
    total_active = B * avg_active
    E_src_weight = torch.randn(S, dD, device=device) * 0.01
    E_tgt = torch.randn(S, dD, device=device) * 0.01
    flat_active = torch.randint(0, S, (total_active,), device=device)
    flat_phi = torch.randn(total_active, D, device=device)
    batch_idx = torch.arange(B, device=device).repeat_interleave(avg_active)
    counts = torch.full((B,), avg_active, dtype=torch.long, device=device)
    targets = torch.randint(0, V, (B,), device=device)

    flat_phi_normed = flat_phi / flat_phi.norm(dim=1, keepdim=True).clamp(min=1e-8)
    batch_starts = torch.arange(B, device=device, dtype=torch.int64) * avg_active

    wp = WordProjection(result, device)
    wp_offsets = torch.from_numpy(result.wp_word_offsets).to(device)
    wp_indices = torch.from_numpy(result.wp_slot_indices).to(device)
    wp_weights_t = torch.from_numpy(result.wp_slot_weights).to(device)

    N_ITER = 200

    # ═══════════════════════════════════════════════════════════
    # Per-component benchmarks
    # ═══════════════════════════════════════════════════════════

    # K1: Bilinear
    for _ in range(5):
        triton_bilinear(E_src_weight, flat_active, flat_phi_normed,
                        batch_starts, counts, B, dD, D)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        ctx = triton_bilinear(E_src_weight, flat_active, flat_phi_normed,
                              batch_starts, counts, B, dD, D)
    torch.cuda.synchronize()
    t_k1 = (time.time() - t0) / N_ITER * 1000

    # K2: Matmul
    slot_scores = ctx @ E_tgt.t()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        slot_scores = ctx @ E_tgt.t()
    torch.cuda.synchronize()
    t_k2 = (time.time() - t0) / N_ITER * 1000

    # K3a: WordProj alone
    for _ in range(5):
        triton_word_proj(slot_scores, wp_offsets, wp_indices,
                         wp_weights_t, B, S, V)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        logits = triton_word_proj(slot_scores, wp_offsets, wp_indices,
                                  wp_weights_t, B, S, V)
    torch.cuda.synchronize()
    t_k3a = (time.time() - t0) / N_ITER * 1000

    # CE alone
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        loss_ref = F.cross_entropy(logits, targets)
    torch.cuda.synchronize()
    t_ce = (time.time() - t0) / N_ITER * 1000

    # K3b: Fused WordProj+CE
    for _ in range(5):
        triton_wp_ce(slot_scores, wp_offsets, wp_indices,
                     wp_weights_t, targets, B, S, V)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        loss_tri = triton_wp_ce(slot_scores, wp_offsets, wp_indices,
                                wp_weights_t, targets, B, S, V)
    torch.cuda.synchronize()
    t_k3b = (time.time() - t0) / N_ITER * 1000

    # Correctness
    loss_ref_val = F.cross_entropy(logits, targets).item()
    loss_tri_val = triton_wp_ce(slot_scores, wp_offsets, wp_indices,
                                wp_weights_t, targets, B, S, V).item()
    ce_diff = abs(loss_ref_val - loss_tri_val)

    print(f"Per-component (Triton):")
    print(f"  K1 bilinear:        {t_k1:.3f}ms")
    print(f"  K2 matmul (cuBLAS): {t_k2:.3f}ms")
    print(f"  K3a word_proj:      {t_k3a:.3f}ms")
    print(f"  CE (PyTorch):       {t_ce:.3f}ms")
    print(f"  K3b wp+ce fused:    {t_k3b:.3f}ms  "
          f"(replaces K3a+CE = {t_k3a+t_ce:.3f}ms)")
    print(f"  CE diff:            {ce_diff:.6f}")
    print()

    # ═══════════════════════════════════════════════════════════
    # Full pipeline: 3 configs
    # ═══════════════════════════════════════════════════════════

    # Config A: All Eager
    for _ in range(5):
        eager_bilinear(E_src_weight, flat_active, flat_phi, batch_idx,
                       counts, B, dD, D)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        ctx = eager_bilinear(E_src_weight, flat_active, flat_phi, batch_idx,
                             counts, B, dD, D)
        scores = ctx @ E_tgt.t()
        logits = wp.forward(scores)
        loss = F.cross_entropy(logits, targets)
    torch.cuda.synchronize()
    t_eager = (time.time() - t0) / N_ITER * 1000

    # Config B: Triton K1 + cuBLAS + Triton K3a + PyTorch CE
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        ctx = triton_bilinear(E_src_weight, flat_active, flat_phi_normed,
                              batch_starts, counts, B, dD, D)
        scores = ctx @ E_tgt.t()
        logits = triton_word_proj(scores, wp_offsets, wp_indices,
                                  wp_weights_t, B, S, V)
        loss = F.cross_entropy(logits, targets)
    torch.cuda.synchronize()
    t_triton_b = (time.time() - t0) / N_ITER * 1000

    # Config C: Triton K1 + cuBLAS + Triton K3b (wp+ce fused)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N_ITER):
        ctx = triton_bilinear(E_src_weight, flat_active, flat_phi_normed,
                              batch_starts, counts, B, dD, D)
        scores = ctx @ E_tgt.t()
        loss = triton_wp_ce(scores, wp_offsets, wp_indices,
                            wp_weights_t, targets, B, S, V)
    torch.cuda.synchronize()
    t_triton_c = (time.time() - t0) / N_ITER * 1000

    print(f"{'='*65}")
    print(f"FULL FORWARD (per batch → per epoch):")
    print(f"  A) Eager:              {t_eager:.3f}ms → "
          f"{t_eager*n_batches/1000:.3f}s/epoch")
    print(f"  B) Triton+cuBLAS+CE:   {t_triton_b:.3f}ms → "
          f"{t_triton_b*n_batches/1000:.3f}s/epoch")
    print(f"  C) Triton+cuBLAS fused:{t_triton_c:.3f}ms → "
          f"{t_triton_c*n_batches/1000:.3f}s/epoch")
    print(f"  Speedup A→C: {t_eager/t_triton_c:.2f}x")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
