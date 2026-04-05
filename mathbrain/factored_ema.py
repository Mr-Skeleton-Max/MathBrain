"""
Factored Linear EMA — O(N·hd²) per timestep instead of O(K·N·hd)
================================================================

When the gate is LINEAR (no SiLU), the distributive law allows:

    score[t,k] = Σ_n c[t,k,n] · (Q[t] ⊙ W_pe[n,:]) · E_k[k]

Defining state S_n[t] = Σ_k c[t,k,n] · E_k[k] ⊗ E_v[k], we get:

    S_n[t] = ρ_n · S_n[t-1] + E_k[x_t] ⊗ E_v[x_t]   ← O(hd²) update
    output[t] = Σ_n (Q[t] ⊙ W_pe[n,:])ᵀ @ S_n[t]      ← O(N·hd²) query

K_max disappears entirely from per-timestep cost.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def factored_ema_forward(Q, x, E_k_slots, E_v_slots, rhos, W_pe, n_heads,
                         c_base=None, unique_slots=None, pad_mask=None):
    """
    Factored linear EMA forward pass (PyTorch, autograd-compatible).

    Args:
        Q: [B, L, D] query (already projected by q_proj)
        x: [B, L] token ids
        E_k_slots: [V, D] key embeddings
        E_v_slots: [V, D] value embeddings
        rhos: [N] decay rates
        W_pe: [N, D] gate projection weights (pe_proj.weight.t())
        n_heads: int
        c_base: [B, K, N] EMA history for unique_slots tokens (optional, from data pipeline)
        unique_slots: [B, K] vocab IDs corresponding to c_base (optional)
        pad_mask: [B, K] bool mask for valid entries in unique_slots (optional)
    Returns:
        output: [B, L, D]
    """
    B, L, D = Q.shape
    N = rhos.shape[0]
    H = n_heads
    hd = D // H
    device = Q.device

    # Per-head views: [V, H, hd]
    E_k = E_k_slots.view(-1, H, hd)
    E_v = E_v_slots.view(-1, H, hd)
    W_pe_h = W_pe.view(N, H, hd)  # [N, H, hd]

    # Initialize state from c_base (full document history via data pipeline)
    if c_base is not None and unique_slots is not None:
        # S_init[b,h,n,:,:] = Σ_k c_base[b,k,n] · E_k[slot_k,h,:] ⊗ E_v[slot_k,h,:]
        K = c_base.shape[1]
        safe_slots = unique_slots.clamp(min=0).long()  # [B, K]
        ek = E_k[safe_slots.reshape(-1)].view(B, K, H, hd)  # [B, K, H, hd]
        ev = E_v[safe_slots.reshape(-1)].view(B, K, H, hd)  # [B, K, H, hd]
        outer = ek.unsqueeze(-1) * ev.unsqueeze(-2)  # [B, K, H, hd, hd]
        # c_base: [B, K, N] → weight each outer product per scale
        # Mask invalid entries
        if pad_mask is not None:
            c_base = c_base * pad_mask.unsqueeze(-1).float()
        # [B, K, 1, N, 1, 1] * [B, K, H, 1, hd, hd] → sum over K → [B, H, N, hd, hd]
        S = torch.einsum('bkn,bkhij->bhnij', c_base, outer)
        S = S.to(torch.float32)
    else:
        S = torch.zeros(B, H, N, hd, hd, device=device, dtype=torch.float32)

    rhos_5d = rhos.view(1, 1, N, 1, 1)

    O = torch.zeros(B, L, H, hd, device=device, dtype=Q.dtype)

    for t in range(L):
        # Variant C: decay all state first
        S = S * rhos_5d

        # Query: [B, H, hd]
        q_t = Q[:, t, :].view(B, H, hd)

        # Modulated query per scale: q_t ⊙ W_pe[n] → [B, H, N, hd]
        # q_t: [B, H, 1, hd], W_pe_h: [1, H, N, hd] → broadcast to [B, H, N, hd]
        q_mod = q_t.unsqueeze(2) * W_pe_h.permute(1, 0, 2).unsqueeze(0)  # [B, H, N, hd]

        # Output: Σ_n q_mod[n]ᵀ @ S_n = Σ_n [B,H,1,hd] @ [B,H,hd,hd] → [B,H,1,hd]
        # Batched: [B, H, N, 1, hd] @ [B, H, N, hd, hd] → [B, H, N, 1, hd]
        o_per_n = torch.matmul(q_mod.unsqueeze(3), S).squeeze(3)  # [B, H, N, hd]
        o_t = o_per_n.sum(dim=2)  # [B, H, hd]
        O[:, t] = o_t

        # State update (Variant C: after attention)
        tok = x[:, t]
        ek_t = E_k[tok]  # [B, H, hd]
        ev_t = E_v[tok]  # [B, H, hd]
        outer = ek_t.unsqueeze(-1) * ev_t.unsqueeze(-2)  # [B, H, hd, hd]
        S = S + outer.unsqueeze(2)  # broadcast across N scales

    return O.reshape(B, L, D)


class FactoredLinearEMALayer(nn.Module):
    """
    Drop-in replacement for SlotAttentionLayer when use_silu=False.
    Uses factored state matrix: O(N·hd²) per timestep, no K_max dependency.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float, n_scales: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_scales = n_scales

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.pe_proj = nn.Linear(n_scales, d_model, bias=False)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_q = nn.LayerNorm(d_model)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x_q, x, E_slots, rhos, C_seq,
                c_base=None, unique_slots=None, pad_mask=None, **kwargs):
        B, L, D = x_q.shape

        # 1. Static embeddings → K, V projections (shared, computed once)
        memory_base = self.ln_kv(E_slots)
        E_k_slots = self.k_proj(memory_base)  # [V, D]
        E_v_slots = self.v_proj(memory_base)  # [V, D]

        # 2. Query gating (LINEAR — no SiLU)
        C_q_curr = C_seq + 1.0
        P_gate_Q = self.pe_proj(C_q_curr)  # [B, L, D]
        Q_emb = x_q * P_gate_Q
        Q_t = self.q_proj(self.ln_q(Q_emb))

        # 3. Factored EMA: state matrix absorbs all vocab slots
        W_pe = self.pe_proj.weight.t()  # [N, D]
        out = factored_ema_forward(
            Q_t, x, E_k_slots, E_v_slots, rhos, W_pe, self.n_heads,
            c_base=c_base, unique_slots=unique_slots, pad_mask=pad_mask,
        )

        out = self.o_proj(out)
        return self.resid_dropout(out)
