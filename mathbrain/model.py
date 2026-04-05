import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flash_ema_attention import flash_ema_forward
from .triton_ema_query import compute_query_ema_history
from .triton_fused_gating import fused_apply_symmetric_query_gating
from .triton_flash_ema import FlashEMAFunction, FlashEMAFunctionSplitK, HAS_TRITON, _build_unique_index, precompute_boundaries  # noqa: F401

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps))

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SlotAttentionLayer(nn.Module):
    """
    Multiplicative EMA Cross-Attention (matching MultiplicativeTrueEMATransformer):
        P_gate = SiLU(pe_proj(C))
        memory = ln_kv(E_slots * P_gate)
        K = k_proj(memory), V = v_proj(memory)
        attn = softmax(Q @ K^T / sqrt(d)) @ V
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float, n_scales: int, use_silu: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Frequency gating
        self.pe_proj = nn.Linear(n_scales, d_model, bias=False)
        self.ln_kv = nn.LayerNorm(d_model)
        self.ln_q = nn.LayerNorm(d_model)
        self.resid_dropout = nn.Dropout(dropout)
        self.use_silu = use_silu

        import os
        self._use_splitk = os.environ.get('USE_V2', '0') == '1'

    def forward(self, x_q, x, E_slots, rhos, C_seq, unique_tensor=None, K_max=None, C_bounds=None):
        B, L, D = x_q.shape
        device = x.device

        # 1. Static Vocabulary Pre-projection
        if unique_tensor is not None:
            safe_unique = torch.clamp(unique_tensor, min=0)
            E_active = F.embedding(safe_unique, E_slots)
            memory_base = self.ln_kv(E_active)
            E_k_active = self.k_proj(memory_base)
            E_v_active = self.v_proj(memory_base)
        else:
            memory_base = self.ln_kv(E_slots)
            E_k_slots = self.k_proj(memory_base)
            E_v_slots = self.v_proj(memory_base)

        # 2. Fused Symmetric Query Gating
        Q_t = fused_apply_symmetric_query_gating(x_q, C_seq, self.pe_proj, self.ln_q, self.q_proj, self.use_silu)

        # 3. Late-Gating Cross-Attention
        if HAS_TRITON and device.type == 'cuda' and unique_tensor is not None:
            H = self.n_heads
            hd = D // H
            Q_mha = Q_t.view(B, L, H, hd).transpose(1, 2).contiguous()
            W_pe_t = self.pe_proj.weight.t().contiguous()

            if self._use_splitk:
                # Split-K: k-chunks run in parallel, then reduce
                out_mha = FlashEMAFunctionSplitK.apply(
                    Q_mha, x, E_k_active, E_v_active, W_pe_t, rhos, E_slots.shape[0],
                    unique_tensor, K_max, C_bounds, self.use_silu
                )
            else:
                # V1: original fused kernel
                out_mha = FlashEMAFunction.apply(
                    Q_mha, x, E_k_active, E_v_active, W_pe_t, rhos, E_slots.shape[0],
                    unique_tensor, K_max, C_bounds, self.use_silu
                )
            out = out_mha.transpose(1, 2).contiguous().view(B, L, D)
        else:
            # PyTorch fallback for macOS / correctness testing
            if unique_tensor is not None:
                memory_base = self.ln_kv(E_slots)
                E_k_slots = self.k_proj(memory_base)
                E_v_slots = self.v_proj(memory_base)

            out = flash_ema_forward(
                Q=Q_t, x=x, E_k_slots=E_k_slots, E_v_slots=E_v_slots,
                rhos=rhos, pe_proj=self.pe_proj, n_heads=self.n_heads, use_silu=self.use_silu
            )

        out = self.o_proj(out)
        return self.resid_dropout(out)


class SlotTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, n_scales, use_silu=True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = SlotAttentionLayer(d_model, n_heads, dropout, n_scales, use_silu)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x_q, x, E_slots, rhos, C_seq, unique_tensor=None, K_max=None, C_bounds=None):
        h = x_q + self.attn(self.attn_norm(x_q), x, E_slots, rhos, C_seq, unique_tensor, K_max, C_bounds)
        h = h + self.ffn(self.ffn_norm(h))
        return h


class SlotTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, N=16, dropout=0.1, use_silu=True, ema_dropout=0.0,
                 min_hl=1.0, max_hl=2048.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_scales = N
        self.ema_dropout = ema_dropout

        half_lives = torch.logspace(math.log10(min_hl), math.log10(max_hl), self.n_scales)
        rhos = torch.exp(-torch.log(torch.tensor(2.0)) / half_lives)
        self.register_buffer("ema_rhos", rhos)

        self.embed = nn.Embedding(vocab_size, d_model)

        d_ff = 256 * ((int(8 * d_model / 3) + 255) // 256)
        self.layers = nn.ModuleList([
            SlotTransformerBlock(d_model, n_heads, d_ff, dropout, self.n_scales, use_silu)
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, x, unique_slots=None, inverse_indices=None, c_base=None, doc_ids=None, pad_mask=None):
        B, L = x.shape
        device = x.device

        E_slots = self.embed.weight
        x_q = self.embed(x)

        unique_tensor, K_max, C_bounds = None, None, None
        
        # 1. Hoist Phase 1 History to run ONCE globally across all layers!
        # Wire the dataset's external chunk states seamlessly to avoid causal boundary breaks
        # Data layer guarantees: chunk is single-document, c_base is correct
        # → No doc_ids needed in Triton kernels (no cross-doc boundaries within chunk)
        C_seq = compute_query_ema_history(x, self.ema_rhos, self.vocab_size, c_base, unique_slots)

        if self.ema_dropout > 0 and self.training:
            C_seq = F.dropout(C_seq, p=self.ema_dropout, training=True)

        if HAS_TRITON and device.type == 'cuda':
            if unique_slots is not None and pad_mask is not None:
                # Use dataloader's expanded unique_slots directly (includes document history tokens).
                # This avoids the old _build_unique_index(x) which only found chunk-local tokens,
                # missing historical tokens with significant EMA state on slow decay scales.
                unique_tensor = unique_slots.clone()
                unique_tensor[~pad_mask] = -1  # Triton convention: -1 = padding
                K_max = pad_mask.sum(dim=1).max().item()
                K_max = max(K_max, 16)  # tl.dot minimum
                unique_tensor = unique_tensor[:, :K_max]

                # c_base is already aligned with unique_slots — no scatter/gather remapping needed
                c_base_kv = c_base[:, :K_max, :] if c_base is not None else None
            else:
                # Fallback: no dataloader metadata, rebuild from chunk (e.g., standalone inference)
                unique_tensor, K_max = _build_unique_index(x, device)
                c_base_kv = None

            C_bounds = precompute_boundaries(x, unique_tensor, self.ema_rhos, 64, c_init=c_base_kv)

        for layer in self.layers:
            x_q = layer(x_q, x, E_slots, self.ema_rhos, C_seq, unique_tensor, K_max, C_bounds)

        x_q = self.final_norm(x_q)
        return self.lm_head(x_q)
