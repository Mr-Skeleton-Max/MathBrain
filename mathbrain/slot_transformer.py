"""Slot Transformer — 用 EMA Fourier PE 替换标准位置编码

架构:
  active_slots → E_src[s] + FourierPE(Q) → TransformerEncoder → mean pool → LM head → logits

关键设计:
  - 双向 self-attention (无 causal mask)
  - Fourier PE: sin/cos 多频率展开 EMA Q 值 (替代整数位置编码)
  - nn.TransformerEncoder: PyTorch 2.0+ → 自动 FlashAttention
  - LM head: LayerNorm → Linear(d_model, V)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════
# Fourier PE — 连续 EMA 值的位置编码
# ═══════════════════════════════════════════════════════════════

class FourierPE(nn.Module):
    """Fourier positional encoding for continuous EMA Q values.

    Q ∈ R^N → sin/cos expansion (N × 2K) → linear proj → R^d_model
    """

    def __init__(self, N_ema: int, K_freq: int, d_model: int):
        super().__init__()
        # Fixed frequencies: π, 2π, 4π, ..., 2^(K-1)π
        freqs = (2.0 ** torch.arange(K_freq).float()) * math.pi
        self.register_buffer('freqs', freqs)

        fourier_dim = N_ema * 2 * K_freq
        self.proj = nn.Linear(fourier_dim, d_model, bias=False)

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Q: (..., N_ema) → (..., d_model)"""
        shape = Q.shape[:-1]
        N = Q.shape[-1]

        x = Q.unsqueeze(-1) * self.freqs                  # (..., N, K)
        fe = torch.cat([x.sin(), x.cos()], dim=-1)        # (..., N, 2K)
        fe = fe.reshape(*shape, -1)                        # (..., N*2K)
        return self.proj(fe)                               # (..., d_model)


def flat_to_padded(flat_slots, flat_Q, pos_lo, counts, device,
                   flat_is_latest=None):
    """Convert flat batch to padded tensors — fully vectorized, no Triton.

    Args:
        flat_slots: (total_active,) int64
        flat_Q: (total_active, N) float32
        pos_lo: (B,) int64
        counts: (B,) int64
        flat_is_latest: (total_active,) bool — optional

    Returns:
        padded_slots: (B, max_k) int64
        padded_Q: (B, max_k, N) float32
        mask: (B, max_k) bool — True = padding (ignored by attention)
        is_latest: (B, max_k) bool — True = this slot was just activated
                   (only if flat_is_latest is provided)
    """
    B = counts.shape[0]
    N = flat_Q.shape[1]
    max_k = int(counts.max().item())

    # Build mask: (B, max_k), True = padding
    offsets = torch.arange(max_k, device=device).unsqueeze(0)  # (1, max_k)
    valid = offsets < counts.unsqueeze(1)                       # (B, max_k)
    mask = ~valid

    # Build gather indices: flat_idx[i,j] = pos_lo[i] + j, clamped
    flat_idx = pos_lo.unsqueeze(1) + offsets                    # (B, max_k)
    flat_idx = flat_idx.clamp(max=flat_slots.size(0) - 1)

    # Gather and zero out padding
    padded_slots = flat_slots[flat_idx] * valid.long()
    padded_Q = flat_Q[flat_idx] * valid.unsqueeze(-1).float()

    if flat_is_latest is not None:
        is_latest = flat_is_latest[flat_idx] & valid  # (B, max_k)
        return padded_slots, padded_Q, mask, is_latest

    return padded_slots, padded_Q, mask


# ═══════════════════════════════════════════════════════════════
# SlotTransformer — 主模块
# ═══════════════════════════════════════════════════════════════

class SlotTransformer(nn.Module):
    """Slot cross-attention decoder with EMA PE.

    Architecture (no encoder):
      1. All slots: embedding = E_src[slot_id] + PE(Q_values)
      2. K/V = non-latest slots' embeddings
      3. Q  = mean-pool of latest slot(s)' embeddings
      4. Cross-attention(Q, K, V) + residual + FFN
      5. LayerNorm → LM head → logits
    """

    def __init__(self, S: int, V: int, N_ema: int, *,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, d_ffn: int = 256,
                 K_freq: int = 8, dropout: float = 0.1,
                 pe_mode: str = 'fourier',
                 q_transform: str = 'none',
                 tie_weights: bool = False):
        super().__init__()
        self.d_model = d_model
        self.S = S
        self.V = V
        self.pe_mode = pe_mode
        self.q_transform = q_transform
        self.tie_weights = tie_weights

        # Slot identity embedding
        # When tie_weights: E_src doubles as output projection,
        # so it must cover V (full vocab) not just S (active slots)
        embed_size = max(S, V) if tie_weights else S
        self.E_src = nn.Embedding(embed_size, d_model)

        # Positional encoding for EMA Q values
        if pe_mode == 'fourier':
            self.pe = FourierPE(N_ema, K_freq, d_model)
        elif pe_mode == 'linear':
            self.pe = nn.Linear(N_ema, d_model, bias=False)
        else:
            raise ValueError(f"Unknown pe_mode: {pe_mode}")

        # Cross-attention layers (stacked num_layers times)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout, batch_first=True),
                'norm_q': nn.LayerNorm(d_model),
                'norm_kv': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_ffn),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ffn, d_model),
                    nn.Dropout(dropout),
                ),
            }))

        # LM head
        self.norm = nn.LayerNorm(d_model)
        if tie_weights:
            # Weight tying: reuse E_src as output projection
            # Output dim = S (slot logits); trainer handles slot→word if needed
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(d_model, V, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.E_src.weight, std=0.02)
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def _apply_q_transform(self, padded_Q, mask):
        if self.q_transform == 'log':
            padded_Q = torch.log(padded_Q.clamp(min=1e-8))
        elif self.q_transform == 'norm':
            valid_f = (~mask).unsqueeze(-1).float()
            q_masked = padded_Q * valid_f
            q_norm = q_masked.norm(dim=1, keepdim=True).clamp(min=1e-8)
            padded_Q = q_masked / q_norm
        return padded_Q

    def forward(self, padded_slots, padded_Q, mask, is_latest=None,
                return_ctx=False):
        """Forward pass.

        Args:
            padded_slots: (B, max_k) int64
            padded_Q: (B, max_k, N) float32
            mask: (B, max_k) bool — True = padding
            is_latest: (B, max_k) bool — True = latest word's slot(s).
                       If None, fallback to mean pool (no cross-attention).
            return_ctx: if True, also return context vectors
        """
        padded_Q = self._apply_q_transform(padded_Q, mask)

        # Step 1: All slots → embedding
        x = self.E_src(padded_slots) + self.pe(padded_Q)  # (B, max_k, d)

        if is_latest is not None:
            # Step 2: K/V mask = padding OR latest (exclude latest from K/V)
            kv_mask = mask | is_latest                         # (B, max_k)
            # Edge case: if ALL non-padding slots are latest (t=0),
            # allow latest to attend to itself
            all_masked = kv_mask.all(dim=1)                    # (B,)
            if all_masked.any():
                kv_mask = kv_mask.clone()
                kv_mask[all_masked] = mask[all_masked]

            # Step 3: Q = mean-pool latest slots → (B, 1, d)
            latest_f = is_latest.unsqueeze(-1).float()         # (B, max_k, 1)
            q = (x * latest_f).sum(dim=1, keepdim=True) \
                / latest_f.sum(dim=1, keepdim=True).clamp(min=1)

            # Step 4: Stacked cross-attention layers
            for layer in self.layers:
                attn_out, _ = layer['cross_attn'](
                    layer['norm_q'](q),
                    layer['norm_kv'](x),
                    x,
                    key_padding_mask=kv_mask,
                )
                q = q + attn_out                               # residual
                q = q + layer['ffn'](q)                        # FFN

            # Step 5: Decode
            ctx = q.squeeze(1)                                 # (B, d)
        else:
            # Fallback: mean pool all valid slots
            valid_f = (~mask).unsqueeze(-1).float()
            ctx = (x * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1)

        ctx = self.norm(ctx)
        if self.tie_weights:
            logits = ctx @ self.E_src.weight.T              # (B, S)
        else:
            logits = self.lm_head(ctx)                      # (B, V)

        if return_ctx:
            return logits, ctx
        return logits

