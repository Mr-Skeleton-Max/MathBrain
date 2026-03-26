import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MathBrainConfig


class SlotTransformerLayer(nn.Module):
    """Single cross-attention + FFN layer using F.scaled_dot_product_attention."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.d_model  = d_model

        # Cross-attention projections
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)

        # Norms (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, 
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        tgt:      (B, 1, D)
        memory:   (B, M, D)
        attn_mask:(B, 1, M) float additive mask (0 = attend, -inf = ignore)
        """
        B, S, D = tgt.shape
        _, M, _ = memory.shape
        H, hd   = self.n_heads, self.head_dim

        # Pre-norm cross-attention
        y = self.norm1(tgt)
        Q = self.q_proj(y).view(B, S, H, hd).transpose(1, 2)   # (B, H, S, hd)
        K = self.k_proj(memory).view(B, M, H, hd).transpose(1, 2)
        V = self.v_proj(memory).view(B, M, H, hd).transpose(1, 2)

        # Manual attention — optimized for Q_len=1
        # Avoids SDPA dispatch overhead and math-mode fallback from mask
        # scores: (B, H, 1, hd) @ (B, H, hd, M) → (B, H, 1, M)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (hd ** -0.5)
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)  # (B, 1, 1, M)
        attn_weights = F.softmax(scores, dim=-1)
        # output: (B, H, 1, M) @ (B, H, M, hd) → (B, H, 1, hd)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        tgt = tgt + self.drop(self.out_proj(attn_out))

        # Pre-norm FFN
        y = self.norm2(tgt)
        tgt = tgt + self.drop(self.ff2(F.gelu(self.ff1(y))))
        return tgt


class SlotTransformer(nn.Module):
    '''
    MathBrain Decoder: Cross-Attention from Current Token to Active Slots.
    Uses F.scaled_dot_product_attention for maximum GPU efficiency.
    '''
    def __init__(self, config: MathBrainConfig):
        super().__init__()
        self.config = config

        # 1. Slot-id embedding (E_src)
        self.slot_embed = nn.Embedding(config.vocab_size, config.d_model)

        # 2. PE from raw Q values
        self.pe_proj = nn.Linear(config.N, config.d_model)

        # 3. Transformer layers
        self.layers = nn.ModuleList([
            SlotTransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # 4. Output
        self.final_ln   = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.output_proj.weight = self.slot_embed.weight

    def forward(self, q_active: torch.Tensor, slot_indices: torch.Tensor, 
                pad_mask: torch.Tensor, q_query: torch.Tensor, 
                idx_query: torch.Tensor) -> torch.Tensor:
        '''
        q_active:    (BT, max_active, N)
        slot_indices:(BT, max_active)
        pad_mask:    (BT, max_active)  bool, True = padded/ignore
        q_query:     (BT, 1, N)
        idx_query:   (BT, 1)
        '''
        # Build key/value memory
        kv_emb = self.slot_embed(slot_indices)      # (BT, M, D)
        kv_pe  = self.pe_proj(q_active)             # (BT, M, D)
        memory = kv_emb + kv_pe                     # (BT, M, D)

        # Build query
        q_emb  = self.slot_embed(idx_query)         # (BT, 1, D)
        q_pe   = self.pe_proj(q_query)              # (BT, 1, D)
        query  = q_emb + q_pe                       # (BT, 1, D)

        # Convert bool mask → float additive mask only if padding exists
        # Passing None lets SDPA use FlashAttention; non-None forces slower math path
        if pad_mask.any():
            float_mask = torch.zeros(pad_mask.shape[0], 1, pad_mask.shape[1],
                                     device=pad_mask.device, dtype=query.dtype)
            float_mask.masked_fill_(pad_mask.unsqueeze(1), float('-inf'))
        else:
            float_mask = None

        # Run layers
        x = query
        for layer in self.layers:
            x = layer(x, memory, attn_mask=float_mask)

        x = self.final_ln(x.squeeze(1))             # (BT, D)
        return self.output_proj(x)                   # (BT, vocab)
