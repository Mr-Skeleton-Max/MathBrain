import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MathBrainConfig

class FourierEncoding(nn.Module):
    """Fourier Positional/Timescale encoding for raw Q values."""
    def __init__(self, n_timescales: int, out_dim: int, K: int = 8):
        super().__init__()
        self.K = K
        # Mapping from (N * 2K) to out_dim
        # We drop the raw scalar to force it to use frequency features (empirically better)
        in_features = n_timescales * K * 2
        self.proj = nn.Linear(in_features, out_dim)
        
    def forward(self, q_vals: torch.Tensor) -> torch.Tensor:
        """
        q_vals: (B, S, N)
        Returns: (B, S, out_dim)
        """
        B, S, N = q_vals.shape
        freqs = torch.arange(1, self.K + 1, device=q_vals.device, dtype=q_vals.dtype) * torch.pi
        # (B, S, N, 1) * (K,) -> (B, S, N, K)
        phases = q_vals.unsqueeze(-1) * freqs.view(1, 1, 1, self.K)
        sin_x = torch.sin(phases)
        cos_x = torch.cos(phases)
        # Concat along last dim: (B, S, N, 2K)
        features = torch.cat([sin_x, cos_x], dim=-1).view(B, S, N * self.K * 2)
        return self.proj(features)


class SlotTransformer(nn.Module):
    """
    MathBrain Decoder: Cross-Attention from Current Token to Active Slots.
    Treats the incoming token as Query, and the active EMA slots as Key/Values.
    """
    def __init__(self, config: MathBrainConfig):
        super().__init__()
        self.config = config
        
        # 1. Embeddings
        self.word_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # 2. Phi Encoder (maps raw N timescales to d_model)
        if config.phi_mode == 'linear':
            self.phi_encoder = nn.Linear(config.N, config.d_model)
        elif config.phi_mode == 'fourier':
            self.phi_encoder = FourierEncoding(config.N, config.d_model, K=8)
        else:
            self.phi_encoder = nn.Sequential(
                nn.Linear(config.N, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model)
            )

        # 3. Transformer Decoder Layers (Cross Attention only, no self-seq-attention)
        # We only attend from the current active Q vector to other active Q vectors. 
        # Actually in MathBrain, the slots ARE the sequence! 
        # So we run standard self-attention over the active slots, then pool.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # 4. Global pooling & Output mapping
        self.final_ln = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_weights:
            self.output_proj.weight = self.word_embed.weight

    def forward(self, q_active: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        q_active: (B, max_active, N)  - The EMA values for active slots
        pad_mask: (B, max_active)     - True for padding (ignored) slots, False for valid
        """
        # Encode Q to rich features (B, max_active, D)
        features = self.phi_encoder(q_active)
        
        # Self-attention over active slots
        # PyTorch TransformerEncoder expects src_key_padding_mask where True = ignore
        attended = self.transformer(features, src_key_padding_mask=pad_mask)
        
        # Global aggregation (mean pooling over valid active slots)
        # Inverse mask: True for valid slots
        valid_mask = (~pad_mask).unsqueeze(-1).to(attended.dtype)
        sum_features = (attended * valid_mask).sum(dim=1)
        valid_counts = valid_mask.sum(dim=1).clamp_min(1.0)
        pooled = sum_features / valid_counts
        
        # Final norm and linear
        x = self.final_ln(pooled)
        logits = self.output_proj(x)
        return logits
