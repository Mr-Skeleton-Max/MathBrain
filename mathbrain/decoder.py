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
    '''
    MathBrain Decoder: Cross-Attention from Current Token to Active Slots.
    Treats the incoming token as Query, and the active EMA slots as Key/Values.
    '''
    def __init__(self, config: MathBrainConfig):
        super().__init__()
        self.config = config
        
        # 1. Embeddings (E_src)
        self.slot_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # 2. Positional Encoding (PE) from Q values
        if config.phi_mode == 'linear':
            self.pe_proj = nn.Linear(config.N, config.d_model)
        elif config.phi_mode == 'fourier':
            self.pe_proj = FourierEncoding(config.N, config.d_model, K=8)
        else:
            self.pe_proj = nn.Sequential(
                nn.Linear(config.N, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model)
            )

        # 3. Transformer Decoder Layers (Cross Attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        
        # 4. Output mapping
        self.final_ln = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_weights:
            self.output_proj.weight = self.slot_embed.weight

    def forward(self, q_active: torch.Tensor, slot_indices: torch.Tensor, pad_mask: torch.Tensor, 
                q_query: torch.Tensor, idx_query: torch.Tensor) -> torch.Tensor:
        '''
        q_active: (B*T, max_active, N)
        slot_indices: (B*T, max_active)
        pad_mask: (B*T, max_active)
        q_query: (B*T, 1, N)
        idx_query: (B*T, 1)
        '''
        # Keys / Values: E_src[slot_i] + PE(Q_i)
        kv_emb = self.slot_embed(slot_indices) # (B*T, max_active, D)
        kv_pe = self.pe_proj(q_active)         # (B*T, max_active, D)
        memory = kv_emb + kv_pe
        
        # Query: E_src[latest_slot] + PE(Q_latest)
        q_emb = self.slot_embed(idx_query)     # (B*T, 1, D)
        q_pe = self.pe_proj(q_query)           # (B*T, 1, D)
        query = q_emb + q_pe
        
        # Cross-attention (tgt=query, memory=KV)
        attended = self.transformer(tgt=query, memory=memory, memory_key_padding_mask=pad_mask)
        
        # Output formulation
        x = self.final_ln(attended.squeeze(1)) # (B*T, D)
        logits = self.output_proj(x)
        return logits
