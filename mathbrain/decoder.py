import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import MathBrainConfig

        # 2. Positional Encoding (PE) from Q values (Linear mapping from raw Q)
        self.pe_proj = nn.Linear(config.N, config.d_model)

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
