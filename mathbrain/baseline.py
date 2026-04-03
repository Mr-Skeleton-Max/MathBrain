import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import RMSNorm, SwiGLU
import math

# RoPE specific functions
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # xq, xk: (B, L, H, hd)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        B, L, D = x.shape
        H, hd = self.n_heads, self.head_dim

        Q = self.wq(x).view(B, L, H, hd)
        K = self.wk(x).view(B, L, H, hd)
        V = self.wv(x).view(B, L, H, hd)

        # Apply RoPE
        Q, K = apply_rotary_emb(Q, K, freqs_cis)

        # SDPA requires (B, H, L, hd)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # FlashAttention (is_causal=True automatically applies lower-triangular mask)
        out = F.scaled_dot_product_attention(
            Q, K, V, 
            dropout_p=self.resid_dropout.p if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.resid_dropout(self.wo(out))

class RoPETransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class RoPETransformerLM(nn.Module):
    """
    Standard Llama-3 style Transformer Baseline with Causal Self-Attention and RoPE.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_seq_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        d_ff = int(8 * d_model / 3)
        d_ff = 256 * ((d_ff + 255) // 256)
        
        self.layers = nn.ModuleList([
            RoPETransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight Tying
        self.lm_head.weight = self.embed.weight
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(self.d_model // n_heads, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor):
        B, L = x.shape
        
        x_emb = self.embed(x)
        
        # Slice freqs_cis for current sequence length
        freqs_cis = self.freqs_cis[:L]
        
        for layer in self.layers:
            x_emb = layer(x_emb, freqs_cis)
            
        x_emb = self.final_norm(x_emb)
        logits = self.lm_head(x_emb)
        
        return logits
