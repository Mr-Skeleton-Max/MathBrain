"""
Flash-EMA: Multiplicative Gating Cross-Attention
=================================================
Matching MultiplicativeTrueEMATransformer from true_ema_vs_rope.py:

    P_gate = SiLU(pe_proj(C))
    memory = LN(E_slots * P_gate)
    K = k_proj(memory), V = v_proj(memory)
    attn = softmax(Q @ K^T / sqrt(d)) @ V

Single loop over L, fully vectorized across B. PyTorch autograd handles backward.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def flash_ema_forward(Q, x, E_k_slots, E_v_slots, rhos, pe_proj, n_heads, use_silu=True):
    """
    Base PyTorch Reference Implementation of O(1) Memory Variant C EMA Gating.
    Calculates identical multiplicative modulations but relies on standard dense operations.
    """
    B, L, D = Q.shape
    V = E_k_slots.shape[0]
    device = Q.device
    N = rhos.shape[0]
    
    H = n_heads
    hd = D // H

    # Evaluate sequentially to avoid O(L \cdot V \cdot D) memory footprint
    O = torch.zeros(B, H, L, hd, device=device)
    
    # Internal baseline sequential buffer
    C_global = torch.zeros((B, V, N), device=device, dtype=torch.float32)

    # Pre-expand E_k and E_v to [B, V, H, hd] for broadcasting later
    E_k = E_k_slots.view(V, H, hd).unsqueeze(0).expand(B, -1, -1, -1)  # [B, V, H, hd]
    E_v = E_v_slots.view(V, H, hd).unsqueeze(0).expand(B, -1, -1, -1)

    arange_B = torch.arange(B, device=device)

    C = C_global.clone()

    for t in range(L):
        # 变体 C 因果老去
        C = C * rhos.unsqueeze(0).unsqueeze(0)    
        
        # 1. 产生 Memory Gating
        p_lin_k = pe_proj(C).view(B, V, H, hd)
        if use_silu:
            P_gate_K = F.silu(p_lin_k)
        else:
            P_gate_K = p_lin_k
        
        # Late-Gating 纯乘性融合
        K = (E_k * P_gate_K).transpose(1, 2)   # [B, H, V, hd]
        V_val = (E_v * P_gate_K).transpose(1, 2)

        # 2. 当前 Query (Q)
        curr_tok = x[:, t]  
        C_q_hist = C[arange_B, curr_tok, :]
        C_q_curr = C_q_hist + 1.0                
        
        p_lin_q = pe_proj(C_q_curr).view(B, H, hd)
        if use_silu:
            P_gate_Q = F.silu(p_lin_q)
        else:
            P_gate_Q = p_lin_q
        
        # Late-Gating Q
        Q_t = Q[:, t:t+1, :].view(B, 1, H, hd).transpose(1, 2)
        
        # 3. 计算对齐与最终输出 O_t
        attn = (Q_t @ K.transpose(-2, -1)) / math.sqrt(hd)         # [B, H, 1, V]
        attn = F.softmax(attn, dim=-1)           # [B, H, 1, V]
        
        O[:, :, t:t+1, :] = attn @ V_val
        
        # 更新 C (Variant C 严格落后一步更新)
        C[arange_B, curr_tok, :] += 1.0

    out = O.transpose(1, 2).contiguous().view(B, L, D)
    return out
