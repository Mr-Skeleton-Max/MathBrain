import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _fused_gating_fwd(
        x_q_ptr, p_pre_ptr, bias_ptr, ln_w_ptr, ln_b_ptr,
        x_ln_ptr, mu_ptr, rstd_ptr, x_emb_ptr, p_gate_ptr,
        stride_xb, stride_xl, stride_pb, stride_pl,
        D: tl.constexpr, BLOCK_D: tl.constexpr, USE_SILU: tl.constexpr
    ):
        pid = tl.program_id(0)
        
        x_q_base = x_q_ptr + pid * stride_xl
        p_pre_base = p_pre_ptr + pid * stride_pl
        x_ln_base = x_ln_ptr + pid * D
        x_emb_base = x_emb_ptr + pid * D
        p_gate_base = p_gate_ptr + pid * D
        
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        
        x = tl.load(x_q_base + offs, mask=mask, other=0.0)
        p_pre = tl.load(p_pre_base + offs, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        
        p_lin = p_pre + bias
        if USE_SILU:
            p_gate = p_lin * tl.sigmoid(p_lin)
        else:
            p_gate = p_lin
        tl.store(p_gate_base + offs, p_gate, mask=mask)
        
        x_emb = x * p_gate
        tl.store(x_emb_base + offs, x_emb, mask=mask)
        
        mean = tl.sum(x_emb, axis=0) / D
        x_diff = x_emb - mean
        var = tl.sum(x_diff * x_diff, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + 1e-5)
        
        tl.store(mu_ptr + pid, mean)
        tl.store(rstd_ptr + pid, rstd)
        
        x_hat = x_diff * rstd
        
        ln_w = tl.load(ln_w_ptr + offs, mask=mask, other=0.0)
        ln_b = tl.load(ln_b_ptr + offs, mask=mask, other=0.0)
        
        y = x_hat * ln_w + ln_b
        tl.store(x_ln_base + offs, y, mask=mask)


    @triton.jit
    def _fused_gating_bwd(
        dy_ptr, x_q_ptr, p_pre_ptr, bias_ptr, ln_w_ptr,
        x_emb_ptr, mu_ptr, rstd_ptr, p_gate_ptr,
        dx_q_ptr, dp_pre_ptr, dln_w_acc_ptr, dln_b_acc_ptr,
        stride_xb, stride_xl, stride_pb, stride_pl,
        D: tl.constexpr, BLOCK_D: tl.constexpr, USE_SILU: tl.constexpr
    ):
        pid = tl.program_id(0)
        
        dy_base = dy_ptr + pid * D
        x_q_base = x_q_ptr + pid * stride_xl
        p_pre_base = p_pre_ptr + pid * stride_pl
        x_emb_base = x_emb_ptr + pid * D
        p_gate_base = p_gate_ptr + pid * D
        
        dx_q_base = dx_q_ptr + pid * stride_xl
        dp_pre_base = dp_pre_ptr + pid * stride_pl
        dln_w_base = dln_w_acc_ptr + pid * D
        dln_b_base = dln_b_acc_ptr + pid * D
        
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        
        dy = tl.load(dy_base + offs, mask=mask, other=0.0)
        x = tl.load(x_q_base + offs, mask=mask, other=0.0)
        p_pre = tl.load(p_pre_base + offs, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + offs, mask=mask, other=0.0)
        ln_w = tl.load(ln_w_ptr + offs, mask=mask, other=0.0)
        x_emb = tl.load(x_emb_base + offs, mask=mask, other=0.0)
        p_gate = tl.load(p_gate_base + offs, mask=mask, other=0.0)
        
        mu = tl.load(mu_ptr + pid)
        rstd = tl.load(rstd_ptr + pid)
        
        x_hat = (x_emb - mu) * rstd
        dln_w = dy * x_hat
        dln_b = dy
        tl.store(dln_w_base + offs, dln_w, mask=mask)
        tl.store(dln_b_base + offs, dln_b, mask=mask)
        
        dx_hat = dy * ln_w
        c1 = tl.sum(dx_hat, axis=0) / D
        c2 = tl.sum(dx_hat * x_hat, axis=0) / D
        dx_emb = (dx_hat - c1 - x_hat * c2) * rstd
        
        dx_q = dx_emb * p_gate
        tl.store(dx_q_base + offs, dx_q, mask=mask)
        
        dp_gate = dx_emb * x
        
        p_lin = p_pre + bias
        if USE_SILU:
            sig = tl.sigmoid(p_lin)
            dsilu = p_gate + sig * (1.0 - p_gate)
        else:
            dsilu = 1.0
        
        dp_pre = dp_gate * dsilu
        tl.store(dp_pre_base + offs, dp_pre, mask=mask)


    class FusedGatingFunction(torch.autograd.Function):
        @staticmethod
        @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
        def forward(ctx, x_q, p_pre, bias, ln_w, ln_b, use_silu=True):
            B, L, D = x_q.shape
            x_q = x_q.contiguous()
            p_pre = p_pre.contiguous()
            
            x_ln = torch.empty((B, L, D), device=x_q.device, dtype=x_q.dtype)
            mu = torch.empty((B * L,), device=x_q.device, dtype=torch.float32)
            rstd = torch.empty((B * L,), device=x_q.device, dtype=torch.float32)
            x_emb = torch.empty((B, L, D), device=x_q.device, dtype=x_q.dtype)
            p_gate = torch.empty((B, L, D), device=x_q.device, dtype=x_q.dtype)
            
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B * L,)
            
            _fused_gating_fwd[grid](
                x_q, p_pre, bias, ln_w, ln_b,
                x_ln, mu, rstd, x_emb, p_gate,
                x_q.stride(0), x_q.stride(1), p_pre.stride(0), p_pre.stride(1),
                D, BLOCK_D=BLOCK_D, USE_SILU=use_silu
            )
            
            ctx.save_for_backward(x_q, p_pre, bias, ln_w, x_emb, mu, rstd, p_gate)
            ctx.use_silu = use_silu
            return x_ln
            
        @staticmethod
        @torch.amp.custom_bwd(device_type='cuda')
        def backward(ctx, dy):
            x_q, p_pre, bias, ln_w, x_emb, mu, rstd, p_gate = ctx.saved_tensors
            B, L, D = x_q.shape
            
            dy = dy.contiguous()
            dx_q = torch.empty_like(x_q)
            dp_pre = torch.empty_like(p_pre)
            dln_w_acc = torch.empty((B * L, D), device=x_q.device, dtype=x_q.dtype)
            dln_b_acc = torch.empty((B * L, D), device=x_q.device, dtype=x_q.dtype)
            
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B * L,)
            
            _fused_gating_bwd[grid](
                dy, x_q, p_pre, bias, ln_w,
                x_emb, mu, rstd, p_gate,
                dx_q, dp_pre, dln_w_acc, dln_b_acc,
                x_q.stride(0), x_q.stride(1), p_pre.stride(0), p_pre.stride(1),
                D, BLOCK_D=BLOCK_D, USE_SILU=ctx.use_silu
            )
            
            dln_w = dln_w_acc.sum(dim=0)
            dln_b = dln_b_acc.sum(dim=0)
            dbias = dp_pre.sum(dim=(0, 1))
            
            return dx_q, dp_pre, dbias, dln_w, dln_b, None

def fused_apply_symmetric_query_gating(x_q, C_seq, pe_proj, ln_q, q_proj, use_silu=True):
    bias_offset = pe_proj.weight.sum(dim=1)
    
    # Base projection: mathematically equals pe_proj(C_q_curr) without reallocating arrays
    P_pre = pe_proj(C_seq) 
    
    if HAS_TRITON and x_q.device.type == 'cuda':
        Q_ln = FusedGatingFunction.apply(x_q, P_pre, bias_offset, ln_q.weight, ln_q.bias, use_silu)
    else:
        import torch.nn.functional as F
        p_lin = P_pre + bias_offset
        if use_silu:
            P_gate = F.silu(p_lin)
        else:
            P_gate = p_lin
        Q_emb = x_q * P_gate
        Q_ln = ln_q(Q_emb)
        
    return q_proj(Q_ln)
