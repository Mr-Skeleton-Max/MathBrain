"""High-performance Triton-based EMA (Exponential Moving Average) Scanner"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True

    @triton.jit
    def _ema_combine(a_l, b_l, a_r, b_r):
        return a_l * a_r, a_r * b_l + b_r

    @triton.jit
    def ema_scan_kernel(
        X_ptr, Q_ptr, RHO_ptr,
        stride_xb, stride_xt, stride_xs,
        stride_qb, stride_qt, stride_qs, stride_qn,
        T, S, N,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid % N
        pid_sn = pid // N
        s = pid_sn % S
        b = pid_sn // S

        rho = tl.load(RHO_ptr + n)
        t_offs = tl.arange(0, BLOCK_T)
        mask = t_offs < T

        x_vals = tl.load(X_ptr + b * stride_xb + t_offs * stride_xt + s * stride_xs,
                         mask=mask, other=0.0)
        a_vals = tl.where(mask, rho, 1.0)
        _, q_vals = tl.associative_scan((a_vals, x_vals), 0, _ema_combine)
        tl.store(Q_ptr + b * stride_qb + t_offs * stride_qt + s * stride_qs + n * stride_qn,
                 q_vals, mask=mask)
except ImportError:
    HAS_TRITON = False

def compute_ema(x: torch.Tensor, rho: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """
    Compute EMA across the sequence using Triton parallel scan.
    
    Args:
        x: (B, T, S) float32 tensor of slot activations (1.0 if active, else 0.0)
        rho: (N,) float32 tensor of decay rates for different timescales
        chunk_size: Block size for parallel scan (must be power of 2 for optimal speed)
        
    Returns:
        Q: (B, T, S, N) float32 tensor of memory states.
    """
    assert x.is_contiguous(), "x must be contiguous"
    B, T, S = x.shape
    N = rho.shape[0]
    
    Q = torch.empty(B, T, S, N, device=x.device, dtype=torch.float32)

    if not HAS_TRITON or x.device.type != 'cuda':
        # CPU/MPS Fallback using simple loop
        q_prev = torch.zeros(B, S, N, device=x.device, dtype=torch.float32)
        rho_exp = rho.view(1, 1, N)
        for t in range(T):
            q_t = q_prev * rho_exp + x.select(1, t).unsqueeze(-1)
            Q[:, t, :, :] = q_t
            q_prev = q_t
        return Q

    if T <= chunk_size:
        BLOCK_T = triton.next_power_of_2(T)
        grid = (B * S * N,)
        ema_scan_kernel[grid](
            x, Q, rho,
            x.stride(0), x.stride(1), x.stride(2),
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            T, S, N, BLOCK_T=BLOCK_T,
        )
    else:
        # Fallback for extreme lengths: chunked with carry-over
        carry = torch.zeros(B, S, N, device=x.device, dtype=torch.float32)
        BLOCK_T = triton.next_power_of_2(chunk_size)
        grid = (B * S * N,)

        for t_start in range(0, T, chunk_size):
            t_end = min(t_start + chunk_size, T)
            chunk_len = t_end - t_start
            x_chunk = x[:, t_start:t_end].contiguous()
            Q_chunk = torch.empty(B, chunk_len, S, N, device=x.device, dtype=torch.float32)
            BT = triton.next_power_of_2(chunk_len)
            
            ema_scan_kernel[grid](
                x_chunk, Q_chunk, rho,
                x_chunk.stride(0), x_chunk.stride(1), x_chunk.stride(2),
                Q_chunk.stride(0), Q_chunk.stride(1), Q_chunk.stride(2), Q_chunk.stride(3),
                chunk_len, S, N, BLOCK_T=BT,
            )
            
            # Apply carry-over from previous block
            if t_start > 0:
                t_indices = torch.arange(1, chunk_len + 1, device=x.device, dtype=torch.float32)
                rho_powers = rho.unsqueeze(0).pow(t_indices.unsqueeze(1)) # (N, chunk_len) -> (N, T)
                # carry: (B, S, N) -> (B, 1, S, N)
                # rho_powers: (N, T) -> (1, T, 1, N)
                Q_chunk += carry.unsqueeze(1) * rho_powers.transpose(0, 1).unsqueeze(0).unsqueeze(2)
                
            Q[:, t_start:t_end] = Q_chunk
            carry = Q[:, t_end - 1, :, :].clone()

    return Q
