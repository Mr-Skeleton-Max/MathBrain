import torch
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _ema_seq_scan_kernel(
        x_ptr,                # [B, L]
        C_seq_ptr,            # [B, L, N]
        T_last_ptr,           # [B, V]
        C_last_ptr,           # [B, V, N]
        D_last_ptr,           # [B, V] — doc_id of last occurrence per vocab
        log_rhos_ptr,         # [N]
        doc_ids_ptr,          # [B, L]
        has_doc_ids: tl.constexpr,
        stride_xb, stride_xl,
        stride_csb, stride_csl, stride_csn,
        B, L, V,
        BLOCK_N: tl.constexpr
    ):
        """
        Scans the sequence to compute the exact EMA history of x_t just before time t.
        Uses per-vocab doc_id tracking for O(1) document boundary handling.
        """
        pid_b = tl.program_id(0)

        x_base = x_ptr + pid_b * stride_xb
        C_seq_base = C_seq_ptr + pid_b * stride_csb

        for t in range(L):
            x_t = tl.load(x_base + t * stride_xl)
            t_last_ptr_x = T_last_ptr + pid_b * V + x_t

            offs_n = tl.arange(0, BLOCK_N)
            c_last_ptrs = C_last_ptr + pid_b * V * BLOCK_N + x_t * BLOCK_N + offs_n

            t_last = tl.load(t_last_ptr_x)          # int32
            c_last = tl.load(c_last_ptrs)            # [N] floats
            log_rhos = tl.load(log_rhos_ptr + offs_n)

            decay_steps = (t - t_last).to(tl.float32)
            decay_factor = tl.exp(decay_steps * log_rhos)

            c_curr = c_last * decay_factor

            # Cross-document check: zero out if last occurrence was in a different document
            if has_doc_ids:
                cur_doc = tl.load(doc_ids_ptr + pid_b * L + t)
                d_last_ptr = D_last_ptr + pid_b * V + x_t
                d_last = tl.load(d_last_ptr)
                same_doc = (d_last == cur_doc) | (t_last < 0)  # t_last < 0 = never seen
                c_curr = c_curr * same_doc.to(tl.float32)
                tl.store(d_last_ptr, cur_doc)

            c_seq_ptrs = C_seq_base + t * stride_csl + offs_n * stride_csn
            tl.store(c_seq_ptrs, c_curr)

            c_new = c_curr + 1.0
            tl.store(t_last_ptr_x, t)
            tl.store(c_last_ptrs, c_new)


def compute_query_ema_history_pt(x: torch.Tensor, rhos: torch.Tensor, vocab_size: int, c_init=None, c_init_keys=None, doc_ids=None):
    """
    Fallback PyTorch implementation for non-Triton systems (macOS).
    Uses per-vocab doc_id tracking for O(1) document boundary handling.
    """
    B, L = x.shape
    N = rhos.shape[0]
    device = x.device

    C_seq = torch.zeros((B, L, N), dtype=torch.float32, device=device)

    T_last = torch.full((B, vocab_size + 1), -1000000.0, dtype=torch.float32, device=device)
    C_last = torch.zeros((B, vocab_size + 1, N), dtype=torch.float32, device=device)
    D_last = torch.full((B, vocab_size + 1), -1, dtype=torch.int32, device=device)

    if c_init is not None and c_init_keys is not None:
        safe_keys = torch.where(c_init_keys == -1, vocab_size, c_init_keys)
        T_last.scatter_(1, safe_keys, 0.0)
        safe_keys_exp = safe_keys.unsqueeze(-1).expand(-1, -1, N)
        C_last.scatter_(1, safe_keys_exp, c_init)
        if doc_ids is not None:
            start_doc = doc_ids[:, 0:1].to(dtype=D_last.dtype).expand_as(safe_keys)
            D_last.scatter_(1, safe_keys, start_doc)

    rhos_exp = rhos.unsqueeze(0)  # [1, N]
    batch_idx = torch.arange(B, device=device)

    for t in range(L):
        x_t = x[:, t]  # [B]
        t_last = T_last[batch_idx, x_t]  # [B]
        c_last = C_last[batch_idx, x_t, :]  # [B, N]

        decay_steps = (t - t_last).unsqueeze(1).float()  # [B, 1]
        decay_factor = rhos_exp ** decay_steps           # [B, N]
        c_curr = c_last * decay_factor  # [B, N]

        # Cross-document check: zero out if token's last occurrence was in a different doc
        if doc_ids is not None:
            cur_doc = doc_ids[:, t].to(torch.int32)  # [B]
            d_last = D_last[batch_idx, x_t]  # [B]
            same_doc = (d_last == cur_doc) | (t_last < 0)  # never seen → treat as same
            c_curr = c_curr * same_doc.unsqueeze(1).float()
            D_last[batch_idx, x_t] = cur_doc

        C_seq[:, t, :] = c_curr

        c_new = c_curr + 1.0
        T_last[batch_idx, x_t] = t
        C_last[batch_idx, x_t, :] = c_new

    return C_seq


def compute_query_ema_history(x: torch.Tensor, rhos: torch.Tensor, vocab_size: int, c_init=None, c_init_keys=None, doc_ids=None):
    """
    Precomputes the EMA history for the specific tokens present in the sequence.
    Resets EMA state at document boundaries when doc_ids is provided.
    """
    if not HAS_TRITON or x.device.type != 'cuda':
        return compute_query_ema_history_pt(x, rhos, vocab_size, c_init, c_init_keys, doc_ids)

    B, L = x.shape
    N = rhos.shape[0]
    device = x.device

    # Normalize doc_ids dtype early (int32 for Triton compatibility)
    has_doc_ids = doc_ids is not None
    if has_doc_ids:
        doc_ids = doc_ids.to(torch.int32).contiguous()
    else:
        doc_ids = torch.zeros(B, L, dtype=torch.int32, device=device)

    C_seq = torch.empty((B, L, N), dtype=torch.float32, device=device)

    T_last = torch.full((B, vocab_size + 1), -1000000, dtype=torch.int32, device=device)
    C_last = torch.zeros((B, vocab_size + 1, N), dtype=torch.float32, device=device)
    D_last = torch.full((B, vocab_size + 1), -1, dtype=torch.int32, device=device)

    if c_init is not None and c_init_keys is not None:
        safe_keys = torch.where(c_init_keys == -1, vocab_size, c_init_keys)
        T_last.scatter_(1, safe_keys, 0)
        safe_keys_exp = safe_keys.unsqueeze(-1).expand(-1, -1, N)
        C_last.scatter_(1, safe_keys_exp, c_init)
        if has_doc_ids:
            start_doc = doc_ids[:, 0:1].expand_as(safe_keys)
            D_last.scatter_(1, safe_keys, start_doc)

    log_rhos = torch.log(rhos).to(torch.float32).contiguous()

    BLOCK_N = triton.next_power_of_2(N)
    assert N <= BLOCK_N and BLOCK_N <= 128, "N must be <= 128 for efficient block size"

    grid = (B,)
    _ema_seq_scan_kernel[grid](
        x, C_seq, T_last, C_last, D_last, log_rhos, doc_ids,
        has_doc_ids,
        x.stride(0), x.stride(1),
        C_seq.stride(0), C_seq.stride(1), C_seq.stride(2),
        B, L, vocab_size + 1,
        BLOCK_N=BLOCK_N
    )

    return C_seq


