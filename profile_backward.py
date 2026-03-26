#!/usr/bin/env python3
"""profile_backward.py — Detailed backward cost breakdown using autograd profiler."""
import argparse, torch, torch.nn.functional as F

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def cuda_time(fn, W=3, R=10):
    for _ in range(W): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(R):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sum(ts)/len(ts), min(ts), max(ts)

def fmt(m,lo=None,hi=None):
    return f"{m:8.2f} ms  (best {lo:.2f}, worst {hi:.2f})" if lo else f"{m:8.2f} ms"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--vocab', type=int, default=8000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seq-len', type=int, default=256)
    p.add_argument('--N', type=int, default=64)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--d-ff', type=int, default=256)
    p.add_argument('--max-active', type=int, default=64)
    args = p.parse_args()

    device = torch.device('cuda')
    B, T, S, N = args.batch_size, args.seq_len, args.vocab, args.N
    BT = B * T; M = args.max_active; D = args.d_model; H = args.n_heads; hd = D // H
    W, R = 3, 10

    from mathbrain.config import MathBrainConfig
    from mathbrain.decoder import SlotTransformer

    cfg = MathBrainConfig.from_half_lives(h_min=1, h_max=1000, vocab_size=S,
        retina_mode='bpe', N=N, d_model=D, n_layers=args.n_layers,
        n_heads=H, d_ff=args.d_ff)

    print(f"\nDevice: {device}  |  BT={BT} M={M} D={D} H={H} hd={hd} L={args.n_layers}")
    print(f"TF32: enabled")

    decoder = SlotTransformer(cfg).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
    decoder.train()

    q_active = torch.randn(BT, M, N, device=device)
    slot_indices = torch.randint(0, S, (BT, M), device=device)
    pad_mask = torch.zeros(BT, M, dtype=torch.bool, device=device)
    q_query = torch.randn(BT, 1, N, device=device)
    idx_query = torch.randint(0, S, (BT, 1), device=device)
    targets = torch.randint(0, S, (BT,), device=device)

    # ═══ PART A: Overall fwd/bwd split ═══
    print(f"\n{'='*60}\n  PART A: Forward vs Backward\n{'='*60}")
    def fwd():
        return decoder(q_active, slot_indices, pad_mask, q_query, idx_query)
    def fwd_loss():
        logits = decoder(q_active, slot_indices, pad_mask, q_query, idx_query)
        return F.cross_entropy(logits, targets)
    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        loss = fwd_loss()
        loss.backward()
    def fwd_bwd_step():
        opt.zero_grad(set_to_none=True)
        loss = fwd_loss()
        loss.backward()
        opt.step()

    m_f,_,_ = cuda_time(fwd, W, R)
    m_fb,_,_ = cuda_time(fwd_bwd, W, R)
    m_fbs,lo,hi = cuda_time(fwd_bwd_step, W, R)
    print(f"  forward:            {fmt(m_f)}")
    print(f"  forward + backward: {fmt(m_fb)}")
    print(f"  fwd + bwd + step:   {fmt(m_fbs, lo, hi)}")
    print(f"  backward only:      {fmt(m_fb - m_f)}")
    print(f"  optimizer.step:     {fmt(m_fbs - m_fb)}")

    # ═══ PART B: Per-component backward via hooks ═══
    print(f"\n{'='*60}\n  PART B: Per-component Backward (via hooks)\n{'='*60}")

    # Measure backward of specific sub-graphs by detaching
    # Strategy: run partial forward, then backward, to isolate each component's bwd cost

    # [B1] output_proj backward: Linear(D→S) with BT=8192
    # This is the largest matmul in backward: grad_weight is (D, S) = (64, 8000)
    x_final = torch.randn(BT, D, device=device, requires_grad=True)
    m,lo,hi = cuda_time(lambda: (
        opt.zero_grad(set_to_none=True),
        F.cross_entropy(decoder.output_proj(decoder.final_ln(x_final)), targets).backward()
    )[-1], W, R)
    print(f"  [B1] final_ln + output_proj bwd  {fmt(m,lo,hi)}")
    print(f"       matmul: ({BT},{D})×({D},{S}) = grad_weight")

    # [B2] Single attention layer backward
    for li, layer in enumerate(decoder.layers):
        mem = torch.randn(BT, M, D, device=device, requires_grad=True)
        tgt = torch.randn(BT, 1, D, device=device, requires_grad=True)
        def attn_bwd(layer=layer, tgt=tgt, mem=mem):
            opt.zero_grad(set_to_none=True)
            out = layer(tgt, mem)
            out.sum().backward()
        m,lo,hi = cuda_time(attn_bwd, W, R)
        print(f"  [B2.{li}] layer{li} full bwd         {fmt(m,lo,hi)}")

        # Breakdown: Q/K/V proj backward
        y_norm = torch.randn(BT, 1, D, device=device, requires_grad=True)
        mem_d = torch.randn(BT, M, D, device=device, requires_grad=True)
        def qkv_bwd(layer=layer, y=y_norm, mem=mem_d):
            opt.zero_grad(set_to_none=True)
            Q = layer.q_proj(y)
            K = layer.k_proj(mem)
            V = layer.v_proj(mem)
            (Q.sum() + K.sum() + V.sum()).backward()
        m,lo,hi = cuda_time(qkv_bwd, W, R)
        print(f"       Q/K/V proj bwd              {fmt(m,lo,hi)}")

        # Breakdown: attention matmul backward
        Q_ = torch.randn(BT, H, 1, hd, device=device, requires_grad=True)
        K_ = torch.randn(BT, H, M, hd, device=device, requires_grad=True)
        V_ = torch.randn(BT, H, M, hd, device=device, requires_grad=True)
        def attn_math_bwd():
            sc = torch.matmul(Q_, K_.transpose(-2,-1)) * (hd**-0.5)
            aw = F.softmax(sc, dim=-1)
            out = torch.matmul(aw, V_)
            out.sum().backward()
        m,lo,hi = cuda_time(attn_math_bwd, W, R)
        print(f"       matmul attn bwd             {fmt(m,lo,hi)}")

        # FFN backward
        ff_in = torch.randn(BT, 1, D, device=device, requires_grad=True)
        def ffn_bwd(layer=layer, x=ff_in):
            opt.zero_grad(set_to_none=True)
            out = layer.ff2(F.gelu(layer.ff1(layer.norm2(x))))
            out.sum().backward()
        m,lo,hi = cuda_time(ffn_bwd, W, R)
        print(f"       FFN bwd                     {fmt(m,lo,hi)}")

    # [B3] Embedding backward
    si = torch.randint(0, S, (BT, M), device=device)
    def embed_bwd():
        opt.zero_grad(set_to_none=True)
        e = decoder.slot_embed(si)
        e.sum().backward()
    m,lo,hi = cuda_time(embed_bwd, W, R)
    print(f"  [B3] slot_embed bwd              {fmt(m,lo,hi)}")
    print(f"       BT×M={BT*M} lookups, grad scatter into ({S},{D})")

    # pe_proj backward
    qa = torch.randn(BT, M, N, device=device, requires_grad=True)
    def pe_bwd():
        opt.zero_grad(set_to_none=True)
        decoder.pe_proj(qa).sum().backward()
    m,lo,hi = cuda_time(pe_bwd, W, R)
    print(f"  [B4] pe_proj bwd                 {fmt(m,lo,hi)}")

    # ═══ PART C: torch.profiler trace ═══
    print(f"\n{'='*60}\n  PART C: Top CUDA kernels (torch.profiler)\n{'='*60}")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            logits = decoder(q_active, slot_indices, pad_mask, q_query, idx_query)
            F.cross_entropy(logits, targets).backward()
            opt.step()
    torch.cuda.synchronize()

    # Print top kernels by CUDA time
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

    print(f"\n  GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

if __name__ == '__main__':
    main()
