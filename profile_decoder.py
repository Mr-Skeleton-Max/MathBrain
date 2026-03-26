#!/usr/bin/env python3
"""profile_decoder.py — Detailed decoder cost breakdown."""
import argparse, torch, torch.nn.functional as F, math

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
    p.add_argument('--n-warmup', type=int, default=3)
    p.add_argument('--n-repeat', type=int, default=10)
    args = p.parse_args()

    device = torch.device('cuda')
    B, T, S, N = args.batch_size, args.seq_len, args.vocab, args.N
    BT = B * T
    M = args.max_active
    D = args.d_model
    H = args.n_heads
    hd = D // H
    W, R = args.n_warmup, args.n_repeat

    from mathbrain.config import MathBrainConfig
    from mathbrain.decoder import SlotTransformer

    cfg = MathBrainConfig.from_half_lives(h_min=1, h_max=1000, vocab_size=S,
        retina_mode='bpe', N=N, d_model=D, n_layers=args.n_layers,
        n_heads=H, d_ff=args.d_ff)

    print(f"\nDevice: {device}  |  BT={BT} M={M} D={D} H={H} hd={hd} ff={args.d_ff} L={args.n_layers}")
    print(f"Warmup={W}  Repeat={R}")

    decoder = SlotTransformer(cfg).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3)

    # Dummy inputs matching production shapes
    q_active = torch.randn(BT, M, N, device=device)
    slot_indices = torch.randint(0, S, (BT, M), device=device)
    pad_mask = torch.zeros(BT, M, dtype=torch.bool, device=device)  # no padding (fixed k)
    q_query = torch.randn(BT, 1, N, device=device)
    idx_query = torch.randint(0, S, (BT, 1), device=device)
    targets = torch.randint(0, S, (BT,), device=device)

    # ═══ PART A: Forward breakdown ═══
    print(f"\n{'='*60}")
    print(f"  PART A: Forward Breakdown  (BT={BT}, M={M})")
    print(f"{'='*60}")

    # [F1] KV embeddings: slot_embed + pe_proj
    m,lo,hi = cuda_time(lambda: decoder.slot_embed(slot_indices) + decoder.pe_proj(q_active), W, R)
    print(f"  [F1] kv_emb + kv_pe         {fmt(m,lo,hi)}")

    m,lo,hi = cuda_time(lambda: decoder.slot_embed(slot_indices), W, R)
    print(f"       slot_embed alone       {fmt(m,lo,hi)}")
    m,lo,hi = cuda_time(lambda: decoder.pe_proj(q_active), W, R)
    print(f"       pe_proj alone          {fmt(m,lo,hi)}")

    # [F2] Query embeddings
    m,lo,hi = cuda_time(lambda: decoder.slot_embed(idx_query) + decoder.pe_proj(q_query), W, R)
    print(f"  [F2] q_emb + q_pe           {fmt(m,lo,hi)}")

    memory = decoder.slot_embed(slot_indices) + decoder.pe_proj(q_active)
    query = decoder.slot_embed(idx_query) + decoder.pe_proj(q_query)

    # [F3] Mask
    def make_mask():
        if pad_mask.any():
            fm = torch.zeros(BT, 1, M, device=device, dtype=query.dtype)
            fm.masked_fill_(pad_mask.unsqueeze(1), float('-inf'))
            return fm
        return None
    m,lo,hi = cuda_time(make_mask, W, R)
    print(f"  [F3] mask check             {fmt(m,lo,hi)}")
    float_mask = make_mask()
    print(f"       mask is None: {float_mask is None}")

    # Per-layer breakdown
    for li, layer in enumerate(decoder.layers):
        # [F4] Q,K,V projections
        def qkv_proj(layer=layer):
            y = layer.norm1(query)
            layer.q_proj(y).view(BT, 1, H, hd).transpose(1,2)
            layer.k_proj(memory).view(BT, M, H, hd).transpose(1,2)
            layer.v_proj(memory).view(BT, M, H, hd).transpose(1,2)
        m,lo,hi = cuda_time(qkv_proj, W, R)
        print(f"  [F4.{li}] L{li} norm+Q/K/V proj  {fmt(m,lo,hi)}")

        # [F5] Manual matmul attention
        y = layer.norm1(query)
        Q_ = layer.q_proj(y).view(BT, 1, H, hd).transpose(1,2)
        K_ = layer.k_proj(memory).view(BT, M, H, hd).transpose(1,2)
        V_ = layer.v_proj(memory).view(BT, M, H, hd).transpose(1,2)

        def attn_only():
            sc = torch.matmul(Q_, K_.transpose(-2,-1)) * (hd**-0.5)
            aw = F.softmax(sc, dim=-1)
            torch.matmul(aw, V_)
        m,lo,hi = cuda_time(attn_only, W, R)
        print(f"  [F5.{li}] L{li} matmul attn      {fmt(m,lo,hi)}")

        # Compare with SDPA (if available)
        def sdpa_no_mask():
            F.scaled_dot_product_attention(Q_, K_, V_)
        m2,lo2,hi2 = cuda_time(sdpa_no_mask, W, R)
        print(f"       SDPA(no mask)          {fmt(m2,lo2,hi2)}")

        if float_mask is not None:
            def sdpa_with_mask():
                m4 = float_mask.unsqueeze(1)
                F.scaled_dot_product_attention(Q_, K_, V_, attn_mask=m4)
            m3,lo3,hi3 = cuda_time(sdpa_with_mask, W, R)
            print(f"       SDPA(with mask)        {fmt(m3,lo3,hi3)}")

        # [F6] out_proj
        attn_out = torch.matmul(F.softmax(torch.matmul(Q_, K_.transpose(-2,-1)) * (hd**-0.5), dim=-1), V_)
        attn_out = attn_out.transpose(1,2).reshape(BT, 1, D)
        m,lo,hi = cuda_time(lambda: layer.out_proj(attn_out), W, R)
        print(f"  [F6.{li}] L{li} out_proj         {fmt(m,lo,hi)}")

        # [F7] FFN
        def ffn_only(layer=layer):
            y = layer.norm2(query)
            layer.ff2(F.gelu(layer.ff1(y)))
        m,lo,hi = cuda_time(ffn_only, W, R)
        print(f"  [F7.{li}] L{li} FFN              {fmt(m,lo,hi)}")

    # [F8] final_ln + output_proj
    m,lo,hi = cuda_time(lambda: decoder.output_proj(decoder.final_ln(query.squeeze(1))), W, R)
    print(f"  [F8] final_ln + out_proj    {fmt(m,lo,hi)}")

    # [F*] Full forward
    decoder.eval()
    def full_fwd():
        with torch.no_grad():
            decoder(q_active, slot_indices, pad_mask, q_query, idx_query)
    m,lo,hi = cuda_time(full_fwd, W, R)
    print(f"  [F*] full forward           {fmt(m,lo,hi)}")

    # ═══ PART B: Backward breakdown ═══
    print(f"\n{'='*60}")
    print(f"  PART B: Forward+Backward+Step")
    print(f"{'='*60}")

    decoder.train()
    def fwd_only():
        decoder(q_active, slot_indices, pad_mask, q_query, idx_query)
    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        logits = decoder(q_active, slot_indices, pad_mask, q_query, idx_query)
        F.cross_entropy(logits, targets).backward()
    def fwd_bwd_step():
        opt.zero_grad(set_to_none=True)
        logits = decoder(q_active, slot_indices, pad_mask, q_query, idx_query)
        F.cross_entropy(logits, targets).backward()
        opt.step()

    m1,_,_ = cuda_time(fwd_only, W, R)
    m2,_,_ = cuda_time(fwd_bwd, W, R)
    m3,lo,hi = cuda_time(fwd_bwd_step, W, R)
    print(f"  [B1] forward only           {fmt(m1)}")
    print(f"  [B2] forward + backward     {fmt(m2)}")
    print(f"  [B3] fwd + bwd + step       {fmt(m3, lo, hi)}")
    print(f"  [B4] backward only          {fmt(m2-m1)}")
    print(f"  [B5] optimizer.step         {fmt(m3-m2)}")
    print(f"       bwd/fwd ratio:         {(m2-m1)/m1:.2f}x")

    # ═══ PART C: M sensitivity ═══
    print(f"\n{'='*60}")
    print(f"  PART C: max_active Sensitivity")
    print(f"{'='*60}")

    for m_val in [16, 32, 64, 128]:
        qa = torch.randn(BT, m_val, N, device=device)
        si = torch.randint(0, S, (BT, m_val), device=device)
        pm = torch.zeros(BT, m_val, dtype=torch.bool, device=device)
        qq = torch.randn(BT, 1, N, device=device)
        iq = torch.randint(0, S, (BT, 1), device=device)
        def run(qa=qa, si=si, pm=pm, qq=qq, iq=iq):
            opt.zero_grad(set_to_none=True)
            logits = decoder(qa, si, pm, qq, iq)
            F.cross_entropy(logits, targets).backward()
            opt.step()
        m,lo,hi = cuda_time(run, W, R)
        print(f"  M={m_val:3d}  fwd+bwd+step = {fmt(m,lo,hi)}")

    print(f"\n  GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ═══ PART D: torch.compile comparison ═══
    print(f"\n{'='*60}")
    print(f"  PART D: torch.compile Comparison  (BT={BT}, M={M})")
    print(f"{'='*60}")

    # Baseline (eager)
    decoder_eager = SlotTransformer(cfg).to(device)
    decoder_eager.load_state_dict(decoder.state_dict())
    opt_eager = torch.optim.AdamW(decoder_eager.parameters(), lr=1e-3)
    decoder_eager.train()

    def eager_fwd_bwd_step():
        opt_eager.zero_grad(set_to_none=True)
        logits = decoder_eager(q_active, slot_indices, pad_mask, q_query, idx_query)
        F.cross_entropy(logits, targets).backward()
        opt_eager.step()
    m_eager, lo, hi = cuda_time(eager_fwd_bwd_step, W, R)
    print(f"  [eager] fwd+bwd+step        {fmt(m_eager,lo,hi)}")

    # torch.compile modes
    for mode in ['default', 'reduce-overhead', 'max-autotune']:
        try:
            decoder_c = SlotTransformer(cfg).to(device)
            decoder_c.load_state_dict(decoder.state_dict())
            opt_c = torch.optim.AdamW(decoder_c.parameters(), lr=1e-3)
            decoder_c.train()
            decoder_c = torch.compile(decoder_c, mode=mode)

            # Extra warmup for compilation
            print(f"  [{mode}] compiling...", end='', flush=True)
            for _ in range(5):
                opt_c.zero_grad(set_to_none=True)
                logits = decoder_c(q_active, slot_indices, pad_mask, q_query, idx_query)
                F.cross_entropy(logits, targets).backward()
                opt_c.step()
            torch.cuda.synchronize()
            print(" done")

            def compiled_step(d=decoder_c, o=opt_c):
                o.zero_grad(set_to_none=True)
                logits = d(q_active, slot_indices, pad_mask, q_query, idx_query)
                F.cross_entropy(logits, targets).backward()
                o.step()
            m_c, lo, hi = cuda_time(compiled_step, W, R)
            speedup = m_eager / m_c
            print(f"  [{mode}] fwd+bwd+step  {fmt(m_c,lo,hi)}  ({speedup:.2f}x)")
        except Exception as ex:
            print(f"  [{mode}] FAILED: {ex}")

    # Also test compiled forward-only
    print()
    decoder_eval = SlotTransformer(cfg).to(device)
    decoder_eval.load_state_dict(decoder.state_dict())
    decoder_eval.eval()

    def eager_fwd():
        with torch.no_grad():
            decoder_eval(q_active, slot_indices, pad_mask, q_query, idx_query)
    m_ef, lo, hi = cuda_time(eager_fwd, W, R)
    print(f"  [eager fwd-only]            {fmt(m_ef,lo,hi)}")

    try:
        decoder_cf = torch.compile(decoder_eval, mode='reduce-overhead')
        print(f"  [compiled fwd] compiling...", end='', flush=True)
        for _ in range(5):
            with torch.no_grad():
                decoder_cf(q_active, slot_indices, pad_mask, q_query, idx_query)
        torch.cuda.synchronize()
        print(" done")
        def compiled_fwd():
            with torch.no_grad():
                decoder_cf(q_active, slot_indices, pad_mask, q_query, idx_query)
        m_cf, lo, hi = cuda_time(compiled_fwd, W, R)
        print(f"  [compiled fwd-only]         {fmt(m_cf,lo,hi)}  ({m_ef/m_cf:.2f}x)")
    except Exception as ex:
        print(f"  [compiled fwd] FAILED: {ex}")

    print(f"\n  GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

if __name__ == '__main__':
    main()
