#!/usr/bin/env python3
"""
profile_batch.py  —  MathBrain V4 single-batch breakdown profiler.

Usage (run on CUDA server):
    python profile_batch.py \
        --bpe-model tokenizers/bpe_8000.model \
        --vocab 8000 \
        --batch-size 32 \
        --seq-len 256

All timings are GPU-wall-clock (CUDA events).
V4: No Q tensor materialized. Register-resident EMA.
"""
import argparse
import torch
import torch.nn.functional as F

def cuda_time(fn, n_warmup=5, n_repeat=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sum(times)/len(times), min(times), max(times)

def fmt(m, lo, hi):
    return f"{m:8.2f} ms  (best {lo:.2f}, worst {hi:.2f})"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bpe-model',   default='tokenizers/bpe_8000.model')
    p.add_argument('--vocab',       type=int, default=8000)
    p.add_argument('--batch-size',  type=int, default=32)
    p.add_argument('--seq-len',     type=int, default=256)
    p.add_argument('--ema-chunk',   type=int, default=32)
    p.add_argument('--N',           type=int, default=64)
    p.add_argument('--d-model',     type=int, default=64)
    p.add_argument('--n-layers',    type=int, default=2)
    p.add_argument('--n-heads',     type=int, default=8,
                   help='Attention heads. head_dim = d_model / n_heads')
    p.add_argument('--d-ff',        type=int, default=256)
    p.add_argument('--h-min',       type=float, default=1)
    p.add_argument('--h-max',       type=float, default=1000)
    p.add_argument('--n-warmup',    type=int, default=5)
    p.add_argument('--n-repeat',    type=int, default=20)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, T, S, N = args.batch_size, args.seq_len, args.vocab, args.N
    C = args.ema_chunk

    peak_write_mb = (B*C*S + B*S*N + B*C*N) * 4 / 1e6  # Q_max + carry + q_query
    old_Q_mb = B * C * S * N * 4 / 1e6
    head_dim = args.d_model // args.n_heads
    print(f"\nDevice: {device}  |  B={B}  T={T}  S={S}  N={N}  ema_chunk={C}")
    print(f"Model: d_model={args.d_model}  n_layers={args.n_layers}  "
          f"n_heads={args.n_heads}  head_dim={head_dim}  d_ff={args.d_ff}")
    print(f"V4 write/chunk: {peak_write_mb:.0f} MB   (vs V3 Q tensor: {old_Q_mb:.0f} MB)")
    print()

    from mathbrain.config import MathBrainConfig
    from mathbrain.encoder import compute_ema_v4, recompute_active_q
    from mathbrain.decoder import SlotTransformer
    from mathbrain.trainer import extract_topk_indices

    cfg = MathBrainConfig.from_half_lives(
        h_min=args.h_min, h_max=args.h_max, vocab_size=S, retina_mode='bpe',
        N=N, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff)

    inputs  = torch.randint(0, S, (B, T), device=device)
    targets = torch.randint(0, S, (B, T), device=device)
    rho     = torch.tensor(cfg.rho_array, device=device)
    carry   = torch.zeros(B, S, N, device=device)

    decoder = SlotTransformer(cfg).to(device)
    opt     = torch.optim.AdamW(decoder.parameters(), lr=1e-3)

    # ── (1) one_hot ────────────────────────────────────────────────────
    sub = inputs[:, :C]
    def t1(): F.one_hot(sub, num_classes=S).float()
    m,lo,hi = cuda_time(t1, args.n_warmup, args.n_repeat)
    print(f"  [1] one_hot (tc={C})              {fmt(m,lo,hi)}")

    # ── (2) Phase 1: compute_ema_v4 ────────────────────────────────────
    x_sub = F.one_hot(sub, num_classes=S).float()
    def t2():
        with torch.no_grad():
            compute_ema_v4(x_sub, rho, sub, init_state=carry)
    m,lo,hi = cuda_time(t2, args.n_warmup, args.n_repeat)
    print(f"  [2] ema_v4 phase1 (tc={C})        {fmt(m,lo,hi)}")

    Qm_sub, qq_sub, _ = compute_ema_v4(x_sub, rho, sub, init_state=carry)
    Qm_flat = Qm_sub.reshape(B*C, S)

    # ── (3) topk ───────────────────────────────────────────────────────
    def t3():
        with torch.no_grad():
            extract_topk_indices(Qm_flat, cfg.eps_q, cfg.max_active_slots)
    m,lo,hi = cuda_time(t3, args.n_warmup, args.n_repeat)
    print(f"  [3] topk (tc={C})                 {fmt(m,lo,hi)}")

    si, pm = extract_topk_indices(Qm_flat, cfg.eps_q, cfg.max_active_slots)
    print(f"      → k={si.shape[1]}")

    # ── (4) Phase 3: recompute_active_q ────────────────────────────────
    def t4():
        with torch.no_grad():
            recompute_active_q(x_sub, rho, si, init_state=carry)
    m,lo,hi = cuda_time(t4, args.n_warmup, args.n_repeat)
    print(f"  [4] recompute_active_q (tc={C})   {fmt(m,lo,hi)}")

    # ── (5) full T EMA pipeline ────────────────────────────────────────
    iters = (T + C - 1) // C
    def collect_full():
        c = torch.zeros(B, S, N, device=device)
        with torch.no_grad():
            for t0 in range(0, T, C):
                t1_ = min(t0+C, T); tc = t1_-t0
                sub_in = inputs[:, t0:t1_]
                xs = F.one_hot(sub_in, num_classes=S).float()
                prev_c = c
                Qm, qq, c = compute_ema_v4(xs, rho, sub_in, init_state=c)
                c = c.detach()
                Qmf = Qm.reshape(B*tc, S)
                si_, pm_ = extract_topk_indices(Qmf, cfg.eps_q, cfg.max_active_slots)
                recompute_active_q(xs, rho, si_, init_state=prev_c)
                del Qm, xs, prev_c

    m_full, lo, hi = cuda_time(collect_full, args.n_warmup, args.n_repeat)
    print(f"\n  [5] full V4 EMA+extract (T={T}, {iters} iters) {fmt(m_full,lo,hi)}")
    print(f"      Per token: {m_full*1000/(B*T):.0f} µs")

    # ── (6) decoder fwd+bwd ───────────────────────────────────────────
    c_ = torch.zeros(B, S, N, device=device)
    l_Qa, l_si, l_pm, l_qq, l_iq = [], [], [], [], []
    with torch.no_grad():
        for t0 in range(0, T, C):
            t1_ = min(t0+C, T); tc = t1_-t0
            sub_in = inputs[:, t0:t1_]
            xs = F.one_hot(sub_in, num_classes=S).float()
            prev_c = c_
            Qm, qq, c_ = compute_ema_v4(xs, rho, sub_in, init_state=c_)
            c_ = c_.detach()
            Qmf = Qm.reshape(B*tc, S)
            si_, pm_ = extract_topk_indices(Qmf, cfg.eps_q, cfg.max_active_slots)
            Qa_ = recompute_active_q(xs, rho, si_, init_state=prev_c)
            iq_ = sub_in.reshape(B*tc)
            l_Qa.append(Qa_); l_si.append(si_); l_pm.append(pm_)
            l_qq.append(qq); l_iq.append(iq_)
            del Qm, xs, prev_c
    gmax = max(a.shape[1] for a in l_Qa)
    def pad2d(t, mx):
        d = mx - t.shape[1]
        return t if d == 0 else torch.cat([t, t.new_zeros(t.shape[0], d, *t.shape[2:])], dim=1)
    def padb(t, mx):
        d = mx - t.shape[1]
        return t if d == 0 else torch.cat([t, t.new_ones(t.shape[0], d, dtype=torch.bool)], dim=1)

    f_Qa = torch.cat([pad2d(a, gmax) for a in l_Qa])
    f_si = torch.cat([pad2d(s, gmax) for s in l_si])
    f_pm = torch.cat([padb(p, gmax)  for p in l_pm])
    f_qq = torch.cat(l_qq)
    f_iq = torch.cat(l_iq)
    y    = targets.reshape(B*T)

    decoder.train()
    def t6():
        opt.zero_grad(set_to_none=True)
        logits = decoder(f_Qa, f_si, f_pm, f_qq, f_iq.unsqueeze(1))
        F.cross_entropy(logits, y).backward()
        opt.step()
    m6, lo6, hi6 = cuda_time(t6, args.n_warmup, args.n_repeat)
    print(f"\n  [6] decoder fwd+bwd (BT={B*T})    {fmt(m6,lo6,hi6)}")

    print(f"\n  [TOTAL] estimated end-to-end     {m_full+m6:8.2f} ms")
    print(f"  tokens/sec ≈ {B*T/((m_full+m6)/1000):.0f}")
    print(f"  GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print()

if __name__ == '__main__':
    main()
