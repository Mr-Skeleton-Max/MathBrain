#!/usr/bin/env python3
"""profile_detailed.py — V5b fused pipeline (no one_hot, Q_max fused)."""
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
    for a,d in [('--vocab',8000),('--batch-size',32),('--seq-len',256),('--ema-chunk',32),
                ('--N',64),('--d-model',64),('--n-layers',2),('--n-heads',8),('--d-ff',256)]:
        p.add_argument(a, type=int, default=d)
    p.add_argument('--h-min', type=float, default=1); p.add_argument('--h-max', type=float, default=1000)
    p.add_argument('--n-warmup', type=int, default=3); p.add_argument('--n-repeat', type=int, default=10)
    args = p.parse_args()

    device = torch.device('cuda')
    B,T,S,N,C = args.batch_size, args.seq_len, args.vocab, args.N, args.ema_chunk
    W,R = args.n_warmup, args.n_repeat

    from mathbrain.config import MathBrainConfig
    from mathbrain.encoder import compute_ema_v5, recompute_slots
    from mathbrain.decoder import SlotTransformer
    from mathbrain.trainer import extract_topk_indices

    cfg = MathBrainConfig.from_half_lives(h_min=args.h_min, h_max=args.h_max, vocab_size=S,
        retina_mode='bpe', N=N, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff)
    hd = args.d_model // args.n_heads
    print(f"\nDevice: {device}  |  B={B} T={T} S={S} N={N} C={C}")
    print(f"Model: d={args.d_model} L={args.n_layers} H={args.n_heads} hd={hd} ff={args.d_ff}")
    print(f"TF32: enabled")

    inputs = torch.randint(0, S, (B, T), device=device)
    targets = torch.randint(0, S, (B, T), device=device)
    rho = torch.tensor(cfg.rho_array, device=device)

    # ═══ PART A: EMA sub-chunk breakdown ═══
    print(f"\n{'='*60}\n  PART A: V5b EMA Pipeline  (one sub-chunk, tc={C})\n{'='*60}")
    sub = inputs[:, :C].contiguous()
    carry = torch.zeros(B, S, N, device=device)
    Btc = B*C

    m,lo,hi = cuda_time(lambda: compute_ema_v5(sub, rho, S, init_state=carry), W, R)
    print(f"  [E1] fused kernel (carry+Qmax) {fmt(m,lo,hi)}")
    print(f"       NO one_hot. NO separate amax.")

    c_out, qm = compute_ema_v5(sub, rho, S, init_state=carry)
    m,lo,hi = cuda_time(lambda: extract_topk_indices(qm, cfg.eps_q, cfg.max_active_slots), W, R)
    si_b, pm_b = extract_topk_indices(qm, cfg.eps_q, cfg.max_active_slots)
    k = si_b.shape[1]
    print(f"  [E2] topk (B,S)=({B},{S}) k={k}  {fmt(m,lo,hi)}")

    si = si_b.unsqueeze(1).expand(B,C,k).reshape(Btc,k)
    iq = sub.reshape(Btc)
    mg = torch.cat([si, iq.unsqueeze(1)], dim=1)
    m,lo,hi = cuda_time(lambda: recompute_slots(sub, rho, S, mg, init_state=carry), W, R)
    print(f"  [E3] merged recompute(k+1)  {fmt(m,lo,hi)}")

    def full_sub():
        c,qm_ = compute_ema_v5(sub, rho, S, init_state=carry)
        sb,_ = extract_topk_indices(qm_, cfg.eps_q, cfg.max_active_slots)
        k_=sb.shape[1]
        se=sb.unsqueeze(1).expand(B,C,k_).reshape(Btc,k_)
        mg_=torch.cat([se, sub.reshape(Btc).unsqueeze(1)], dim=1)
        recompute_slots(sub, rho, S, mg_, init_state=carry)
    m,lo,hi = cuda_time(full_sub, W, R)
    print(f"  [E*] full sub-chunk         {fmt(m,lo,hi)}")

    # ═══ PART B: Full T pipeline ═══
    print(f"\n{'='*60}\n  PART B: Full V5b EMA Pipeline  (T={T}, {(T+C-1)//C} iters)\n{'='*60}")
    def full_ema():
        c = torch.zeros(B, S, N, device=device)
        with torch.no_grad():
            for t0 in range(0, T, C):
                t1=min(t0+C,T); tc=t1-t0
                si_=inputs[:, t0:t1].contiguous()
                prev=c; c,qm_=compute_ema_v5(si_, rho, S, init_state=c)
                c=c.detach()
                sb,_=extract_topk_indices(qm_, cfg.eps_q, cfg.max_active_slots)
                k_=sb.shape[1]
                se=sb.unsqueeze(1).expand(B,tc,k_).reshape(B*tc,k_)
                mg_=torch.cat([se, si_.reshape(B*tc).unsqueeze(1)], dim=1)
                recompute_slots(si_, rho, S, mg_, init_state=prev)
                del prev
    m_ema,lo,hi = cuda_time(full_ema, W, R)
    print(f"  [FULL] v5b ema pipeline    {fmt(m_ema,lo,hi)}")
    print(f"         per token: {m_ema*1000/(B*T):.0f} µs")

    # ═══ PART C: Decoder ═══
    c_ = torch.zeros(B, S, N, device=device)
    l_Qa, l_si, l_pm, l_qq, l_iq = [], [], [], [], []
    with torch.no_grad():
        for t0 in range(0,T,C):
            t1=min(t0+C,T); tc=t1-t0
            si_=inputs[:,t0:t1].contiguous()
            prev=c_; c_,qm_=compute_ema_v5(si_, rho, S, init_state=c_); c_=c_.detach()
            sb,pb_=extract_topk_indices(qm_,cfg.eps_q,cfg.max_active_slots)
            k_=sb.shape[1]
            se=sb.unsqueeze(1).expand(B,tc,k_).reshape(B*tc,k_)
            pm_e=pb_.unsqueeze(1).expand(B,tc,k_).reshape(B*tc,k_)
            iq_=si_.reshape(B*tc)
            mg_=torch.cat([se,iq_.unsqueeze(1)],dim=1)
            out=recompute_slots(si_,rho,S,mg_,init_state=prev)
            l_Qa.append(out[:,:k_]); l_si.append(se); l_pm.append(pm_e)
            l_qq.append(out[:,k_:]); l_iq.append(iq_)
    gmax=max(a.shape[1] for a in l_Qa)
    def p2d(t,mx):
        d=mx-t.shape[1]; return t if d==0 else torch.cat([t,t.new_zeros(t.shape[0],d,*t.shape[2:])],1)
    def pb(t,mx):
        d=mx-t.shape[1]; return t if d==0 else torch.cat([t,t.new_ones(t.shape[0],d,dtype=torch.bool)],1)
    f_Qa=torch.cat([p2d(a,gmax) for a in l_Qa]); f_si=torch.cat([p2d(s,gmax) for s in l_si])
    f_pm=torch.cat([pb(p,gmax) for p in l_pm]); f_qq=torch.cat(l_qq); f_iq=torch.cat(l_iq)
    y=targets.reshape(B*T)

    print(f"\n{'='*60}\n  PART C: Decoder  BT={B*T}, k={gmax}\n{'='*60}")
    decoder_e = SlotTransformer(cfg).to(device); opt_e = torch.optim.AdamW(decoder_e.parameters(), lr=1e-3)
    decoder_e.train()
    def eager_step():
        opt_e.zero_grad(set_to_none=True)
        logits=decoder_e(f_Qa,f_si,f_pm,f_qq,f_iq.unsqueeze(1))
        F.cross_entropy(logits,y).backward(); opt_e.step()
    m_eager,lo,hi = cuda_time(eager_step, W, R)
    print(f"  [eager]    fwd+bwd+step    {fmt(m_eager,lo,hi)}")

    decoder_c = SlotTransformer(cfg).to(device)
    decoder_c.load_state_dict(decoder_e.state_dict())
    opt_c = torch.optim.AdamW(decoder_c.parameters(), lr=1e-3)
    decoder_c.train(); decoder_c = torch.compile(decoder_c, mode='max-autotune')
    print("  [compiled] compiling...", end='', flush=True)
    for _ in range(5):
        opt_c.zero_grad(set_to_none=True)
        logits=decoder_c(f_Qa,f_si,f_pm,f_qq,f_iq.unsqueeze(1))
        F.cross_entropy(logits,y).backward(); opt_c.step()
    torch.cuda.synchronize(); print(" done")
    def compiled_step():
        opt_c.zero_grad(set_to_none=True)
        logits=decoder_c(f_Qa,f_si,f_pm,f_qq,f_iq.unsqueeze(1))
        F.cross_entropy(logits,y).backward(); opt_c.step()
    m_compiled,lo,hi = cuda_time(compiled_step, W, R)
    print(f"  [compiled] fwd+bwd+step    {fmt(m_compiled,lo,hi)}  ({m_eager/m_compiled:.2f}x)")

    # ═══ Summary ═══
    print(f"\n{'='*60}\n  SUMMARY (V5b fused + TF32)\n{'='*60}")
    for label, m_dec in [("eager", m_eager), ("compiled", m_compiled)]:
        total = m_ema + m_dec
        print(f"\n  --- {label} ---")
        print(f"  EMA pipeline:    {m_ema:8.2f} ms  ({m_ema/total*100:.0f}%)")
        print(f"  Decoder:         {m_dec:8.2f} ms  ({m_dec/total*100:.0f}%)")
        print(f"  End-to-end:      {total:8.2f} ms")
        print(f"  tokens/sec:      {B*T/(total/1000):.0f}")
    print(f"\n  GPU memory peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

if __name__=='__main__': main()
