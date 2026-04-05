"""
Per-Slot EMA Throughput Profiler
================================
Run on GPU server to identify actual bottlenecks.

Usage:
    python scripts/profile_throughput.py --d_model 512 --n_layers 6    # current 46M config
    python scripts/profile_throughput.py --d_model 1024 --n_layers 24  # target 350M config

Outputs:
    1. Component-level timing breakdown (ms per step)
    2. Memory usage breakdown (MB)
    3. Throughput (tokens/sec)
    4. torch.profiler trace (viewable in chrome://tracing)
"""
import argparse
import torch
import torch.nn.functional as F
import time
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def create_model(args):
    from mathbrain.model import SlotTransformerLM
    model = SlotTransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        N=args.N,
        dropout=0.0,
        min_hl=1.0,
        max_hl=args.max_hl,
    ).cuda()
    return model


def create_mock_batch(args, model):
    """Create a realistic mock batch with expanded unique_slots (including history tokens)."""
    B = args.batch_size
    L = args.seq_len + 1  # +1 for target
    V = args.vocab_size
    N = args.N
    device = 'cuda'

    # Simulate realistic token distribution (zipf-like)
    tokens = torch.randint(0, V, (B, L), device=device)

    # Simulate expanded unique_slots: chunk tokens + history tokens
    chunk_unique_per_sample = int(L * 0.6)  # ~60% unique in chunk
    history_extra = int(chunk_unique_per_sample * 0.5)  # +50% from history
    K_max = chunk_unique_per_sample + history_extra

    unique_slots = torch.randint(0, V, (B, K_max), device=device)
    pad_mask = torch.ones(B, K_max, dtype=torch.bool, device=device)
    c_bases = torch.randn(B, K_max, N, device=device) * 0.1
    inverse_indices = torch.randint(0, chunk_unique_per_sample, (B, L), device=device)
    doc_ids = torch.zeros(B, L, dtype=torch.long, device=device)

    return tokens, unique_slots, pad_mask, inverse_indices, c_bases, doc_ids


def warmup(model, batch, args, n_warmup=5):
    """Warmup GPU and JIT compile Triton kernels."""
    tokens, unique_slots, pad_mask, inverse_indices, c_bases, doc_ids = batch
    x = tokens[:, :-1].contiguous()
    y = tokens[:, 1:].contiguous()
    inv_x = inverse_indices[:, :-1].contiguous()
    doc_x = doc_ids[:, :-1].contiguous()

    for _ in range(n_warmup):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, unique_slots=unique_slots, inverse_indices=inv_x,
                          c_base=c_bases, doc_ids=doc_x, pad_mask=pad_mask)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def benchmark_throughput(model, batch, args, n_steps=20):
    """Measure raw throughput (tokens/sec)."""
    tokens, unique_slots, pad_mask, inverse_indices, c_bases, doc_ids = batch
    x = tokens[:, :-1].contiguous()
    y = tokens[:, 1:].contiguous()
    inv_x = inverse_indices[:, :-1].contiguous()
    doc_x = doc_ids[:, :-1].contiguous()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(n_steps):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x, unique_slots=unique_slots, inverse_indices=inv_x,
                          c_base=c_bases, doc_ids=doc_x, pad_mask=pad_mask)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_step = args.batch_size * args.seq_len
    total_tokens = tokens_per_step * n_steps
    throughput = total_tokens / elapsed
    ms_per_step = (elapsed / n_steps) * 1000
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return throughput, ms_per_step, peak_mem


def component_timing(model, batch, args, n_steps=5):
    """Break down timing by component using CUDA events."""
    tokens, unique_slots, pad_mask, inverse_indices, c_bases, doc_ids = batch
    x = tokens[:, :-1].contiguous()
    y = tokens[:, 1:].contiguous()
    inv_x = inverse_indices[:, :-1].contiguous()
    doc_x = doc_ids[:, :-1].contiguous()

    from mathbrain.triton_ema_query import compute_query_ema_history
    from mathbrain.triton_flash_ema import _build_unique_index, precompute_boundaries, FlashEMAFunction
    from mathbrain.triton_fused_gating import fused_apply_symmetric_query_gating

    # We'll manually time the major stages
    timings = {
        'query_ema_scan': [],
        'boundary_precompute': [],
        'per_layer_forward': [],
        'loss_compute': [],
        'backward': [],
    }

    for _ in range(n_steps):
        model.zero_grad(set_to_none=True)

        # Stage 1: Query EMA scan
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            C_seq = compute_query_ema_history(x, model.ema_rhos, model.vocab_size,
                                              c_bases, unique_slots)
        torch.cuda.synchronize()
        timings['query_ema_scan'].append(time.perf_counter() - t0)

        # Stage 2: Unique index + boundary precompute
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        unique_tensor = unique_slots.clone()
        unique_tensor[~pad_mask] = -1
        K_max = pad_mask.sum(dim=1).max().item()
        K_max = max(K_max, 16)
        unique_tensor = unique_tensor[:, :K_max]
        c_base_kv = c_bases[:, :K_max, :]
        C_bounds = precompute_boundaries(x, unique_tensor, model.ema_rhos, 64, c_init=c_base_kv)
        torch.cuda.synchronize()
        timings['boundary_precompute'].append(time.perf_counter() - t0)

        # Stage 3: Layer forward passes
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            E_slots = model.embed.weight
            x_q = model.embed(x)
            for layer in model.layers:
                x_q = layer(x_q, x, E_slots, model.ema_rhos, C_seq,
                           unique_tensor, K_max, C_bounds)
            x_q = model.final_norm(x_q)
            logits = model.lm_head(x_q)
        torch.cuda.synchronize()
        timings['per_layer_forward'].append(time.perf_counter() - t0)

        # Stage 4: Loss
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
        torch.cuda.synchronize()
        timings['loss_compute'].append(time.perf_counter() - t0)

        # Stage 5: Backward
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        timings['backward'].append(time.perf_counter() - t0)

    # Average, skip first iteration (JIT) if enough samples
    result = {}
    for k, v in timings.items():
        vals = v[1:] if len(v) > 1 else v
        result[k] = sum(vals) / len(vals) * 1000  # ms
    return result


def run_torch_profiler(model, batch, args):
    """Generate torch.profiler trace for detailed kernel-level analysis."""
    tokens, unique_slots, pad_mask, inverse_indices, c_bases, doc_ids = batch
    x = tokens[:, :-1].contiguous()
    y = tokens[:, 1:].contiguous()
    inv_x = inverse_indices[:, :-1].contiguous()
    doc_x = doc_ids[:, :-1].contiguous()

    trace_dir = os.path.join(os.path.dirname(__file__), '..', 'profiler_traces')
    os.makedirs(trace_dir, exist_ok=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(6):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x, unique_slots=unique_slots, inverse_indices=inv_x,
                              c_base=c_bases, doc_ids=doc_x, pad_mask=pad_mask)
                loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            loss.backward()
            model.zero_grad(set_to_none=True)
            prof.step()

    # Print summary
    print("\n=== Top 20 CUDA Kernels by Time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print(f"\nFull trace saved to: {trace_dir}/")
    print("View with: tensorboard --logdir profiler_traces")
    return trace_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--max_hl', type=float, default=2048.0)
    parser.add_argument('--vocab_size', type=int, default=50304)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--skip_profiler', action='store_true', help='Skip torch.profiler trace')
    args = parser.parse_args()

    # Compute model size
    d_ff = 256 * ((int(8 * args.d_model / 3) + 255) // 256)
    embed_params = args.vocab_size * args.d_model
    layer_params = (
        4 * args.d_model * args.d_model +  # q,k,v,o proj
        args.N * args.d_model +  # pe_proj
        2 * args.d_model +  # ln_kv, ln_q (approx)
        3 * args.d_model * d_ff  # SwiGLU
    )
    total_params = embed_params + args.n_layers * layer_params + args.d_model  # +norm
    print(f"{'='*60}")
    print(f"Per-Slot EMA Throughput Profiler")
    print(f"{'='*60}")
    print(f"Model: d={args.d_model}, L={args.n_layers}, H={args.n_heads}, N={args.N}")
    print(f"Params: ~{total_params/1e6:.1f}M (embed={embed_params/1e6:.1f}M, per_layer={layer_params/1e6:.2f}M)")
    print(f"Batch: B={args.batch_size}, seq_len={args.seq_len}")
    print(f"Tokens/step: {args.batch_size * args.seq_len:,}")
    print(f"{'='*60}")

    print("\n[1/5] Creating model...")
    model = create_model(args)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  Actual params: {actual_params/1e6:.1f}M")

    print("\n[2/5] Creating mock batch...")
    batch = create_mock_batch(args, model)
    K_max_actual = batch[1].shape[1]
    print(f"  K_max (with history): {K_max_actual}")

    print("\n[3/5] Warmup ({} steps)...".format(5))
    warmup(model, batch, args)
    print("  Done.")

    print("\n[4/5] Throughput benchmark (20 steps)...")
    throughput, ms_per_step, peak_mem = benchmark_throughput(model, batch, args)
    print(f"  Throughput: {throughput:,.0f} tokens/sec")
    print(f"  Time/step:  {ms_per_step:.1f} ms")
    print(f"  Peak memory: {peak_mem:.0f} MB")

    # Estimate time for 15B tokens
    tokens_target = 15e9
    hours = tokens_target / throughput / 3600
    print(f"\n  Estimated time for 15B tokens: {hours:.1f} hours ({hours/24:.1f} days)")
    if hours > 168:
        print(f"  WARNING: exceeds 1-week budget ({168} hours)")
    elif hours > 84:
        print(f"  NOTE: fits in 1 week on 2 GPUs (with ~{168/hours:.1f}x margin)")
    else:
        print(f"  GOOD: fits in 1 week on 1 GPU")

    print("\n[5/5] Component timing breakdown...")
    timings = component_timing(model, batch, args)
    total_ms = sum(timings.values())
    print(f"\n  {'Component':<25} {'Time (ms)':>10} {'Fraction':>10}")
    print(f"  {'-'*45}")
    for k, v in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>10.2f} {v/total_ms*100:>9.1f}%")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<25} {total_ms:>10.2f}")

    # Identify bottleneck
    bottleneck = max(timings, key=timings.get)
    print(f"\n  BOTTLENECK: {bottleneck} ({timings[bottleneck]/total_ms*100:.1f}% of total)")

    if not args.skip_profiler:
        print("\n[BONUS] Running torch.profiler for kernel-level trace...")
        trace_dir = run_torch_profiler(model, batch, args)
    else:
        print("\n[BONUS] Skipped torch.profiler (use --skip_profiler to skip)")

    print(f"\n{'='*60}")
    print("DONE. Key numbers to report back:")
    print(f"  1. Throughput: {throughput:,.0f} tok/s")
    print(f"  2. ms/step: {ms_per_step:.1f}")
    print(f"  3. Peak mem: {peak_mem:.0f} MB")
    print(f"  4. Bottleneck: {bottleneck}")
    print(f"  5. 15B tokens ETA: {hours:.1f}h")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
