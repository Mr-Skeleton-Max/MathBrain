#!/usr/bin/env python3
"""Benchmark: Dynamic EMA + E_word fusion GPU Pipeline

优化总结:
  - CPU retina hash → per-sentence x_indicators → GPU
  - 每 batch 动态 GPU EMA + alive filter
  - Phi CUDA warp shuffle encoder
  - Triton fused bilinear (forward + atomic backward)
  - E_word = wp_dense @ E_tgt (消除 sparse word_proj)
  - Dense matmul backward for word_proj (消除 atomic_add)
"""
import sys, time, argparse
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MathBrain.config import MathBrainConfig
from MathBrain.retina import HashRetina
from MathBrain.streaming_preprocessor import build_vocab
from MathBrain.gpu_preprocessor import gpu_preprocess, iter_batches_gpu
from MathBrain.gpu_phi_encoder import GPUPhiEncoder
from MathBrain.triton_kernels import triton_bilinear, build_wp_dense


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='datasets/tinystories_1000.txt')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda')
    cfg = MathBrainConfig(
        N=8, RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
        D_PHI=32, CP_RANK=32)

    corpus = [l.strip() for l in open(args.corpus) if l.strip()]
    retina = HashRetina(cfg)
    print(f"Corpus: {len(corpus)} sentences")

    t0 = time.time()
    vocab = build_vocab(corpus, retina)
    print(f"[1] build_vocab: {time.time()-t0:.2f}s")

    t0 = time.time()
    data = gpu_preprocess(corpus, vocab, cfg, device)
    t_prep = time.time() - t0
    print(f"[2] preprocess: {t_prep:.2f}s")

    phi_gpu = GPUPhiEncoder(cfg, device='cuda')
    S, V, D = vocab.S, vocab.V, cfg.D_PHI
    dD = cfg.CP_RANK * D

    E_src = torch.nn.Embedding(S, dD).to(device)
    E_tgt = torch.nn.Parameter(torch.randn(S, dD, device=device) * 0.01)
    wp_off = torch.from_numpy(vocab.wp_word_offsets).to(device)
    wp_idx = torch.from_numpy(vocab.wp_slot_indices).to(device)
    wp_wt = torch.from_numpy(vocab.wp_slot_weights).to(device)
    wp_dense = build_wp_dense(wp_off, wp_idx, wp_wt, S, V, device)
    optimizer = torch.optim.AdamW([E_src.weight, E_tgt], lr=0.01)

    # Warmup
    for b in iter_batches_gpu(data, args.batch_size, device, shuffle=False):
        phi_n = phi_gpu.encode_normalized(b.flat_Q)
        Bi = b.counts.shape[0]
        optimizer.zero_grad(set_to_none=True)
        ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                              b.pos_lo, b.counts, Bi, dD, D, S)
        E_word = wp_dense @ E_tgt
        logits = ctx @ E_word.t()
        loss = F.cross_entropy(logits, b.targets)
        loss.backward()
        optimizer.step()
        break
    torch.cuda.synchronize()

    mem_mb = torch.cuda.max_memory_allocated() / 1e6
    print(f"\n  GPU memory: {mem_mb:.0f}MB")
    print(f"\n[3] Training: {data.total_pos:,} pos, "
          f"{len(data.sentences)} sentences")

    for epoch in range(args.epochs):
        torch.cuda.synchronize()
        t_epoch = time.time()
        epoch_loss = torch.tensor(0.0, device=device)
        n_b = 0

        for b in iter_batches_gpu(data, args.batch_size, device):
            phi_n = phi_gpu.encode_normalized(b.flat_Q)
            Bi = b.counts.shape[0]
            optimizer.zero_grad(set_to_none=True)
            ctx = triton_bilinear(E_src.weight, b.flat_slots, phi_n,
                                  b.pos_lo, b.counts, Bi, dD, D, S)
            E_word = wp_dense @ E_tgt
            logits = ctx @ E_word.t()
            loss = F.cross_entropy(logits, b.targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach()
            n_b += 1

        torch.cuda.synchronize()
        ms = (time.time() - t_epoch) * 1000
        avg_loss = (epoch_loss / n_b).item()
        print(f"  epoch {epoch}: loss={avg_loss:.4f}, {ms:.0f}ms "
              f"({ms/n_b:.1f}ms/batch, {n_b} batches)")

    print(f"\n{'='*60}")
    print(f"preprocess (one-time): {t_prep:.2f}s")
    print(f"GPU epoch:             ~{ms:.0f}ms")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
