#!/usr/bin/env python3
"""Profile: 动态 EMA batch — 直接调用 _batch_ema_and_filter"""
import sys, time
import torch
sys.path.insert(0, '/root/autodl-tmp')

from MathBrain.config import MathBrainConfig
from MathBrain.gpu_preprocessor import gpu_preprocess, _batch_ema_and_filter
from MathBrain.streaming_preprocessor import build_vocab
from MathBrain.retina import HashRetina

cfg = MathBrainConfig(
    N=8, RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
    D_PHI=32, CP_RANK=32)

corpus = [l.strip() for l in open('datasets/tinystories_1000.txt') if l.strip()][:1000]
retina = HashRetina(cfg)
vocab = build_vocab(corpus, retina)

device = torch.device('cuda')
data = gpu_preprocess(corpus, vocab, cfg, device, verbose=True)

# 取一个 batch 的句子
batch_sents = []
batch_pos = 0
for s in data.sentences:
    batch_sents.append(s)
    batch_pos += s.n_pos
    if batch_pos >= 4096:
        break

print(f"\nBatch: {len(batch_sents)} sentences, {batch_pos} positions")

# Warmup
for _ in range(3):
    _batch_ema_and_filter(batch_sents, data.rho, data.eps_q, device)
torch.cuda.synchronize()

# Time it
N_RUNS = 10
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_RUNS):
    b = _batch_ema_and_filter(batch_sents, data.rho, data.eps_q, device)
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
ms = elapsed / N_RUNS * 1000
print(f"_batch_ema_and_filter: {ms:.3f}ms")
print(f"  output: {b.flat_slots.shape[0]} active, {b.counts.shape[0]} positions")
