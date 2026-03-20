#!/usr/bin/env python3
"""Step 12b: Two-Layer 修复实验

修复 step12 的问题:
1. Soft attention → hard attention (top-1 winner word, count=1)
   避免 Q2 slot 爆炸 (307K entries → 正常范围)
2. 增加 residual 变体: Layer 2 同时看到实际词 + L1 预测
3. 自组织 L1: supervised warmup 4 rounds → 再切换 winner-take-all

Variants:
  A: L2 只看 L1 的 winner prediction (hard attention)
  B: L2 看实际词 + L1 的 winner prediction (residual)
  C: L2 只看实际词 + L1 的 top-3 soft (residual + soft)
"""

from __future__ import annotations
import sys, time, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.wake_dataset import WakeDataset
from mathbrain.wake_tree_trainer_mlx import TreeWakeTrainerMLX
from mathbrain.evaluator_mlx import DenseEvaluatorMLX
from mathbrain.inference_fast import SparseMatrixDecoder
from experiments.run_parallel_wake import load_corpus

try:
    import mlx.core as mx
except ImportError:
    raise RuntimeError("需要 MLX")

CORPUS_PATH = "datasets/tinystories_10.txt"
LAMBDA_B_ROUND = 0.05
ETA = 0.5
L1_ROUNDS = 64
L2_ROUNDS = 64


def tokenize(s):
    return WakeDataset._tokenize(s)


def build_fresh_model(corpus):
    model = MathBrain(MathBrainConfig())
    for s in corpus:
        for w in tokenize(s):
            model._ensure_word(w)
    return model


def get_decoder(model):
    return SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)


def decay_b_hash(model, decay_rate=0.05):
    n = model.b._n_entries
    if n > 0:
        model.b._vecs[:n] *= (1 - decay_rate)


def train_layer1_batch(model, corpus, rounds=64, lambda_b=0.05):
    """Batch train Layer 1 (standard edge-level wake)."""
    trainer = TreeWakeTrainerMLX(model)
    ds = trainer.build_dataset(corpus, verbose=False, shard_size=4)
    trainer.run_corpus_parallel(ds, rounds=16, shard_size=4, verbose=False, sync_at_end=True)
    if hasattr(trainer, 'sync_to_hash'):
        trainer.sync_to_hash()

    model.b._hash_table.clear()
    model.b._n_entries = 0
    model.b.n_b = 0
    model.b.global_step = 0
    model.b._a_cache_dirty = True
    trainer.invalidate_dense_state()
    model.b._eta = ETA

    if hasattr(trainer, '_gold_idx_np'):
        trainer._gold_idx_np = None
    if hasattr(trainer, '_word_proj_mx'):
        trainer._word_proj_mx = None

    ds = trainer.build_dataset(corpus, verbose=False, shard_size=4)
    for r in range(1, rounds + 1):
        if trainer._b_flat is not None:
            trainer._b_flat = trainer._b_flat * (1.0 - lambda_b)
            trainer.b_dense = trainer._b_flat.reshape(trainer.b_dense.shape)
            mx.eval(trainer._b_flat)
        trainer.run_round(ds, verbose=False)

    if hasattr(trainer, 'sync_to_hash'):
        trainer.sync_to_hash()
    ev = DenseEvaluatorMLX(model, ds.slot_universe, trainer=trainer)
    res = ev.evaluate_dataset(ds)
    return res


def evaluate_variant(model1, model2, corpus, decoder1, decoder2, mode='hard'):
    """Evaluate two-layer pipeline.

    mode:
      'hard'     - L2 sees only L1's winner word (hard attention)
      'residual' - L2 sees actual word + L1's winner word
      'residual_soft' - L2 sees actual word + L1's top-3 (softmax weighted)
    """
    correct = total = 0
    for sentence in corpus:
        words = tokenize(sentence)
        if len(words) < 2:
            continue
        model1.q.reset()
        model2.q.reset()
        for i in range(len(words) - 1):
            # Layer 1: observe actual word, predict
            model1.q.update(model1.retina.encode(words[i]))
            active1, Q1 = model1.q.get_active()
            if len(active1) == 0:
                total += 1
                continue
            phi1 = model1.phi_encoder.encode(Q1)
            scores1 = model1.b.predict(active1, phi1)

            # Layer 2 Q update based on mode
            if mode == 'hard':
                # Only L1's winner prediction
                ws1 = decoder1.decode_top_k(scores1, k=1)
                if ws1 and ws1[0][1] > 0:
                    winner = ws1[0][0]
                    counts = {s: 1 for s in model2.word_to_slots[winner].tolist()}
                    model2.q.update(counts)
                else:
                    model2.q.update({})

            elif mode == 'residual':
                # Actual word + L1's winner
                actual_counts = model2.retina.encode(words[i])
                model2.q.update(actual_counts)
                ws1 = decoder1.decode_top_k(scores1, k=1)
                if ws1 and ws1[0][1] > 0:
                    winner = ws1[0][0]
                    # 用 0.5 的权重加入预测词 (避免主导实际观测)
                    pred_counts = {s: 0.5 for s in model2.word_to_slots[winner].tolist()}
                    model2.q.update(pred_counts)

            elif mode == 'residual_soft':
                # Actual word + L1's top-3 (softmax)
                actual_counts = model2.retina.encode(words[i])
                model2.q.update(actual_counts)
                ws1 = decoder1.decode_top_k(scores1, k=3)
                ws1_pos = [(w, s) for w, s in ws1 if s > 0]
                if ws1_pos:
                    raw = np.array([s for _, s in ws1_pos])
                    raw = raw - raw.max()
                    probs = np.exp(raw)
                    probs /= probs.sum()
                    pred_counts = {}
                    for (w, _), p in zip(ws1_pos, probs):
                        for s in model2.word_to_slots[w].tolist():
                            pred_counts[s] = pred_counts.get(s, 0) + p * 0.5
                    model2.q.update(pred_counts)

            # Layer 2 predict
            active2, Q2 = model2.q.get_active()
            if len(active2) == 0:
                total += 1
                continue
            phi2 = model2.phi_encoder.encode(Q2)
            scores2 = model2.b.predict(active2, phi2)
            ws2 = decoder2.decode_top_k(scores2, k=1)
            if ws2 and ws2[0][0] == words[i + 1]:
                correct += 1
            total += 1
    return correct, total


def train_layer2_online(model1, model2, corpus, decoder1, mode='hard', rounds=64):
    """Train Layer 2 online with specified input mode."""
    for r in range(1, rounds + 1):
        decay_b_hash(model2, LAMBDA_B_ROUND)
        for sentence in corpus:
            words = tokenize(sentence)
            if len(words) < 2:
                continue
            model1.q.reset()
            model2.q.reset()
            for i in range(len(words) - 1):
                # Layer 1 forward
                model1.q.update(model1.retina.encode(words[i]))
                active1, Q1 = model1.q.get_active()
                if len(active1) == 0:
                    continue
                phi1 = model1.phi_encoder.encode(Q1)
                scores1 = model1.b.predict(active1, phi1)

                # Layer 2 Q update
                if mode == 'hard':
                    ws1 = decoder1.decode_top_k(scores1, k=1)
                    if ws1 and ws1[0][1] > 0:
                        winner = ws1[0][0]
                        counts = {s: 1 for s in model2.word_to_slots[winner].tolist()}
                        model2.q.update(counts)
                    else:
                        model2.q.update({})

                elif mode == 'residual':
                    actual_counts = model2.retina.encode(words[i])
                    model2.q.update(actual_counts)
                    ws1 = decoder1.decode_top_k(scores1, k=1)
                    if ws1 and ws1[0][1] > 0:
                        winner = ws1[0][0]
                        pred_counts = {s: 0.5 for s in model2.word_to_slots[winner].tolist()}
                        model2.q.update(pred_counts)

                elif mode == 'residual_soft':
                    actual_counts = model2.retina.encode(words[i])
                    model2.q.update(actual_counts)
                    ws1 = decoder1.decode_top_k(scores1, k=3)
                    ws1_pos = [(w, s) for w, s in ws1 if s > 0]
                    if ws1_pos:
                        raw = np.array([s for _, s in ws1_pos])
                        raw = raw - raw.max()
                        probs = np.exp(raw)
                        probs /= probs.sum()
                        pred_counts = {}
                        for (w, _), p in zip(ws1_pos, probs):
                            for s in model2.word_to_slots[w].tolist():
                                pred_counts[s] = pred_counts.get(s, 0) + p * 0.5
                        model2.q.update(pred_counts)

                # Layer 2 write (supervised)
                active2, Q2 = model2.q.get_active()
                if len(active2) == 0:
                    continue
                phi2 = model2.phi_encoder.encode(Q2)
                target_counts = model2.retina.encode(words[i + 1])
                model2.b.write(target_counts, active2, phi2, None, None,
                               a_knowledge=model2.a)

        if r in [1, 4, 8, 16, 32, 48, 64]:
            decoder2 = get_decoder(model2)
            c, t = evaluate_variant(model1, model2, corpus, decoder1, decoder2, mode)
            acc = 100 * c / t if t > 0 else 0
            n_active = model2.q._n
            print(f"    r{r:2d}: {acc:.2f}%  B2={model2.b._n_entries}  Q2_active={n_active}")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
print("Loading corpus...")
corpus = load_corpus(CORPUS_PATH)

# Phase 1: Single-layer baseline
print(f"\n{'='*70}")
print("Phase 1: Single-layer baseline")
print(f"{'='*70}")
model1 = build_fresh_model(corpus)
res1 = train_layer1_batch(model1, corpus, rounds=L1_ROUNDS)
print(f"  Layer 1: {res1.accuracy:.2f}% ({res1.correct}/{res1.total})")
decoder1 = get_decoder(model1)

# Phase 2A: Hard attention (L2 sees only L1's winner)
print(f"\n{'='*70}")
print("Phase 2A: Two-layer (L2 sees L1 winner only, hard attention)")
print(f"{'='*70}")
model2a = build_fresh_model(corpus)
model2a.b._eta = ETA
train_layer2_online(model1, model2a, corpus, decoder1, mode='hard', rounds=L2_ROUNDS)

decoder2a = get_decoder(model2a)
c2a, t2a = evaluate_variant(model1, model2a, corpus, decoder1, decoder2a, 'hard')
acc2a = 100 * c2a / t2a

# Phase 2B: Residual (L2 sees actual word + L1 winner)
print(f"\n{'='*70}")
print("Phase 2B: Two-layer (L2 sees actual word + L1 winner, residual)")
print(f"{'='*70}")
model2b = build_fresh_model(corpus)
model2b.b._eta = ETA
train_layer2_online(model1, model2b, corpus, decoder1, mode='residual', rounds=L2_ROUNDS)

decoder2b = get_decoder(model2b)
c2b, t2b = evaluate_variant(model1, model2b, corpus, decoder1, decoder2b, 'residual')
acc2b = 100 * c2b / t2b

# Phase 2C: Residual + soft (L2 sees actual word + L1 top-3 soft)
print(f"\n{'='*70}")
print("Phase 2C: Two-layer (L2 sees actual word + L1 top-3 soft, residual)")
print(f"{'='*70}")
model2c = build_fresh_model(corpus)
model2c.b._eta = ETA
train_layer2_online(model1, model2c, corpus, decoder1, mode='residual_soft', rounds=L2_ROUNDS)

decoder2c = get_decoder(model2c)
c2c, t2c = evaluate_variant(model1, model2c, corpus, decoder1, decoder2c, 'residual_soft')
acc2c = 100 * c2c / t2c

# Summary
print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(f"  Phase 1  - Single-layer:                  {res1.accuracy:.2f}%")
print(f"  Phase 2A - L2: L1 winner only (hard):     {acc2a:.2f}%")
print(f"  Phase 2B - L2: actual + L1 winner (res):  {acc2b:.2f}%")
print(f"  Phase 2C - L2: actual + L1 top3 (res+sf): {acc2c:.2f}%")
