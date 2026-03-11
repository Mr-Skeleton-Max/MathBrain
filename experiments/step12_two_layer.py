#!/usr/bin/env python3
"""Step 12: Two-Layer MathBrain 实验

测试多层时间累积是否能提升预测精度:
  Layer 1: 从原始 token 输入，产生预测分布
  Layer 2: 从 Layer 1 的预测流作为输入，预测下一个词

流程 (每个 timestep i):
  1. 观察 w_i → Layer 1 Q/φ/B → 预测分布 P1
  2. P1 top-k → soft slot counts → Layer 2 Q/φ/B → 预测 w_{i+1}

实验:
  Phase 1: 单层 baseline (batch, edge-level, λ_B=0.05)
  Phase 2: 两层 (frozen supervised L1 → supervised L2)
  Phase 3: 两层 (self-organizing L1 → supervised L2)
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

# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════
CORPUS_PATH = "datasets/tinystories_10.txt"
LAMBDA_B_ROUND = 0.05
ETA = 0.5
TOP_K = 5            # Layer 1 → Layer 2 的 top-k 词数
L1_ROUNDS = 64       # Layer 1 batch 训练轮次
L2_ROUNDS = 64       # Layer 2 online 训练轮次
SO_ROUNDS = 128      # Self-organizing Layer 1 训练轮次


# ═══════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════
def tokenize(s):
    return WakeDataset._tokenize(s)


def build_fresh_model(corpus):
    """Build MathBrain with vocabulary registered."""
    model = MathBrain(MathBrainConfig())
    for s in corpus:
        for w in tokenize(s):
            model._ensure_word(w)
    return model


def get_decoder(model):
    return SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)


def layer1_predict_word_scores(model, decoder, active_slots, phi, top_k=5):
    """Layer 1 forward: 返回 top-k (word, score) pairs."""
    if len(active_slots) == 0:
        return []
    scores = model.b.predict(active_slots, phi)
    return decoder.decode_top_k(scores, k=top_k)


def word_scores_to_soft_counts(word_scores, word_to_slots):
    """将 [(word, score), ...] 转为 {slot: activation} for Q.update().

    使用 softmax 归一化，使总激活量 ≈ 1 个正常词的观测量。
    """
    if not word_scores:
        return {}

    raw = [(w, s) for w, s in word_scores if s > 0]
    if not raw:
        return {}

    scores_arr = np.array([s for _, s in raw])
    # softmax → probs sum to 1
    scores_arr = scores_arr - scores_arr.max()
    probs = np.exp(scores_arr)
    probs /= probs.sum()

    soft_counts = {}
    for (word, _), prob in zip(raw, probs):
        for slot in word_to_slots[word].tolist():
            soft_counts[slot] = soft_counts.get(slot, 0) + prob
    return soft_counts


def decay_b_hash(model, decay_rate=0.05):
    """Apply round-level decay to all B hash table entries."""
    n = model.b._n_entries
    if n > 0:
        model.b._vecs[:n] *= (1 - decay_rate)


def evaluate_single_layer(model, corpus, decoder):
    """Evaluate single-layer B prediction."""
    correct = total = 0
    for sentence in corpus:
        words = tokenize(sentence)
        if len(words) < 2:
            continue
        model.q.reset()
        for i in range(len(words) - 1):
            model.q.update(model.retina.encode(words[i]))
            active, Q_vals = model.q.get_active()
            if len(active) == 0:
                total += 1
                continue
            phi = model.phi_encoder.encode(Q_vals)
            ws = decoder.decode_top_k(model.b.predict(active, phi), k=1)
            if ws and ws[0][0] == words[i + 1]:
                correct += 1
            total += 1
    return correct, total


def evaluate_two_layer(model1, model2, corpus, decoder1, decoder2, top_k=5):
    """Evaluate two-layer pipeline: L1 predict → feed L2 → L2 predict."""
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
            word_scores1 = layer1_predict_word_scores(
                model1, decoder1, active1,
                model1.phi_encoder.encode(Q1) if len(active1) > 0 else np.zeros((0, model1.config.d)),
                top_k=top_k)

            # Feed Layer 1's predictions to Layer 2's Q
            soft_counts = word_scores_to_soft_counts(word_scores1, model2.word_to_slots)
            if soft_counts:
                model2.q.update(soft_counts)

            # Layer 2: predict
            active2, Q2 = model2.q.get_active()
            if len(active2) == 0:
                total += 1
                continue
            phi2 = model2.phi_encoder.encode(Q2)
            ws2 = decoder2.decode_top_k(model2.b.predict(active2, phi2), k=1)
            if ws2 and ws2[0][0] == words[i + 1]:
                correct += 1
            total += 1
    return correct, total


# ═══════════════════════════════════════════════
# Phase 1: Single-Layer Baseline
# ═══════════════════════════════════════════════
print("Loading corpus...")
corpus = load_corpus(CORPUS_PATH)
print(f"  {len(corpus)} sentences")

print(f"\n{'='*70}")
print(f"Phase 1: Single-layer baseline (batch, edge-level, λ_B={LAMBDA_B_ROUND})")
print(f"{'='*70}")

model1 = build_fresh_model(corpus)
trainer1 = TreeWakeTrainerMLX(model1)

# Init dense structure
ds1 = trainer1.build_dataset(corpus, verbose=False, shard_size=4)
trainer1.run_corpus_parallel(ds1, rounds=16, shard_size=4, verbose=False, sync_at_end=True)
if hasattr(trainer1, 'sync_to_hash'):
    trainer1.sync_to_hash()

# Clear and retrain with decay
model1.b._hash_table.clear()
model1.b._n_entries = 0
model1.b.n_b = 0
model1.b.global_step = 0
model1.b._a_cache_dirty = True
trainer1.invalidate_dense_state()
model1.b._eta = ETA

if hasattr(trainer1, '_gold_idx_np'):
    trainer1._gold_idx_np = None
if hasattr(trainer1, '_word_proj_mx'):
    trainer1._word_proj_mx = None

ds1 = trainer1.build_dataset(corpus, verbose=False, shard_size=4)

for r in range(1, L1_ROUNDS + 1):
    if trainer1._b_flat is not None:
        trainer1._b_flat = trainer1._b_flat * (1.0 - LAMBDA_B_ROUND)
        trainer1.b_dense = trainer1._b_flat.reshape(trainer1.b_dense.shape)
        mx.eval(trainer1._b_flat)
    trainer1.run_round(ds1, verbose=False)
    if r in [16, 32, 48, 64]:
        if hasattr(trainer1, 'sync_to_hash'):
            trainer1.sync_to_hash()
        ev = DenseEvaluatorMLX(model1, ds1.slot_universe, trainer=trainer1)
        res = ev.evaluate_dataset(ds1)
        print(f"  r{r:2d}: {res.accuracy:.2f}%")

# Freeze Layer 1: sync to hash table for predict()
if hasattr(trainer1, 'sync_to_hash'):
    trainer1.sync_to_hash()
ev = DenseEvaluatorMLX(model1, ds1.slot_universe, trainer=trainer1)
res1 = ev.evaluate_dataset(ds1)
print(f"\n  Layer 1 final: {res1.accuracy:.2f}% ({res1.correct}/{res1.total})")

decoder1 = get_decoder(model1)

# Also evaluate with online predict (make sure hash-based predict matches)
c1_online, t1_online = evaluate_single_layer(model1, corpus, decoder1)
print(f"  Layer 1 online eval: {100*c1_online/t1_online:.2f}% ({c1_online}/{t1_online})")


# ═══════════════════════════════════════════════
# Phase 2: Two-Layer (supervised L1 → supervised L2)
# ═══════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"Phase 2: Two-layer (frozen supervised L1 → supervised L2)")
print(f"  Layer 2 receives Layer 1's top-{TOP_K} predictions as soft Q input")
print(f"{'='*70}")

model2 = build_fresh_model(corpus)
model2.b._eta = ETA
decoder2 = get_decoder(model2)

t0_phase2 = time.time()

for r in range(1, L2_ROUNDS + 1):
    # B decay for Layer 2
    decay_b_hash(model2, LAMBDA_B_ROUND)

    # Train Layer 2 online
    for sentence in corpus:
        words = tokenize(sentence)
        if len(words) < 2:
            continue
        model1.q.reset()
        model2.q.reset()

        for i in range(len(words) - 1):
            # Layer 1 forward: observe actual word
            model1.q.update(model1.retina.encode(words[i]))
            active1, Q1 = model1.q.get_active()
            if len(active1) == 0:
                continue
            phi1 = model1.phi_encoder.encode(Q1)
            word_scores1 = layer1_predict_word_scores(
                model1, decoder1, active1, phi1, top_k=TOP_K)

            # Feed to Layer 2's Q
            soft_counts = word_scores_to_soft_counts(word_scores1, model2.word_to_slots)
            if soft_counts:
                model2.q.update(soft_counts)

            # Layer 2 write (supervised: knows target w_{i+1})
            active2, Q2 = model2.q.get_active()
            if len(active2) == 0:
                continue
            phi2 = model2.phi_encoder.encode(Q2)
            target_counts = model2.retina.encode(words[i + 1])
            model2.b.write(target_counts, active2, phi2, None, None,
                           a_knowledge=model2.a)

    if r in [1, 4, 8, 16, 32, 48, 64]:
        c, t = evaluate_two_layer(model1, model2, corpus, decoder1, decoder2, TOP_K)
        acc = 100 * c / t if t > 0 else 0
        print(f"  r{r:2d}: two-layer={acc:.2f}% ({c}/{t})"
              f"  B2_entries={model2.b._n_entries}")

phase2_ms = (time.time() - t0_phase2) * 1000
c2_final, t2_final = evaluate_two_layer(model1, model2, corpus, decoder1, decoder2, TOP_K)
acc2 = 100 * c2_final / t2_final if t2_final > 0 else 0
print(f"\n  Phase 2 final: {acc2:.2f}% ({c2_final}/{t2_final}), {phase2_ms:.0f}ms")


# ═══════════════════════════════════════════════
# Phase 3: Two-Layer (self-organizing L1 → supervised L2)
# ═══════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"Phase 3: Two-layer (self-organizing L1 → supervised L2)")
print(f"  Layer 1: winner-take-all, λ_B={LAMBDA_B_ROUND}, {SO_ROUNDS} rounds")
print(f"{'='*70}")

model_so = build_fresh_model(corpus)
model_so.b._eta = ETA
decoder_so = get_decoder(model_so)
vocab_list = sorted(model_so.vocab)

t0_phase3 = time.time()

print("  Training self-organizing Layer 1 (winner-take-all)...")
for r in range(1, SO_ROUNDS + 1):
    decay_b_hash(model_so, LAMBDA_B_ROUND)

    for sentence in corpus:
        words = tokenize(sentence)
        if len(words) < 2:
            continue
        model_so.q.reset()

        for i in range(len(words) - 1):
            model_so.q.update(model_so.retina.encode(words[i]))
            active, Q_vals = model_so.q.get_active()
            if len(active) == 0:
                continue
            phi = model_so.phi_encoder.encode(Q_vals)

            # Forward: find winner
            scores = model_so.b.predict(active, phi)
            winners = decoder_so.decode_top_k(scores, k=1)

            if winners and winners[0][1] > 0:
                winner_word = winners[0][0]
            else:
                # Bootstrapping: random word
                winner_word = vocab_list[random.randint(0, len(vocab_list) - 1)]

            # Write: strengthen winner's connections
            target_counts = model_so.retina.encode(winner_word)
            model_so.b.write(target_counts, active, phi, None, None,
                             a_knowledge=model_so.a)

    if r in [1, 4, 8, 16, 32, 64, 96, 128]:
        c_so, t_so = evaluate_single_layer(model_so, corpus, decoder_so)
        acc_so = 100 * c_so / t_so if t_so > 0 else 0
        print(f"    r{r:3d}: L1 self-org accuracy={acc_so:.2f}%  "
              f"B_entries={model_so.b._n_entries}")

# Train Layer 2 on self-organizing L1
print("\n  Training supervised Layer 2 on self-organizing L1...")
model2_so = build_fresh_model(corpus)
model2_so.b._eta = ETA
decoder2_so = get_decoder(model2_so)

for r in range(1, L2_ROUNDS + 1):
    decay_b_hash(model2_so, LAMBDA_B_ROUND)

    for sentence in corpus:
        words = tokenize(sentence)
        if len(words) < 2:
            continue
        model_so.q.reset()
        model2_so.q.reset()

        for i in range(len(words) - 1):
            # Layer 1 forward
            model_so.q.update(model_so.retina.encode(words[i]))
            active1, Q1 = model_so.q.get_active()
            if len(active1) == 0:
                continue
            phi1 = model_so.phi_encoder.encode(Q1)
            word_scores1 = layer1_predict_word_scores(
                model_so, decoder_so, active1, phi1, top_k=TOP_K)

            # Feed to Layer 2
            soft_counts = word_scores_to_soft_counts(word_scores1, model2_so.word_to_slots)
            if soft_counts:
                model2_so.q.update(soft_counts)

            # Layer 2 write (supervised)
            active2, Q2 = model2_so.q.get_active()
            if len(active2) == 0:
                continue
            phi2 = model2_so.phi_encoder.encode(Q2)
            target_counts = model2_so.retina.encode(words[i + 1])
            model2_so.b.write(target_counts, active2, phi2, None, None,
                              a_knowledge=model2_so.a)

    if r in [1, 4, 8, 16, 32, 48, 64]:
        c, t = evaluate_two_layer(model_so, model2_so, corpus, decoder_so, decoder2_so, TOP_K)
        acc = 100 * c / t if t > 0 else 0
        print(f"    r{r:2d}: two-layer={acc:.2f}% ({c}/{t})"
              f"  B2_entries={model2_so.b._n_entries}")

phase3_ms = (time.time() - t0_phase3) * 1000
c3_final, t3_final = evaluate_two_layer(model_so, model2_so, corpus, decoder_so, decoder2_so, TOP_K)
acc3 = 100 * c3_final / t3_final if t3_final > 0 else 0


# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════
print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(f"  Phase 1 - Single-layer (supervised):              {res1.accuracy:.2f}%")
print(f"  Phase 2 - Two-layer (supervised L1 → L2):         {acc2:.2f}%")
print(f"  Phase 3 - Two-layer (self-org L1 → supervised L2):{acc3:.2f}%")
print(f"\n  Phase 2 time: {phase2_ms/1000:.1f}s")
print(f"  Phase 3 time: {phase3_ms/1000:.1f}s")
