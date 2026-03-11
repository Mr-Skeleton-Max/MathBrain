#!/usr/bin/env python3
"""Step 2: 基于 step1 的 dream data 训练 A.

目标: A 的归一化 word scores 直接拟合 one-hot (狄利克雷分布)。
不用 softmax，不用 CE。直接 MSE(word_scores_normalized, one_hot_target)。
只 sleep 一次。
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.optimizers as optim

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.wake_dataset import WakeDataset
from mathbrain.wake_tree_trainer_mlx import TreeWakeTrainerMLX
from mathbrain.inference_fast import SparseMatrixDecoder
from experiments.run_parallel_wake import load_corpus, evaluate_model
from experiments.export_dream_dataset import _build_q_distribution_dataset
from experiments.train_a_from_qdist import _prepare_dataset

CORPUS_PATH = "datasets/tinystories_10.txt"
DREAM_PATH = ".codex_logs/step1_dream.npz"

# ============================================================
# 1. Rebuild model + wake (same as step1)
# ============================================================
corpus = load_corpus(CORPUS_PATH)
model = MathBrain(MathBrainConfig())
for s in corpus:
    for w in WakeDataset._tokenize(s):
        model._ensure_word(w)

trainer = TreeWakeTrainerMLX(model)
dataset = trainer.build_dataset(corpus, verbose=False, shard_size=4)
trainer.run_corpus_parallel(dataset, rounds=16, shard_size=4, verbose=False, sync_at_end=True)
if hasattr(trainer, 'sync_to_hash'):
    trainer.sync_to_hash()

result = evaluate_model(model, corpus, dataset, trainer, 'mlx', 'parallel')
print(f"Wake A+B acc: {result['correct']}/{result['total']} = {result['accuracy']:.2f}%")

# ============================================================
# 2. Load dream data + build word projection + teacher top-1
# ============================================================
dream_data, _ = _build_q_distribution_dataset(model, corpus, threshold=0.01)
Path(DREAM_PATH).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(DREAM_PATH, **dream_data)
data = _prepare_dataset(model, np.load(DREAM_PATH))

slot_universe = data['slot_universe']
active_local = data['active_local']       # (P, n_max)
phi = data['phi']                          # (P, n_max, D)
active_mask = data['active_mask']          # (P, n_max)
teacher_scores = data['teacher_scores']    # (P, S) slot-level

# Word projection
if model._decoder_dirty or model._decoder is None:
    model._decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)
    model._decoder_dirty = False
vocab_list = model._decoder.word_list

slot_to_compact = {int(s): i for i, s in enumerate(slot_universe.tolist())}
V = len(vocab_list)
S = len(slot_universe)
word_proj = np.zeros((V, S), dtype=np.float32)
for word_idx, word in enumerate(vocab_list):
    slots = model.word_to_slots[word]
    compact = sorted(slot_to_compact[int(sl)] for sl in slots if int(sl) in slot_to_compact)
    if compact:
        word_proj[word_idx, compact] = 1.0 / len(compact)

# Teacher word scores → top-1 labels
teacher_word = teacher_scores @ word_proj.T   # (P, V)
teacher_top1 = np.argmax(teacher_word, axis=1).astype(np.int32)

print(f"Training data: {len(teacher_top1)} positions, V={V}, S={S}")

# ============================================================
# 3. Train A: MSE(normalized_word_scores, one_hot)
# ============================================================
d_rank = int(model.a.d)
D_phi = int(model.a.D)
n_slots = len(slot_universe)
n_pos = active_local.shape[0]

# Random init
E_src = np.random.randn(n_slots, d_rank, D_phi).astype(np.float32) * 0.01
E_tgt = np.random.randn(n_slots, d_rank, D_phi).astype(np.float32) * 0.01

# MLX arrays
params = {'E_src': mx.array(E_src), 'E_tgt': mx.array(E_tgt)}
active_local_mx = mx.array(active_local)
phi_mx = mx.array(phi)
active_mask_mx = mx.array(active_mask)
labels_mx = mx.array(teacher_top1)
word_proj_t_mx = mx.array(word_proj.T)   # (S, V)

# One-hot targets: (P, V)
one_hot_np = np.zeros((n_pos, V), dtype=np.float32)
one_hot_np[np.arange(n_pos), teacher_top1] = 1.0
one_hot_mx = mx.array(one_hot_np)

EPOCHS = 80
BATCH_SIZE = 128
LR = 0.01
WD = 1e-4

n_batches = max(1, int(np.ceil(n_pos / BATCH_SIZE)))
lr_schedule = optim.cosine_decay(init=LR, decay_steps=max(1, EPOCHS * n_batches))
optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=WD)

def _loss_fn(params, batch_active, batch_phi, batch_mask, batch_target):
    # A's slot-level scores (normalized)
    src_emb = params['E_src'][batch_active]               # (B, n_max, d, D)
    ctx = mx.sum(
        src_emb * batch_phi[:, :, None, :] * batch_mask[:, :, None, None],
        axis=1
    )                                                      # (B, d, D)
    ctx_flat = mx.reshape(ctx, (int(ctx.shape[0]), -1))   # (B, d*D)
    E_tgt_flat = mx.reshape(params['E_tgt'], (int(params['E_tgt'].shape[0]), -1))
    slot_scores = mx.matmul(ctx_flat, mx.transpose(E_tgt_flat))  # (B, S)
    active_count = mx.sum(batch_mask, axis=1, keepdims=True) + 1e-8
    slot_scores = slot_scores / active_count

    # Word-level scores (no softmax)
    word_scores = mx.matmul(slot_scores, word_proj_t_mx)  # (B, V)

    # MSE against one-hot target
    return mx.mean((word_scores - batch_target) ** 2)

loss_and_grad_fn = mx.value_and_grad(_loss_fn)

best_loss = float('inf')
best_params = None

for epoch in range(EPOCHS):
    perm = np.random.permutation(n_pos).astype(np.int32)
    epoch_loss = 0.0
    for bi in range(n_batches):
        s = bi * BATCH_SIZE
        e = min(s + BATCH_SIZE, n_pos)
        idx = mx.array(perm[s:e])
        loss, grads = loss_and_grad_fn(
            params,
            active_local_mx[idx],
            phi_mx[idx],
            active_mask_mx[idx],
            one_hot_mx[idx],
        )
        optimizer.update(params, grads)
        mx.eval(loss, params)
        epoch_loss += float(loss)
    epoch_loss /= n_batches

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        mx.eval(params)
        best_params = {k: np.array(v) for k, v in params.items()}

    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        # Quick eval: A's word scores on first 100 positions
        mx.eval(params)
        E_s = np.array(params['E_src'])
        E_t = np.array(params['E_tgt'])
        e_src_norm = np.mean(np.linalg.norm(E_s.reshape(n_slots, -1), axis=1))
        e_tgt_norm = np.mean(np.linalg.norm(E_t.reshape(n_slots, -1), axis=1))
        print(f"  epoch {epoch:3d}: loss={epoch_loss:.6f}  "
              f"E_src={e_src_norm:.2f}  E_tgt={e_tgt_norm:.2f}")

print(f"\nBest loss: {best_loss:.6f}")

# ============================================================
# 4. Evaluate A after sleep
# ============================================================
E_src_best = best_params['E_src']
E_tgt_best = best_params['E_tgt']

# Compute A's word scores on all positions
from experiments.train_a_from_qdist import _batched_student_logits_numpy
student_logits = _batched_student_logits_numpy(
    E_src_best, E_tgt_best, active_local, phi, active_mask, batch_size=256
)
student_word = student_logits @ word_proj.T   # (P, V)
student_top1 = np.argmax(student_word, axis=1)

# Gold labels
gold_slot_lists = data['gold_slot_lists']
gold_by_tuple = {}
for word_idx, word in enumerate(vocab_list):
    slots = model.word_to_slots[word]
    compact = sorted(slot_to_compact[int(sl)] for sl in slots if int(sl) in slot_to_compact)
    if compact:
        gold_by_tuple[tuple(compact)] = word_idx

gold_labels = np.full(n_pos, -1, dtype=np.int32)
for i, gsl in enumerate(gold_slot_lists):
    key = tuple(sorted(slot_to_compact[int(s)] for s in gsl.tolist() if int(s) in slot_to_compact))
    if key in gold_by_tuple:
        gold_labels[i] = gold_by_tuple[key]

valid = gold_labels >= 0
a_match_teacher = float(np.mean(student_top1 == teacher_top1))
a_match_gold = float(np.mean(student_top1[valid] == gold_labels[valid]))

print(f"\n=== A after single sleep ===")
print(f"  A top-1 matches teacher: {a_match_teacher*100:.2f}%")
print(f"  A top-1 matches gold:    {a_match_gold*100:.2f}%")

# Score distribution of A
print(f"\n  A word score stats:")
print(f"    mean of max:  {np.mean(np.max(student_word, axis=1)):.4f}")
print(f"    mean of min:  {np.mean(np.min(student_word, axis=1)):.4f}")
print(f"    mean of sum:  {np.mean(np.sum(student_word, axis=1)):.4f}")
print(f"    correct word score mean: {np.mean(student_word[np.arange(n_pos), teacher_top1]):.4f}")

# Sample
print(f"\n  Samples (first 10):")
for i in range(min(10, n_pos)):
    sw = student_word[i]
    top1_w = vocab_list[student_top1[i]]
    top1_s = sw[student_top1[i]]
    gold_w = vocab_list[gold_labels[i]] if gold_labels[i] >= 0 else "?"
    teacher_w = vocab_list[teacher_top1[i]]
    top3_idx = np.argsort(sw)[-3:][::-1]
    top3 = [(vocab_list[j], f"{sw[j]:.3f}") for j in top3_idx]
    print(f"    pos[{i}] gold={gold_w:10s} teacher={teacher_w:10s} "
          f"A_top1={top1_w:10s}({top1_s:.3f})  top3={top3}")

e_src_norm = np.mean(np.linalg.norm(E_src_best.reshape(n_slots, -1), axis=1))
e_tgt_norm = np.mean(np.linalg.norm(E_tgt_best.reshape(n_slots, -1), axis=1))
print(f"\n  E norms: src={e_src_norm:.2f}  tgt={e_tgt_norm:.2f}")
