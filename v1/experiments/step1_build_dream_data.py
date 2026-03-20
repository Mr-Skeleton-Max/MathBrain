#!/usr/bin/env python3
"""Step 1: Fresh model → wake → export dream dataset → inspect word-level top-1."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.wake_dataset import WakeDataset
from mathbrain.wake_tree_trainer_mlx import TreeWakeTrainerMLX
from experiments.run_parallel_wake import load_corpus, evaluate_model
from experiments.export_dream_dataset import _build_q_distribution_dataset
from experiments.train_a_from_qdist import _prepare_dataset

CORPUS_PATH = "datasets/tinystories_10.txt"
DREAM_OUT = ".codex_logs/step1_dream.npz"

# --- 1. Fresh model + register vocab ---
corpus = load_corpus(CORPUS_PATH)
model = MathBrain(MathBrainConfig())
for s in corpus:
    for w in WakeDataset._tokenize(s):
        model._ensure_word(w)
print(f"Vocab: {len(model.vocab)} words, slots up to {model.config.K}")

# --- 2. Wake training (cycle 1: random A, empty B → train B) ---
trainer = TreeWakeTrainerMLX(model)
dataset = trainer.build_dataset(corpus, verbose=False, shard_size=4)
trainer.run_corpus_parallel(dataset, rounds=16, shard_size=4, verbose=False, sync_at_end=True)
if hasattr(trainer, 'sync_to_hash'):
    trainer.sync_to_hash()

result = evaluate_model(model, corpus, dataset, trainer, 'mlx', 'parallel')
print(f"Wake A+B acc: {result['correct']}/{result['total']} = {result['accuracy']:.2f}%")

# --- 3. Export dream dataset (Q → slot-level teacher scores) ---
dream_data, dream_info = _build_q_distribution_dataset(model, corpus, threshold=0.01)
Path(DREAM_OUT).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(DREAM_OUT, **dream_data)
print(f"\nDream dataset: {dream_info['total_positions']} positions, "
      f"{dream_info['slot_universe_size']} slots")
print(f"  teacher peak mean: {dream_info['teacher_peak_after_mean']:.4f}")
print(f"  nonzero slots/pos: {dream_info['teacher_nonzero_after_mean']:.1f}")

# --- 4. Inspect: teacher slot scores → word scores → top-1 ---
data = _prepare_dataset(model, np.load(DREAM_OUT))
slot_universe = data['slot_universe']
teacher_scores = data['teacher_scores']   # (P, S) slot-level
gold_slot_lists = data['gold_slot_lists']

# Build word projection
from mathbrain.inference_fast import SparseMatrixDecoder
if model._decoder_dirty or model._decoder is None:
    model._decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)
    model._decoder_dirty = False
decoder = model._decoder
vocab_list = decoder.word_list

slot_to_compact = {int(s): i for i, s in enumerate(slot_universe.tolist())}
V = len(vocab_list)
S = len(slot_universe)
word_proj = np.zeros((V, S), dtype=np.float32)
gold_by_tuple = {}
for word_idx, word in enumerate(vocab_list):
    slots = model.word_to_slots[word]
    compact = sorted(slot_to_compact[int(sl)] for sl in slots if int(sl) in slot_to_compact)
    if compact:
        word_proj[word_idx, compact] = 1.0 / len(compact)
        gold_by_tuple[tuple(compact)] = word_idx

# Teacher word scores
teacher_word = teacher_scores @ word_proj.T   # (P, V)
teacher_top1 = np.argmax(teacher_word, axis=1)

# Gold labels
gold_labels = np.full(len(gold_slot_lists), -1, dtype=np.int32)
for i, gsl in enumerate(gold_slot_lists):
    key = tuple(sorted(slot_to_compact[int(s)] for s in gsl.tolist() if int(s) in slot_to_compact))
    if key in gold_by_tuple:
        gold_labels[i] = gold_by_tuple[key]

valid = gold_labels >= 0
teacher_acc = float(np.mean(teacher_top1[valid] == gold_labels[valid]))
print(f"\n=== Word-level teacher top-1 ===")
print(f"  Teacher top-1 matches gold: {teacher_acc*100:.2f}%")
print(f"  Teacher word score stats:")
print(f"    peak (correct word): mean={np.mean(np.max(teacher_word, axis=1)):.4f}")

# Show a few examples
print(f"\n  Sample positions (first 10):")
for i in range(min(10, len(teacher_top1))):
    tw = teacher_word[i]
    top1_word = vocab_list[teacher_top1[i]]
    top1_score = tw[teacher_top1[i]]
    gold_word = vocab_list[gold_labels[i]] if gold_labels[i] >= 0 else "?"
    gold_score = tw[gold_labels[i]] if gold_labels[i] >= 0 else 0.0
    top5_idx = np.argsort(tw)[-5:][::-1]
    top5 = [(vocab_list[j], f"{tw[j]:.4f}") for j in top5_idx]
    match = "✓" if teacher_top1[i] == gold_labels[i] else "✗"
    print(f"    pos[{i}] gold={gold_word}({gold_score:.4f}) "
          f"top1={top1_word}({top1_score:.4f}) {match}  top5={top5}")

# Training data summary
print(f"\n=== Sleep training data summary ===")
print(f"  Positions: {len(teacher_top1)}")
print(f"  Vocab size (V): {V}")
print(f"  Slot universe (S): {S}")
print(f"  Teacher top-1 accuracy: {teacher_acc*100:.2f}%")
print(f"  Target: one-hot on teacher top-1 word")
print(f"  A's normalized word scores should directly match this one-hot")
