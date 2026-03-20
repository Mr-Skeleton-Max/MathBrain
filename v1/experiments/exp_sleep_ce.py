#!/usr/bin/env python3
"""Wake-Sleep cycles with cross-entropy sleep loss.

核心改变: A 的 Sleep 目标从 MSE(slot分布) → CE(word top-1 one-hot)
- Teacher = combined (γ·A + B) word-level scores → argmax → one-hot
- Student = A's word-level scores → softmax → cross-entropy
- A 的低秩结构天然过滤噪声，只保留 top-1 信号
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathbrain import MathBrain, MathBrainConfig
from experiments.inspect_sleep_targets import _restore_model
from experiments.run_parallel_wake import (
    load_corpus,
    evaluate_model,
    collect_a_stats,
)
from experiments.export_dream_dataset import _build_q_distribution_dataset
from experiments.train_a_from_qdist import (
    _ensure_vocab,
    _prepare_dataset,
    _extract_student_factors,
    _batched_student_logits_numpy,
    _evaluate_a_only_accuracy,
    _save_snapshot,
)
from mathbrain.wake_dataset import WakeDataset

try:
    import mlx.core as mx
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _build_word_projection(model, slot_universe: np.ndarray):
    """Build word ↔ slot projection matrix.

    Returns:
        word_proj: (V, S) float32 - word to slot compact projection
        vocab_list: list of word strings
        gold_by_tuple: dict mapping tuple(sorted compact slots) → word_idx
    """
    from mathbrain.inference_fast import SparseMatrixDecoder
    if model._decoder_dirty or model._decoder is None:
        model._decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)
        model._decoder_dirty = False
    decoder = model._decoder
    vocab_list = decoder.word_list

    slot_to_compact = {int(s): i for i, s in enumerate(slot_universe.tolist())}
    V = len(vocab_list)
    S = len(slot_universe)
    proj = np.zeros((V, S), dtype=np.float32)
    gold_by_tuple = {}

    for word_idx, word in enumerate(vocab_list):
        slots = model.word_to_slots[word]
        compact_slots = sorted(
            slot_to_compact[int(slot)] for slot in slots
            if int(slot) in slot_to_compact
        )
        if compact_slots:
            weight = 1.0 / len(compact_slots)
            proj[word_idx, compact_slots] = weight
            gold_by_tuple[tuple(compact_slots)] = word_idx

    return proj, vocab_list, gold_by_tuple


def _sleep_a_ce(model, dream_npz_path: Path, corpus: list[str],
                epochs: int, batch_size: int, lr: float,
                zero_threshold: float,
                weight_decay: float = 1e-4) -> dict:
    """Train A via word-level cross-entropy on teacher's top-1.

    Instead of MSE on slot-level scores:
      loss = ||A_slots - teacher_slots||²

    We use CE on word-level predictions:
      teacher_word = teacher_slots @ word_proj.T → argmax → label
      student_word = A_slots @ word_proj.T → softmax → CE(label)
    """
    dataset_npz = np.load(dream_npz_path)
    data = _prepare_dataset(model, dataset_npz)
    slot_universe = data['slot_universe']
    active_local = data['active_local']       # (P, n_max)
    phi = data['phi']                          # (P, n_max, D)
    active_mask = data['active_mask']          # (P, n_max)
    teacher_scores = data['teacher_scores']    # (P, S) slot-level
    gold_slot_lists = data['gold_slot_lists']

    # --- 1. Build word projection ---
    word_proj, vocab_list, gold_by_tuple = _build_word_projection(model, slot_universe)
    V = len(vocab_list)
    S = len(slot_universe)

    # --- 2. Teacher word scores → top-1 labels ---
    teacher_word_scores = teacher_scores @ word_proj.T   # (P, V)
    teacher_labels = np.argmax(teacher_word_scores, axis=1).astype(np.int32)  # (P,)

    # Gold word labels (for evaluation, not training)
    # gold_slot_lists contains raw slot IDs; gold_by_tuple keys are compact indices
    slot_to_compact = {int(s): i for i, s in enumerate(slot_universe.tolist())}
    gold_labels = np.full(len(gold_slot_lists), -1, dtype=np.int32)
    for i, gold_slots in enumerate(gold_slot_lists):
        compact = sorted(
            slot_to_compact[int(s)] for s in gold_slots.tolist()
            if int(s) in slot_to_compact
        )
        key = tuple(compact)
        if key in gold_by_tuple:
            gold_labels[i] = gold_by_tuple[key]

    # Teacher accuracy (how often teacher top-1 matches gold)
    valid_gold = gold_labels >= 0
    teacher_gold_match = float(np.mean(teacher_labels[valid_gold] == gold_labels[valid_gold])) if np.any(valid_gold) else 0.0

    # --- 3. Student setup ---
    d_rank = max(1, int(model.a.d))
    D_phi = int(model.a.D)
    n_slots = len(slot_universe)

    if model.a.has_knowledge and model.a.E_src:
        E_src, E_tgt = _extract_student_factors(model, slot_universe)
    else:
        # First sleep: random init
        E_src = np.random.randn(n_slots, d_rank, D_phi).astype(np.float32) * 0.01
        E_tgt = np.random.randn(n_slots, d_rank, D_phi).astype(np.float32) * 0.01

    # Before metrics
    before_logits = _batched_student_logits_numpy(
        E_src, E_tgt, active_local, phi, active_mask, batch_size=256
    )
    before_word = before_logits @ word_proj.T
    before_top1 = np.argmax(before_word, axis=1)
    before_teacher_match = float(np.mean(before_top1 == teacher_labels))
    before_gold_match = float(np.mean(before_top1[valid_gold] == gold_labels[valid_gold])) if np.any(valid_gold) else 0.0

    # A-only accuracy before
    before_acc = _evaluate_a_only_accuracy(
        model, slot_universe, active_local, phi, active_mask, gold_slot_lists
    )

    print(f"    Before CE sleep: teacher_match={before_teacher_match:.4f}  "
          f"gold_match={before_gold_match:.4f}  a_word_acc={before_acc['word_acc']:.4f}")
    print(f"    Teacher gold accuracy: {teacher_gold_match:.4f}")
    print(f"    Vocab: {V} words, {S} slots, d_rank={d_rank}")

    # --- 4. MLX training ---
    params = {'E_src': mx.array(E_src), 'E_tgt': mx.array(E_tgt)}
    active_local_mx = mx.array(active_local)
    phi_mx = mx.array(phi)
    active_mask_mx = mx.array(active_mask)
    labels_mx = mx.array(teacher_labels)
    word_proj_t_mx = mx.array(word_proj.T)   # (S, V)

    n_pos = active_local.shape[0]
    bs = max(1, batch_size)
    n_batches = max(1, int(np.ceil(n_pos / bs)))

    lr_schedule = optim.cosine_decay(init=float(lr), decay_steps=max(1, epochs * n_batches))
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)

    def _loss_fn(params, batch_active, batch_phi, batch_mask, batch_labels):
        # A's slot-level scores
        src_emb = params['E_src'][batch_active]               # (B, n_max, d, D)
        ctx = mx.sum(
            src_emb * batch_phi[:, :, None, :] * batch_mask[:, :, None, None],
            axis=1
        )                                                      # (B, d, D)
        ctx_flat = mx.reshape(ctx, (int(ctx.shape[0]), -1))   # (B, d*D)
        E_tgt_flat = mx.reshape(params['E_tgt'], (int(params['E_tgt'].shape[0]), -1))  # (S, d*D)
        slot_scores = mx.matmul(ctx_flat, mx.transpose(E_tgt_flat))  # (B, S)
        active_count = mx.sum(batch_mask, axis=1, keepdims=True) + 1e-8
        slot_scores = slot_scores / active_count

        # Project to word-level scores
        word_scores = mx.matmul(slot_scores, word_proj_t_mx)  # (B, V)

        # Cross-entropy with one-hot target on teacher's top-1
        # Numerically stable: log_softmax then gather
        shifted = word_scores - mx.max(word_scores, axis=1, keepdims=True)
        log_sum_exp = mx.log(mx.sum(mx.exp(shifted), axis=1, keepdims=True) + 1e-8)
        log_probs = shifted - log_sum_exp                      # (B, V)

        # Gather log-prob of correct label
        nll = -mx.take_along_axis(log_probs, batch_labels[:, None], axis=1)  # (B, 1)
        return mx.mean(nll)

    loss_and_grad_fn = mx.value_and_grad(_loss_fn)

    best_loss = float('inf')
    best_src = E_src.copy()
    best_tgt = E_tgt.copy()
    loss_history = []

    for epoch in range(epochs):
        perm = np.random.permutation(n_pos).astype(np.int32)
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start = batch_idx * bs
            end = min(start + bs, n_pos)
            batch_ids = mx.array(perm[start:end])
            loss, grads = loss_and_grad_fn(
                params,
                active_local_mx[batch_ids],
                phi_mx[batch_ids],
                active_mask_mx[batch_ids],
                labels_mx[batch_ids],
            )
            optimizer.update(params, grads)
            mx.eval(loss, params)
            epoch_loss += float(loss)
        epoch_loss /= n_batches
        loss_history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_src = np.array(params['E_src'])
            best_tgt = np.array(params['E_tgt'])

    # --- 5. Write back best ---
    for idx, slot in enumerate(slot_universe.tolist()):
        model.a.E_src[int(slot)] = best_src[idx].astype(np.float32, copy=True)
        model.a.E_tgt[int(slot)] = best_tgt[idx].astype(np.float32, copy=True)
    model.a.has_knowledge = True
    model.a.sleep_cycle_count = getattr(model.a, 'sleep_cycle_count', 0) + 1
    model.a._build_cache()
    if hasattr(model.b, '_a_cache_dirty'):
        model.b._a_cache_dirty = True

    # --- 6. After metrics ---
    after_logits = _batched_student_logits_numpy(
        best_src, best_tgt, active_local, phi, active_mask, batch_size=256
    )
    after_word = after_logits @ word_proj.T
    after_top1 = np.argmax(after_word, axis=1)
    after_teacher_match = float(np.mean(after_top1 == teacher_labels))
    after_gold_match = float(np.mean(after_top1[valid_gold] == gold_labels[valid_gold])) if np.any(valid_gold) else 0.0

    after_acc = _evaluate_a_only_accuracy(
        model, slot_universe, active_local, phi, active_mask, gold_slot_lists
    )

    # E norms
    e_src_norms = np.linalg.norm(best_src.reshape(len(slot_universe), -1), axis=1)
    e_tgt_norms = np.linalg.norm(best_tgt.reshape(len(slot_universe), -1), axis=1)

    print(f"    After CE sleep: teacher_match={after_teacher_match:.4f}  "
          f"gold_match={after_gold_match:.4f}  a_word_acc={after_acc['word_acc']:.4f}  "
          f"ce_loss={best_loss:.6f}")
    print(f"    E norms: src={np.mean(e_src_norms):.2f}  tgt={np.mean(e_tgt_norms):.2f}")

    return {
        'best_loss': float(best_loss),
        'loss_history': loss_history,
        'teacher_gold_acc': teacher_gold_match,
        'before': {
            'teacher_match': before_teacher_match,
            'gold_match': before_gold_match,
            'a_word_acc': before_acc['word_acc'],
        },
        'after': {
            'teacher_match': after_teacher_match,
            'gold_match': after_gold_match,
            'a_word_acc': after_acc['word_acc'],
            'a_correct': after_acc['correct_word'],
            'a_total': after_acc['total'],
        },
        'E_src_norm': float(np.mean(e_src_norms)),
        'E_tgt_norm': float(np.mean(e_tgt_norms)),
    }


def main():
    parser = argparse.ArgumentParser(description='Wake-Sleep with CE loss')
    parser.add_argument('--init-snapshot', default=None, help='Initial snapshot (omit for fresh model)')
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--cycles', type=int, default=64)
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--sleep-epochs', type=int, default=80)
    parser.add_argument('--sleep-lr', type=float, default=0.01)
    parser.add_argument('--sleep-batch-size', type=int, default=128)
    parser.add_argument('--sleep-wd', type=float, default=1e-4,
                        help='AdamW weight decay for E_src/E_tgt')
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--shard-size', type=int, default=4)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create model
    corpus = load_corpus(args.corpus)
    if args.init_snapshot:
        model, _ = _restore_model(args.init_snapshot)
    else:
        model = MathBrain(MathBrainConfig())
    for s in corpus:
        for w in WakeDataset._tokenize(s):
            model._ensure_word(w)

    from mathbrain.wake_tree_trainer_mlx import TreeWakeTrainerMLX

    cycle_log = []

    for cycle in range(1, args.cycles + 1):
        cycle_t0 = time.time()
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle}/{args.cycles}  [CE sleep]")
        print(f"{'='*60}")

        # --- 1. Clear B ---
        model.b.clear()

        # --- 2. Wake ---
        trainer = TreeWakeTrainerMLX(model)
        try:
            dataset = trainer.build_dataset(corpus, verbose=False, shard_size=args.shard_size)
        except TypeError:
            dataset = trainer.build_dataset(corpus, verbose=False)

        t_wake = time.time()
        trainer.run_corpus_parallel(
            dataset, rounds=args.rounds,
            shard_size=args.shard_size, verbose=False, sync_at_end=True,
        )
        if hasattr(trainer, 'sync_to_hash'):
            trainer.sync_to_hash()
        wake_ms = (time.time() - t_wake) * 1000

        # Evaluate A+B
        result_ab = evaluate_model(model, corpus, dataset, trainer, 'mlx', 'parallel')
        wake_acc = result_ab['accuracy']
        print(f"  Wake: A+B acc = {result_ab['correct']}/{result_ab['total']} = {wake_acc:.2f}%  ({wake_ms:.0f}ms)")

        # --- 3. Export dream dataset ---
        t_dream = time.time()
        dream_path = output_dir / f"dream_cycle{cycle:03d}.npz"
        dream_data, dream_info = _build_q_distribution_dataset(model, corpus, threshold=args.threshold)
        np.savez_compressed(dream_path, **dream_data)
        dream_ms = (time.time() - t_dream) * 1000

        # --- 4. Sleep (CE) ---
        t_sleep = time.time()
        sleep_result = _sleep_a_ce(
            model, dream_path, corpus,
            epochs=args.sleep_epochs,
            batch_size=args.sleep_batch_size,
            lr=args.sleep_lr,
            zero_threshold=args.threshold,
            weight_decay=args.sleep_wd,
        )
        sleep_ms = (time.time() - t_sleep) * 1000

        a_word_acc = sleep_result['after']['a_word_acc']
        teacher_match = sleep_result['after']['teacher_match']
        gold_match = sleep_result['after']['gold_match']
        a_stats = collect_a_stats(model)

        print(f"  Sleep CE: a_word_acc={a_word_acc*100:.2f}%  "
              f"teacher_match={teacher_match:.4f}  gold_match={gold_match:.4f}  "
              f"ce_loss={sleep_result['best_loss']:.6f}  ({sleep_ms:.0f}ms)")
        print(f"  A norm: E_src={a_stats['E_src_norm_mean']:.2f}  E_tgt={a_stats['E_tgt_norm_mean']:.2f}")

        cycle_ms = (time.time() - cycle_t0) * 1000

        entry = {
            'cycle': cycle,
            'wake_acc': wake_acc,
            'wake_correct': result_ab['correct'],
            'wake_total': result_ab['total'],
            'a_word_acc': a_word_acc,
            'a_correct': sleep_result['after']['a_correct'],
            'teacher_match': teacher_match,
            'gold_match': gold_match,
            'teacher_gold_acc': sleep_result['teacher_gold_acc'],
            'ce_loss': sleep_result['best_loss'],
            'E_src_norm': a_stats['E_src_norm_mean'],
            'E_tgt_norm': a_stats['E_tgt_norm_mean'],
            'wake_ms': wake_ms,
            'dream_ms': dream_ms,
            'sleep_ms': sleep_ms,
            'cycle_ms': cycle_ms,
        }
        cycle_log.append(entry)

        # Save log incrementally
        with open(output_dir / 'cycle_log.json', 'w') as f:
            json.dump(cycle_log, f, indent=2, ensure_ascii=False)

        # Clean up dream npz
        dream_path.unlink(missing_ok=True)

    # Save final snapshot
    _save_snapshot(model, output_dir / 'final_snapshot.npz')
    print(f"\nDone. Final snapshot saved to {output_dir / 'final_snapshot.npz'}")
    print(f"Log saved to {output_dir / 'cycle_log.json'}")


if __name__ == '__main__':
    main()
