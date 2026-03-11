#!/usr/bin/env python3
"""Train A from full Q -> score-distribution targets."""

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

try:
    import mlx.core as mx
    import mlx.optimizers as optim
except Exception as exc:  # pragma: no cover
    raise RuntimeError("MLX is required for this script") from exc

from experiments.inspect_sleep_targets import _restore_model
from experiments.run_parallel_wake import load_corpus
from mathbrain.inference_fast import SparseMatrixDecoder
from mathbrain.wake_dataset import WakeDataset


def _ensure_vocab(model, corpus: list[str]):
    for sentence in corpus:
        for word in WakeDataset._tokenize(sentence):
            model._ensure_word(word)


def _prepare_dataset(model, dataset_npz: np.lib.npyio.NpzFile) -> dict:
    slot_universe = dataset_npz['slot_universe'].astype(np.int32, copy=False)
    slot_to_local = {int(slot): idx for idx, slot in enumerate(slot_universe.tolist())}

    active_offsets = dataset_npz['active_offsets'].astype(np.int64, copy=False)
    active_slots = dataset_npz['active_slots'].astype(np.int32, copy=False)
    q_offsets = dataset_npz['q_offsets'].astype(np.int64, copy=False)
    q_vals = dataset_npz['q_vals'].astype(np.float32, copy=False)
    gold_offsets = dataset_npz['gold_offsets'].astype(np.int64, copy=False)
    gold_slots = dataset_npz['gold_slots'].astype(np.int32, copy=False)
    teacher_scores = dataset_npz['teacher_scores'].astype(np.float32, copy=False)
    sent_idx = dataset_npz['sent_idx'].astype(np.int32, copy=False)
    pos_idx = dataset_npz['pos_idx'].astype(np.int32, copy=False)

    phi_flat = model.phi_encoder.encode_normalized_mlx(q_vals)
    phi_flat = np.asarray(phi_flat, dtype=np.float32)

    n_pos = int(teacher_scores.shape[0])
    width = int(max(active_offsets[i + 1] - active_offsets[i] for i in range(n_pos)))
    q_dim = int(q_vals.shape[1])
    phi_dim = int(phi_flat.shape[1])

    active_local = np.zeros((n_pos, width), dtype=np.int32)
    q_padded = np.zeros((n_pos, width, q_dim), dtype=np.float32)
    phi_padded = np.zeros((n_pos, width, phi_dim), dtype=np.float32)
    active_mask = np.zeros((n_pos, width), dtype=np.float32)
    active_count = np.zeros((n_pos,), dtype=np.int32)

    gold_slot_lists: list[np.ndarray] = []
    for idx in range(n_pos):
        a0, a1 = int(active_offsets[idx]), int(active_offsets[idx + 1])
        q0, q1 = int(q_offsets[idx]), int(q_offsets[idx + 1])
        g0, g1 = int(gold_offsets[idx]), int(gold_offsets[idx + 1])
        slots = active_slots[a0:a1]
        q_rows = q_vals[q0:q1]
        phi_rows = phi_flat[q0:q1]
        local_rows = np.array([slot_to_local[int(slot)] for slot in slots.tolist()], dtype=np.int32)
        count = int(len(local_rows))
        if count > 0:
            active_local[idx, :count] = local_rows
            q_padded[idx, :count] = q_rows
            phi_padded[idx, :count] = phi_rows
            active_mask[idx, :count] = 1.0
            active_count[idx] = count
        gold_slot_lists.append(gold_slots[g0:g1].astype(np.int32, copy=True))

    return {
        'slot_universe': slot_universe,
        'active_local': active_local,
        'q_vals': q_padded,
        'phi': phi_padded,
        'active_mask': active_mask,
        'active_count': active_count,
        'teacher_scores': teacher_scores,
        'gold_slot_lists': gold_slot_lists,
        'sent_idx': sent_idx,
        'pos_idx': pos_idx,
    }


def _extract_student_factors(model, slot_universe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = int(model.a.d)
    D = int(model.a.D)
    E_src = np.zeros((len(slot_universe), d, D), dtype=np.float32)
    E_tgt = np.zeros((len(slot_universe), d, D), dtype=np.float32)
    for idx, slot in enumerate(slot_universe.tolist()):
        E_src[idx] = model.a.E_src[int(slot)]
        E_tgt[idx] = model.a.E_tgt[int(slot)]
    return E_src, E_tgt


def _batched_student_logits_numpy(E_src: np.ndarray,
                                  E_tgt: np.ndarray,
                                  active_local: np.ndarray,
                                  phi: np.ndarray,
                                  active_mask: np.ndarray,
                                  batch_size: int) -> np.ndarray:
    n_pos = int(active_local.shape[0])
    n_slots = int(E_tgt.shape[0])
    out = np.zeros((n_pos, n_slots), dtype=np.float32)
    E_tgt_flat = E_tgt.reshape(n_slots, -1)
    d_rank = max(1, int(E_tgt.shape[1]))

    for start in range(0, n_pos, batch_size):
        end = min(start + batch_size, n_pos)
        batch_active = active_local[start:end]
        batch_phi = phi[start:end]
        batch_mask = active_mask[start:end]
        src_emb = E_src[batch_active]
        ctx = np.sum(src_emb * batch_phi[:, :, None, :] * batch_mask[:, :, None, None], axis=1)
        ctx_flat = ctx.reshape(end - start, -1)
        active_count = np.sum(batch_mask, axis=1, keepdims=True) + 1e-8
        out[start:end] = (ctx_flat @ E_tgt_flat.T) / active_count
    return out


def _distribution_metrics(student_logits: np.ndarray,
                          teacher_scores: np.ndarray,
                          zero_threshold: float) -> dict:
    diff = student_logits - teacher_scores
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    teacher_top1 = np.argmax(teacher_scores, axis=1)
    student_top1 = np.argmax(student_logits, axis=1)
    top1_match = float(np.mean(teacher_top1 == student_top1))

    teacher_rank = []
    cosine = []
    zero_mask = teacher_scores <= 0.0
    zero_violation = student_logits[zero_mask] > float(zero_threshold)
    zero_pred_mean = float(np.mean(np.abs(student_logits[zero_mask]))) if np.any(zero_mask) else 0.0
    zero_violation_rate = float(np.mean(zero_violation)) if zero_violation.size > 0 else 0.0

    for idx in range(student_logits.shape[0]):
        teacher_t1 = teacher_top1[idx]
        teacher_rank.append(1 + int(np.sum(student_logits[idx] > student_logits[idx, teacher_t1])))
        denom = (np.linalg.norm(student_logits[idx]) * np.linalg.norm(teacher_scores[idx])) + 1e-8
        cosine.append(float(np.dot(student_logits[idx], teacher_scores[idx]) / denom))

    return {
        'mse': mse,
        'rmse': rmse,
        'cosine_mean': float(np.mean(cosine)),
        'top1_match': top1_match,
        'teacher_top1_rank_mean': float(np.mean(teacher_rank)),
        'student_peak_mean': float(np.mean(np.max(student_logits, axis=1))),
        'teacher_peak_mean': float(np.mean(np.max(teacher_scores, axis=1))),
        'student_absmax': float(np.max(np.abs(student_logits))),
        'teacher_absmax': float(np.max(np.abs(teacher_scores))),
        'zero_pred_abs_mean': zero_pred_mean,
        'zero_violation_rate': zero_violation_rate,
    }


def _evaluate_a_only_accuracy(model,
                              slot_universe: np.ndarray,
                              active_local: np.ndarray,
                              phi: np.ndarray,
                              active_mask: np.ndarray,
                              gold_slot_lists: list[np.ndarray]) -> dict:
    if model._decoder_dirty or model._decoder is None:
        model._decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)
        model._decoder_dirty = False
    decoder = model._decoder

    correct_word = 0
    correct_slot = 0
    total = len(gold_slot_lists)

    for idx in range(total):
        count = int(np.sum(active_mask[idx]))
        active_slots = slot_universe[active_local[idx, :count]]
        phi_rows = phi[idx, :count]
        scores_a = model.a.predict(active_slots.astype(np.int32, copy=False), phi_rows.astype(np.float32, copy=False))
        pred_slot = int(np.argmax(scores_a))
        gold_slots = gold_slot_lists[idx]
        correct_slot += int(pred_slot in set(gold_slots.tolist()))

        target_score = np.zeros((model.config.K,), dtype=np.float32)
        target_score[gold_slots] = 1.0
        gold_word = decoder.decode_top_k(target_score, 1)[0][0]
        pred_word = decoder.decode_top_k(scores_a, 1)[0][0]
        correct_word += int(pred_word == gold_word)

    return {
        'slot_top1_acc': float(correct_slot / max(1, total)),
        'word_acc': float(correct_word / max(1, total)),
        'correct_word': int(correct_word),
        'total': int(total),
    }


def _save_snapshot(model, output_path: Path):
    a = model.a
    slots = np.array(sorted(a.E_src.keys()), dtype=np.int32)
    a_E_src = np.stack([a.E_src[int(slot)] for slot in slots], axis=0).astype(np.float32, copy=False)
    a_E_tgt = np.stack([a.E_tgt[int(slot)] for slot in slots], axis=0).astype(np.float32, copy=False)

    b = model.b
    n_entries = int(getattr(b, '_n_entries', 0))
    if n_entries > 0:
        b_src = np.array(b._srcs[:n_entries], copy=True)
        b_tgt = np.array(b._tgts[:n_entries], copy=True)
        b_vec = np.array(b._vecs[:n_entries], copy=True)
        b_time = np.array(b._times[:n_entries], copy=True)
    else:
        b_src = np.empty((0,), dtype=np.int32)
        b_tgt = np.empty((0,), dtype=np.int32)
        b_vec = np.empty((0, model.config.d), dtype=np.float32)
        b_time = np.empty((0,), dtype=np.int32)

    np.savez_compressed(
        output_path,
        a_slots=slots,
        a_E_src=a_E_src,
        a_E_tgt=a_E_tgt,
        b_src=b_src,
        b_tgt=b_tgt,
        b_vec=b_vec,
        b_time=b_time,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, type=str)
    parser.add_argument('--dream-dataset', required=True, type=str)
    parser.add_argument('--corpus', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--zero-threshold', type=float, default=0.01)
    parser.add_argument('--topk', type=int, default=0,
                        help='Top-K sparse loss: only compute MSE on teacher top-K slots per position. 0=all.')
    parser.add_argument('--topk-weight', type=float, default=1.0,
                        help='Extra weight multiplier for top-K slots. Final weight = 1 + topk_weight for top-K, 1 for background.')
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    dream_dataset_path = Path(args.dream_dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, _ = _restore_model(snapshot_path)
    corpus = load_corpus(args.corpus)
    _ensure_vocab(model, corpus)

    dataset_npz = np.load(dream_dataset_path)
    data = _prepare_dataset(model, dataset_npz)
    slot_universe = data['slot_universe']
    active_local = data['active_local']
    phi = data['phi']
    active_mask = data['active_mask']
    teacher_scores = data['teacher_scores']
    gold_slot_lists = data['gold_slot_lists']

    # Top-K weighted mask 预计算
    topk = int(args.topk)
    topk_weight = float(args.topk_weight)
    n_slots = int(teacher_scores.shape[1])
    if topk > 0 and topk < n_slots:
        # 混合权重：background=1, top-K=1+topk_weight
        topk_mask = np.ones_like(teacher_scores, dtype=np.float32)
        for i in range(teacher_scores.shape[0]):
            row = teacher_scores[i]
            indices = np.argpartition(row, -topk)[-topk:]
            topk_mask[i, indices] += topk_weight
        print(f"Top-K weighted loss: K={topk}, weight={topk_weight}, "
              f"top-K weight={1+topk_weight}, background weight=1")
    else:
        topk_mask = np.ones_like(teacher_scores, dtype=np.float32)
        topk = 0

    E_src_np, E_tgt_np = _extract_student_factors(model, slot_universe)
    init_src_np = E_src_np.copy()
    init_tgt_np = E_tgt_np.copy()

    before_logits = _batched_student_logits_numpy(
        E_src_np, E_tgt_np, active_local, phi, active_mask, batch_size=max(1, int(args.batch_size))
    )
    before_dist = _distribution_metrics(before_logits, teacher_scores, zero_threshold=float(args.zero_threshold))
    before_acc = _evaluate_a_only_accuracy(model, slot_universe, active_local, phi, active_mask, gold_slot_lists)

    params = {
        'E_src': mx.array(E_src_np),
        'E_tgt': mx.array(E_tgt_np),
    }
    active_local_mx = mx.array(active_local)
    phi_mx = mx.array(phi)
    active_mask_mx = mx.array(active_mask)
    teacher_mx = mx.array(teacher_scores)
    init_src_mx = mx.array(init_src_np)
    init_tgt_mx = mx.array(init_tgt_np)
    topk_mask_mx = mx.array(topk_mask)

    n_pos = int(active_local.shape[0])
    batch_size = max(1, int(args.batch_size))
    n_batches = max(1, int(np.ceil(n_pos / batch_size)))
    lr_schedule = optim.cosine_decay(init=float(args.lr), decay_steps=max(1, int(args.epochs) * n_batches))
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=float(args.weight_decay))

    d_rank = max(1, int(model.a.d))

    def _loss_fn(params, batch_active, batch_phi, batch_mask, batch_teacher, batch_topk_mask):
        src_emb = params['E_src'][batch_active]
        ctx = mx.sum(src_emb * batch_phi[:, :, None, :] * batch_mask[:, :, None, None], axis=1)
        ctx_flat = mx.reshape(ctx, (int(ctx.shape[0]), -1))
        E_tgt_flat = mx.reshape(params['E_tgt'], (int(params['E_tgt'].shape[0]), -1))
        logits_raw = mx.matmul(ctx_flat, mx.transpose(E_tgt_flat))
        active_count = mx.sum(batch_mask, axis=1, keepdims=True) + 1e-8
        logits = logits_raw / active_count
        diff = (logits - batch_teacher) ** 2
        masked_diff = diff * batch_topk_mask
        return mx.sum(masked_diff) / (mx.sum(batch_topk_mask) + 1e-8)

    loss_and_grad_fn = mx.value_and_grad(_loss_fn)

    best_loss = float('inf')
    best_src = E_src_np.copy()
    best_tgt = E_tgt_np.copy()
    loss_history = []

    t0 = time.time()
    for epoch in range(int(args.epochs)):
        perm = np.random.permutation(n_pos).astype(np.int32, copy=False)
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_pos)
            batch_ids_np = perm[start:end]
            batch_ids = mx.array(batch_ids_np)
            batch_active = active_local_mx[batch_ids]
            batch_phi = phi_mx[batch_ids]
            batch_mask = active_mask_mx[batch_ids]
            batch_teacher = teacher_mx[batch_ids]
            batch_topk = topk_mask_mx[batch_ids]
            loss, grads = loss_and_grad_fn(params, batch_active, batch_phi, batch_mask, batch_teacher, batch_topk)
            optimizer.update(params, grads)
            mx.eval(loss, params)
            epoch_loss += float(loss)

        epoch_loss /= float(n_batches)
        loss_history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_src = np.array(params['E_src'])
            best_tgt = np.array(params['E_tgt'])

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == int(args.epochs) - 1:
            print(f"epoch {epoch + 1:03d}: loss={epoch_loss:.8f}")

    train_ms = (time.time() - t0) * 1000.0

    for idx, slot in enumerate(slot_universe.tolist()):
        model.a.E_src[int(slot)] = best_src[idx].astype(np.float32, copy=True)
        model.a.E_tgt[int(slot)] = best_tgt[idx].astype(np.float32, copy=True)
    model.a.has_knowledge = True
    model.a._build_cache()

    after_logits = _batched_student_logits_numpy(
        best_src, best_tgt, active_local, phi, active_mask, batch_size=max(1, int(args.batch_size))
    )
    after_dist = _distribution_metrics(after_logits, teacher_scores, zero_threshold=float(args.zero_threshold))
    after_acc = _evaluate_a_only_accuracy(model, slot_universe, active_local, phi, active_mask, gold_slot_lists)

    _save_snapshot(model, output_path)
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'snapshot': str(snapshot_path),
            'dream_dataset': str(dream_dataset_path),
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'zero_threshold': float(args.zero_threshold),
            'topk': topk,
            'topk_weight': topk_weight,
            'train_ms': train_ms,
            'best_loss': float(best_loss),
            'before_distribution': before_dist,
            'after_distribution': after_dist,
            'before_a_only_accuracy': before_acc,
            'after_a_only_accuracy': after_acc,
            'loss_history': loss_history,
        }, f, ensure_ascii=False, indent=2)

    print("Distribution consistency (before)")
    print(json.dumps(before_dist, ensure_ascii=False, indent=2))
    print("Distribution consistency (after)")
    print(json.dumps(after_dist, ensure_ascii=False, indent=2))
    print("A-only accuracy (before)")
    print(json.dumps(before_acc, ensure_ascii=False, indent=2))
    print("A-only accuracy (after)")
    print(json.dumps(after_acc, ensure_ascii=False, indent=2))
    print(f"Saved trained snapshot: {output_path}")


if __name__ == '__main__':
    main()
