#!/usr/bin/env python3
"""Export full wake Q -> next-token score distributions from a saved snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.inspect_sleep_targets import _restore_model
from experiments.run_parallel_wake import load_corpus
from mathbrain.wake_dataset import WakeDataset


def _encode_phi_normalized(model, q_vals: np.ndarray) -> np.ndarray:
    phi_encoder = model.phi_encoder
    if hasattr(phi_encoder, 'encode_normalized_mlx'):
        phi = np.array(phi_encoder.encode_normalized_mlx(q_vals), copy=False)
    elif hasattr(phi_encoder, 'encode_normalized'):
        phi = phi_encoder.encode_normalized(q_vals)
    else:
        phi = phi_encoder.encode(q_vals)
        norms = np.linalg.norm(phi, axis=1, keepdims=True)
        phi = phi / (norms + 1e-8)
    return np.asarray(phi, dtype=np.float32)


def _build_q_distribution_dataset(model, corpus: list[str], threshold: float) -> tuple[dict, dict]:
    retina = model.retina
    rho = model.config.rho.astype(np.float32, copy=False)
    theta_q = float(model.config.THETA_Q)
    eps_q = float(model.config.EPS_Q)

    for sentence in corpus:
        for word in WakeDataset._tokenize(sentence):
            model._ensure_word(word)

    slot_universe = np.array(sorted(int(slot) for slot in model._all_vocab_slots), dtype=np.int32)
    if len(slot_universe) == 0:
        raise ValueError('No vocabulary slots found in corpus/model.')

    active_offsets = [0]
    q_offsets = [0]
    gold_offsets = [0]
    active_slots_parts: list[np.ndarray] = []
    q_vals_parts: list[np.ndarray] = []
    phi_parts: list[np.ndarray] = []
    gold_slots_parts: list[np.ndarray] = []
    teacher_scores_parts: list[np.ndarray] = []
    sent_idx_parts: list[int] = []
    pos_idx_parts: list[int] = []

    nonzero_before = []
    nonzero_after = []
    peak_before = []
    peak_after = []
    positive_after = []

    total_positions = 0
    total_negative_zeroed = 0

    for sent_idx, sentence in enumerate(corpus):
        words = WakeDataset._tokenize(sentence)
        if len(words) < 2:
            continue

        encoded = [retina.encode(word) for word in words]
        active_by_pos, q_by_pos, t_steps = WakeDataset._build_sentence_q(
            encoded[:-1], rho, theta_q, eps_q
        )
        if t_steps == 0:
            continue

        for pos in range(t_steps):
            active_slots = active_by_pos[pos]
            if len(active_slots) == 0:
                continue

            gold_slots = np.fromiter(encoded[pos + 1].keys(), dtype=np.int32)
            if len(gold_slots) == 0:
                continue

            q_vals = q_by_pos[pos].astype(np.float32, copy=False)
            phi_hat = _encode_phi_normalized(model, q_vals)

            scores_b = model.b.predict(active_slots, phi_hat)
            scores_a = model.a.predict(active_slots, phi_hat)
            teacher_scores = (scores_b + scores_a)[slot_universe].astype(np.float32, copy=False)
            thresholded = teacher_scores.copy()
            negative_zeroed = int(np.sum(thresholded < threshold))
            thresholded[thresholded < threshold] = 0.0

            active_slots_parts.append(active_slots.astype(np.int32, copy=True))
            q_vals_parts.append(q_vals.astype(np.float32, copy=True))
            phi_parts.append(phi_hat.astype(np.float32, copy=True))
            gold_slots_parts.append(gold_slots.astype(np.int32, copy=True))
            teacher_scores_parts.append(thresholded.astype(np.float32, copy=False))
            active_offsets.append(active_offsets[-1] + len(active_slots))
            q_offsets.append(q_offsets[-1] + len(active_slots))
            gold_offsets.append(gold_offsets[-1] + len(gold_slots))
            sent_idx_parts.append(int(sent_idx))
            pos_idx_parts.append(int(pos))

            nonzero_before.append(int(np.sum(np.abs(teacher_scores) > 0.0)))
            nonzero_after.append(int(np.sum(thresholded > 0.0)))
            peak_before.append(float(np.max(teacher_scores)))
            peak_after.append(float(np.max(thresholded)))
            positive_after.append(float(np.sum(thresholded > 0.0)))
            total_negative_zeroed += negative_zeroed
            total_positions += 1

    if total_positions == 0:
        raise ValueError('No valid Q -> distribution samples were built.')

    data = {
        'slot_universe': slot_universe,
        'active_offsets': np.asarray(active_offsets, dtype=np.int64),
        'active_slots': np.concatenate(active_slots_parts, axis=0).astype(np.int32, copy=False),
        'q_offsets': np.asarray(q_offsets, dtype=np.int64),
        'q_vals': np.concatenate(q_vals_parts, axis=0).astype(np.float32, copy=False),
        'phi': np.concatenate(phi_parts, axis=0).astype(np.float32, copy=False),
        'gold_offsets': np.asarray(gold_offsets, dtype=np.int64),
        'gold_slots': np.concatenate(gold_slots_parts, axis=0).astype(np.int32, copy=False),
        'teacher_scores': np.stack(teacher_scores_parts, axis=0).astype(np.float32, copy=False),
        'sent_idx': np.asarray(sent_idx_parts, dtype=np.int32),
        'pos_idx': np.asarray(pos_idx_parts, dtype=np.int32),
    }

    info = {
        'threshold': float(threshold),
        'total_positions': int(total_positions),
        'slot_universe_size': int(len(slot_universe)),
        'teacher_nonzero_before_mean': float(np.mean(nonzero_before)),
        'teacher_nonzero_after_mean': float(np.mean(nonzero_after)),
        'teacher_nonzero_after_max': int(np.max(nonzero_after)),
        'teacher_peak_before_mean': float(np.mean(peak_before)),
        'teacher_peak_after_mean': float(np.mean(peak_after)),
        'positive_after_mean': float(np.mean(positive_after)),
        'total_negative_or_subthreshold_zeroed': int(total_negative_zeroed),
    }
    return data, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, type=str)
    parser.add_argument('--corpus', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--threshold', type=float, default=0.01)
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, _ = _restore_model(snapshot_path)
    corpus = load_corpus(args.corpus)
    data, info = _build_q_distribution_dataset(model, corpus, threshold=float(args.threshold))

    np.savez_compressed(output_path, **data)

    meta = {
        'snapshot': str(snapshot_path),
        'corpus': str(Path(args.corpus)),
        **info,
    }
    with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved dream dataset: {output_path}")
    print(
        f"  positions={info['total_positions']}, slots={info['slot_universe_size']}, "
        f"threshold={info['threshold']:.4f}, nonzero_after_mean={info['teacher_nonzero_after_mean']:.2f}, "
        f"peak_after_mean={info['teacher_peak_after_mean']:.4f}"
    )


if __name__ == '__main__':
    main()
