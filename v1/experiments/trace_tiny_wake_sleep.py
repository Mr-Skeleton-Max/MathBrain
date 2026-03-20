#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.inference_fast import SparseMatrixDecoder
from mathbrain.sleep_dream import prepare_dream_distillation_data
from mathbrain.wake_dataset import WakeDataset
from mathbrain.wake_tree_trainer import TreeWakeTrainer

try:
    from mathbrain.wake_tree_trainer_mlx import TreeWakeTrainerMLX
    HAS_WAKE_MLX = True
except Exception:
    HAS_WAKE_MLX = False


DEFAULT_CORPUS = [
    "red blue red blue",
    "red green red green",
    "blue red blue red",
    "green red green red",
]


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(values)))
    if k >= len(values):
        return np.argsort(values)[::-1].astype(np.int32, copy=False)
    part = np.argpartition(values, -k)[-k:]
    return part[np.argsort(values[part])[::-1]].astype(np.int32, copy=False)


def build_model(args) -> MathBrain:
    cfg = MathBrainConfig()
    cfg.K = args.k
    cfg.NGRAM_SIZE = args.ngram
    cfg.NGRAM_SCALES = (args.ngram,)
    cfg.PHI_MODE = "chaos"
    cfg.D_PHI = args.d_phi
    cfg.CHAOS_N_FOLDS = args.chaos_folds
    cfg.CHAOS_ALPHA = args.chaos_alpha
    cfg.CP_RANK = args.cp_rank
    cfg.ETA = args.eta
    cfg.LAMBDA_WD = args.lambda_wd
    cfg.SLEEP_SOLVER = "dream"
    cfg.SLEEP_GLOBAL_DECAY = args.sleep_global_decay
    cfg.GAMMA_DECAY = args.gamma_decay
    cfg.GAMMA_ADAPTIVE = False
    cfg.SLEEP_DREAM_SAMPLES = args.dream_samples
    cfg.SLEEP_DREAM_ACTIVE = args.dream_group
    cfg.SLEEP_DREAM_TOPK = args.dream_topk
    cfg.SLEEP_DREAM_PHASES = args.dream_waves
    cfg.SLEEP_DREAM_BATCH_SIZE = args.dream_batch
    cfg.SLEEP_DREAM_EPOCHS = args.sleep_epochs
    cfg.SLEEP_DREAM_TEMPERATURE = args.dream_temp
    cfg.SLEEP_DREAM_PROX = args.dream_prox
    cfg.SLEEP_DREAM_LOGIT_WEIGHT = 1.0
    cfg.SLEEP_DREAM_KL_WEIGHT = 0.0
    cfg.SLEEP_DREAM_EPISODE_LEN = args.dream_episode
    cfg.SLEEP_DREAM_SEQ_MULTIPLIER = args.dream_seq_multiplier
    cfg.SLEEP_DREAM_PROBE_COUNT = 1
    cfg.SLEEP_DREAM_UNIFORM_MIX = args.dream_uniform_mix
    cfg.SLEEP_DREAM_SEQUENCE_SOURCE = args.dream_sequence_source
    cfg.SLEEP_LR = args.sleep_lr
    cfg.SLEEP_MAX_EPOCHS = args.sleep_epochs
    cfg.SLEEP_MIN_EPOCHS = 0
    cfg.SLEEP_PATIENCE = args.sleep_epochs
    cfg.SLEEP_REL_TOL = 0.0
    cfg.SLEEP_TRACE_FULL_IO = bool(args.full_io)
    cfg.SLEEP_TRACE_MAX_CONTEXTS = int(args.trace_max_contexts)
    cfg._build_derived()
    return MathBrain(cfg)


def build_trainer(model: MathBrain, backend: str):
    if backend == "mlx" and HAS_WAKE_MLX:
        return TreeWakeTrainerMLX(model)
    return TreeWakeTrainer(model)


def ensure_vocab(model: MathBrain, corpus: list[str]):
    for sentence in corpus:
        for word in WakeDataset._tokenize(sentence):
            model._ensure_word(word)


def get_decoder(model: MathBrain):
    if model._decoder_dirty or model._decoder is None:
        model._decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)
        model._decoder_dirty = False
    return model._decoder


def slot_to_words(model: MathBrain) -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}
    for word, slots in sorted(model.word_to_slots.items()):
        for slot in slots.tolist():
            mapping.setdefault(int(slot), []).append(word)
    return mapping


def format_slot(slot: int, slot_words: dict[int, list[str]]) -> str:
    tags = "/".join(slot_words.get(int(slot), []))
    return f"{int(slot)}[{tags}]"


def tokenize_positions(model: MathBrain, corpus: list[str]) -> list[dict]:
    positions: list[dict] = []
    rho = model.config.rho.astype(np.float32, copy=False)
    theta_q = float(model.config.THETA_Q)
    eps_q = float(model.config.EPS_Q)
    phi_encoder = model.phi_encoder
    for sent_idx, sentence in enumerate(corpus):
        words = WakeDataset._tokenize(sentence)
        if len(words) < 2:
            continue
        encoded = [model.retina.encode(word) for word in words]
        active_by_pos, q_by_pos, t_steps = WakeDataset._build_sentence_q(encoded[:-1], rho, theta_q, eps_q)
        for pos_idx in range(t_steps):
            active_slots = active_by_pos[pos_idx]
            q_vals = q_by_pos[pos_idx]
            if len(active_slots) == 0:
                continue
            if hasattr(phi_encoder, "encode_normalized"):
                phi_hat = phi_encoder.encode_normalized(q_vals)
            else:
                phi = phi_encoder.encode(q_vals)
                phi_hat = phi / (np.linalg.norm(phi, axis=1, keepdims=True) + 1e-8)
            positions.append({
                "sentence_idx": sent_idx,
                "position_idx": pos_idx,
                "context_words": words[:pos_idx + 1],
                "source_word": words[pos_idx],
                "target_word": words[pos_idx + 1],
                "active_slots": active_slots.astype(np.int32, copy=False),
                "q_vals": q_vals.astype(np.float32, copy=False),
                "phi_hat": phi_hat.astype(np.float32, copy=False),
                "target_slots": np.fromiter(encoded[pos_idx + 1].keys(), dtype=np.int32),
            })
    return positions


def evaluate_parallel(model: MathBrain, dataset, trainer) -> dict:
    if hasattr(dataset, "active_offsets"):
        correct = 0
        total = 0
        decoder = get_decoder(model)
        for pos in range(dataset.total_positions):
            a0, a1 = int(dataset.active_offsets[pos]), int(dataset.active_offsets[pos + 1])
            t0, t1 = int(dataset.target_offsets[pos]), int(dataset.target_offsets[pos + 1])
            active = dataset.active_slots[a0:a1]
            phi = dataset.phi_hat[a0:a1]
            gold_slots = dataset.target_slots[t0:t1]
            scores = model.predict_slots(active, phi)
            pred_word = decoder.decode_top_k(scores, 1)[0][0]
            target_score = np.zeros((model.config.K,), dtype=np.float32)
            target_score[gold_slots] = 1.0
            gold_word = decoder.decode_top_k(target_score, 1)[0][0]
            correct += int(pred_word == gold_word)
            total += 1
        return {"accuracy": (100.0 * correct / total) if total else 0.0, "correct": correct, "total": total}
    return {"accuracy": 0.0, "correct": 0, "total": 0}


def compute_positive_edge_table(model: MathBrain, positions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    step = int(model.b.global_step + 1)
    if model.a.has_knowledge:
        model.b._precompute_a_embeddings(model.a)
    for item in positions:
        active = item["active_slots"]
        targets = item["target_slots"]
        phi_hat = item["phi_hat"]
        n_src = len(active)
        n_tgt = len(targets)
        if n_src == 0 or n_tgt == 0:
            continue
        all_src = np.repeat(active, n_tgt)
        all_tgt = np.tile(targets, n_src)
        phi_rep = np.repeat(phi_hat, n_tgt, axis=0)
        b_strength = model.b._compute_b_strength_vectorized(all_src, all_tgt, phi_rep, step)
        a_strength = model.b._compute_a_strength_vectorized(all_src, all_tgt, phi_rep)
        a_strength = a_strength / float(max(1, model.a.d))
        gap = 1.0 - (a_strength + b_strength)
        for idx in range(len(all_src)):
            rows.append({
                "context_words": item["context_words"],
                "target_word": item["target_word"],
                "src": int(all_src[idx]),
                "tgt": int(all_tgt[idx]),
                "a_strength": float(a_strength[idx]),
                "b_strength": float(b_strength[idx]),
                "gap": float(gap[idx]),
            })
    return rows


def print_positive_edge_summary(label: str, rows: list[dict], slot_words: dict[int, list[str]]):
    print(label)
    if not rows:
        print("  <empty>")
        print()
        return
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        src_word = slot_words.get(int(row["src"]), ["?"])[0]
        tgt_word = slot_words.get(int(row["tgt"]), ["?"])[0]
        key = (src_word, tgt_word)
        stats = grouped.setdefault(key, {"count": 0.0, "a_sum": 0.0, "b_sum": 0.0, "gap_sum": 0.0})
        stats["count"] += 1.0
        stats["a_sum"] += row["a_strength"]
        stats["b_sum"] += row["b_strength"]
        stats["gap_sum"] += row["gap"]
    for (src_word, tgt_word), stats in sorted(grouped.items()):
        count = max(1.0, stats["count"])
        print(
            f"  {src_word} -> {tgt_word}: "
            f"A_mean={stats['a_sum'] / count:.4f} "
            f"B_mean={stats['b_sum'] / count:.4f} "
            f"gap_mean={stats['gap_sum'] / count:.4f} "
            f"n={int(count)}"
        )
    print()


def print_positive_edge_table(label: str, rows: list[dict], slot_words: dict[int, list[str]]):
    print(label)
    if not rows:
        print("  <empty>")
        print()
        return
    for row in rows:
        ctx = " ".join(row["context_words"])
        print(
            f"  ctx='{ctx}' -> {row['target_word']}: "
            f"{format_slot(row['src'], slot_words)} -> {format_slot(row['tgt'], slot_words)} "
            f"A={row['a_strength']:.4f} B={row['b_strength']:.4f} gap={row['gap']:.4f}"
        )
    print()


def print_b_entries(model: MathBrain, slot_words_map: dict[int, list[str]], label: str = "Wake B entries"):
    entries = sorted(model.b.get_entries(), key=lambda item: (int(item[0]), int(item[1])))
    print(label)
    if not entries:
        print("  <empty>")
        print()
        return
    for src, tgt, vec in entries:
        print(
            f"  {format_slot(src, slot_words_map)} -> {format_slot(tgt, slot_words_map)} "
            f"norm={np.linalg.norm(vec):.4f} vec={np.array2string(vec, precision=4, suppress_small=False)}"
        )
    print()


def capture_b_edges(model: MathBrain) -> dict[tuple[int, int], np.ndarray]:
    return {
        (int(src), int(tgt)): np.asarray(vec, dtype=np.float32).copy()
        for src, tgt, vec in model.b.get_entries()
    }


def print_position_inputs(label: str, positions: list[dict], slot_words_map: dict[int, list[str]]):
    print(label)
    for idx, item in enumerate(positions):
        print(
            f"  pos[{idx}] sent={item['sentence_idx']} local_pos={item['position_idx']} "
            f"ctx='{' '.join(item['context_words'])}' src='{item['source_word']}' tgt='{item['target_word']}'"
        )
        for row_idx, slot in enumerate(item["active_slots"].tolist()):
            print(
                f"    active[{row_idx}] {format_slot(slot, slot_words_map)} "
                f"q={np.array2string(item['q_vals'][row_idx], precision=4, suppress_small=False)} "
                f"phi_hat={np.array2string(item['phi_hat'][row_idx], precision=4, suppress_small=False)}"
            )
        print(
            "    target slots:",
            ", ".join(format_slot(slot, slot_words_map) for slot in item["target_slots"].tolist())
        )
    print()


def print_a_factors(label: str, model: MathBrain, slots: list[int], slot_words_map: dict[int, list[str]]):
    print(label)
    if not model.a.has_knowledge:
        print("  <A empty>")
        print()
        return
    for slot in slots:
        src = model.a.E_src.get(int(slot))
        tgt = model.a.E_tgt.get(int(slot))
        print(f"  slot {format_slot(slot, slot_words_map)}")
        print(f"    E_src={np.array2string(src, precision=4, suppress_small=False)}")
        print(f"    E_tgt={np.array2string(tgt, precision=4, suppress_small=False)}")
    print()


def candidate_logits_from_a(model: MathBrain,
                            all_slots: np.ndarray,
                            active_local: np.ndarray,
                            phi_rows: np.ndarray,
                            candidate_local: np.ndarray) -> np.ndarray:
    if not model.a.has_knowledge:
        return np.zeros((len(candidate_local),), dtype=np.float32)
    active_slots = all_slots[active_local]
    phi_norm = np.linalg.norm(phi_rows, axis=1, keepdims=True) + 1e-8
    phi_hat = phi_rows / phi_norm
    src_emb = np.stack([model.a.E_src[int(slot)] for slot in active_slots.tolist()], axis=0)
    ctx = np.sum(src_emb * phi_hat[:, None, :], axis=0)
    tgt_emb = np.stack([model.a.E_tgt[int(all_slots[int(idx)])] for idx in candidate_local.tolist()], axis=0)
    logits_raw = np.sum(tgt_emb * ctx[None, :, :], axis=(1, 2))
    return (logits_raw / float(max(1, len(active_slots)) * max(1, model.a.d))).astype(np.float32, copy=False)


def slot_scores_to_word_scores(model: MathBrain, scores: np.ndarray) -> list[tuple[str, float]]:
    return get_decoder(model).decode_top_k(scores, min(8, max(1, len(model.vocab))))


def print_wake_positions(label: str,
                         model: MathBrain,
                         positions: list[dict],
                         slot_words_map: dict[int, list[str]],
                         tracked_slots: np.ndarray | None = None):
    print(label)
    decoder = get_decoder(model)
    if tracked_slots is None:
        tracked_slots = np.asarray(sorted(slot_words_map.keys()), dtype=np.int32)
    for item in positions:
        scores_b = model.b.predict(item["active_slots"], item["phi_hat"])
        scores_a = model.a.predict(item["active_slots"], item["phi_hat"])
        merged = scores_a + scores_b
        target_score = np.zeros((model.config.K,), dtype=np.float32)
        target_score[item["target_slots"]] = 1.0
        gold_word = decoder.decode_top_k(target_score, 1)[0][0]
        print(
            f"  sent={item['sentence_idx']} pos={item['position_idx']} "
            f"ctx='{' '.join(item['context_words'])}' gold='{gold_word}'"
        )
        print("    active:", ", ".join(format_slot(slot, slot_words_map) for slot in item["active_slots"].tolist()))
        print("    top merged:", slot_scores_to_word_scores(model, merged))
        print("    top A    :", slot_scores_to_word_scores(model, scores_a))
        print("    top B    :", slot_scores_to_word_scores(model, scores_b))
        print("    merged tracked:")
        for slot in tracked_slots.tolist():
            print(f"      {format_slot(slot, slot_words_map)} -> {merged[int(slot)]:.4f}")
        print("    A tracked:")
        for slot in tracked_slots.tolist():
            print(f"      {format_slot(slot, slot_words_map)} -> {scores_a[int(slot)]:.4f}")
        print("    B tracked:")
        for slot in tracked_slots.tolist():
            print(f"      {format_slot(slot, slot_words_map)} -> {scores_b[int(slot)]:.4f}")
    print()


def print_word_slots(model: MathBrain, slot_words_map: dict[int, list[str]]):
    print("Word -> slots")
    for word in sorted(model.vocab):
        slots = model.word_to_slots[word]
        print(f"  {word}: {', '.join(format_slot(slot, slot_words_map) for slot in slots.tolist())}")
    print()


def capture_a_edges(model: MathBrain, slots: list[int]) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
    for src in slots:
        for tgt in slots:
            vec = model.a.compute_entry(int(src), int(tgt)).astype(np.float32, copy=False)
            if np.max(np.abs(vec)) > 1e-8:
                out[(int(src), int(tgt))] = vec.copy()
    return out


def print_a_edge_delta(before: dict[tuple[int, int], np.ndarray],
                       after: dict[tuple[int, int], np.ndarray],
                       slot_words_map: dict[int, list[str]]):
    keys = sorted(set(before.keys()) | set(after.keys()))
    print("A edge delta on involved slots")
    if not keys:
        print("  <empty>")
        print()
        return
    for key in keys:
        src, tgt = key
        vec_before = before.get(key, np.zeros_like(next(iter(after.values()))) if after else np.zeros((0,), dtype=np.float32))
        vec_after = after.get(key, np.zeros_like(next(iter(before.values()))) if before else np.zeros((0,), dtype=np.float32))
        if vec_before.shape != vec_after.shape:
            size = max(vec_before.shape[0] if vec_before.ndim else 0, vec_after.shape[0] if vec_after.ndim else 0)
            if vec_before.shape != (size,):
                vec_before = np.zeros((size,), dtype=np.float32)
            if vec_after.shape != (size,):
                vec_after = np.zeros((size,), dtype=np.float32)
        delta = vec_after - vec_before
        print(
            f"  {format_slot(src, slot_words_map)} -> {format_slot(tgt, slot_words_map)} "
            f"before={np.array2string(vec_before, precision=4, suppress_small=False)} "
            f"after={np.array2string(vec_after, precision=4, suppress_small=False)} "
            f"delta={np.array2string(delta, precision=4, suppress_small=False)}"
        )
    print()


def build_dream_bundle(model: MathBrain, corpus: list[str], n_dreams: int, topk: int, debug_trace: bool = False):
    entries = model.b.get_entries()
    involved_slots = sorted({int(src) for src, _, _ in entries} | {int(tgt) for _, tgt, _ in entries} | {int(slot) for slots in model.word_to_slots.values() for slot in slots.tolist()})
    candidate_topk = max(1, min(int(topk), len(involved_slots))) if involved_slots else 1
    return prepare_dream_distillation_data(
        b_memory=model.b,
        entries=entries,
        all_slots=involved_slots,
        config=model.config,
        a_knowledge=model.a,
        global_decay=float(model.config.SLEEP_GLOBAL_DECAY),
        retina=model.retina,
        vocab_words=sorted(model.vocab),
        n_dreams=n_dreams,
        dream_size=int(model.config.SLEEP_DREAM_ACTIVE),
        candidate_topk=candidate_topk,
        n_waves=int(model.config.SLEEP_DREAM_PHASES),
        episode_len=int(model.config.SLEEP_DREAM_EPISODE_LEN),
        probe_count=int(model.config.SLEEP_DREAM_PROBE_COUNT),
        uniform_mix=float(model.config.SLEEP_DREAM_UNIFORM_MIX),
        rng_seed=0,
        dream_corpus=list(corpus),
        verbose=False,
        debug_trace=debug_trace,
    )


def print_parallel_wake_round(round_no: int,
                              trainer,
                              dataset,
                              positions: list[dict],
                              slot_words_map: dict[int, list[str]],
                              shard_size: int):
    print(f"Wake round {round_no}")
    trainer._prepare_a_cache()
    snapshot = trainer.export_frozen_b_snapshot()
    print(f"  snapshot step={snapshot.step} frozen_entries={len(snapshot.keys_sorted)}")
    if len(snapshot.keys_sorted) > 0:
        print("  frozen snapshot entries")
        for idx in range(len(snapshot.keys_sorted)):
            src = int(snapshot.keys_sorted[idx] // trainer.K)
            tgt = int(snapshot.keys_sorted[idx] % trainer.K)
            print(
                f"    {format_slot(src, slot_words_map)} -> {format_slot(tgt, slot_words_map)} "
                f"vec={np.array2string(snapshot.vecs_sorted[idx], precision=4, suppress_small=False)}"
            )

    b_before = capture_b_edges(trainer.model)
    stats_list = []

    for shard_idx, shard in enumerate(dataset.iter_shards(shard_size)):
        print(f"  shard[{shard_idx}] positions=[{shard.position_start}, {shard.position_end})")
        for local_pos in range(shard.n_positions):
            global_pos = int(shard.position_start + local_pos)
            item = positions[global_pos]
            a0 = int(shard.active_offsets[local_pos])
            a1 = int(shard.active_offsets[local_pos + 1])
            t0 = int(shard.target_offsets[local_pos])
            t1 = int(shard.target_offsets[local_pos + 1])
            active_slots = shard.active_slots[a0:a1]
            phi_hat = shard.phi_hat[a0:a1]
            target_slots = shard.target_slots[t0:t1]
            print(
                f"    pos[{global_pos}] ctx='{' '.join(item['context_words'])}' "
                f"src='{item['source_word']}' tgt='{item['target_word']}'"
            )
            for row_idx, slot in enumerate(active_slots.tolist()):
                print(
                    f"      active[{row_idx}] {format_slot(slot, slot_words_map)} "
                    f"q={np.array2string(item['q_vals'][row_idx], precision=4, suppress_small=False)} "
                    f"phi_hat={np.array2string(phi_hat[row_idx], precision=4, suppress_small=False)}"
                )
            print(
                "      target slots:",
                ", ".join(format_slot(slot, slot_words_map) for slot in target_slots.tolist())
            )

            if len(active_slots) > 0 and len(target_slots) > 0:
                all_src = np.repeat(active_slots, len(target_slots))
                all_tgt = np.tile(target_slots, len(active_slots))
                phi_rep = np.repeat(phi_hat, len(target_slots), axis=0)
                b_strength = trainer._compute_b_strength_from_snapshot(all_src, all_tgt, phi_rep, snapshot)
                if trainer.a.has_knowledge:
                    a_strength = trainer.b._compute_a_strength_vectorized(all_src, all_tgt, phi_rep)
                    a_strength = a_strength / float(max(1, trainer.a.d))
                else:
                    a_strength = np.zeros_like(b_strength)
                gaps = 1.0 - (a_strength + b_strength)
                delta = trainer.b._eta * gaps[:, None] * phi_rep
                for pair_idx in range(len(all_src)):
                    print(
                        f"      pair[{pair_idx}] {format_slot(int(all_src[pair_idx]), slot_words_map)} "
                        f"-> {format_slot(int(all_tgt[pair_idx]), slot_words_map)} "
                        f"A={a_strength[pair_idx]:.4f} B={b_strength[pair_idx]:.4f} gap={gaps[pair_idx]:.4f} "
                        f"delta={np.array2string(delta[pair_idx], precision=4, suppress_small=False)}"
                    )

        stats = trainer.propose_shard(shard, snapshot)
        stats_list.append(stats)
        if stats is not None:
            phi_mean = stats.phi_sum / stats.count[:, None]
            gap_mean = stats.gap_sum / stats.count
            delta = trainer.b._eta * gap_mean[:, None] * phi_mean
            print(f"  shard[{shard_idx}] merged")
            for idx in range(len(stats.keys)):
                src = int(stats.keys[idx] // trainer.K)
                tgt = int(stats.keys[idx] % trainer.K)
                print(
                    f"    {format_slot(src, slot_words_map)} -> {format_slot(tgt, slot_words_map)} "
                    f"count={int(stats.count[idx])} "
                    f"phi_sum={np.array2string(stats.phi_sum[idx], precision=4, suppress_small=False)} "
                    f"gap_sum={stats.gap_sum[idx]:.4f} "
                    f"phi_mean={np.array2string(phi_mean[idx], precision=4, suppress_small=False)} "
                    f"gap_mean={gap_mean[idx]:.4f} "
                    f"delta={np.array2string(delta[idx], precision=4, suppress_small=False)}"
                )

    merged = trainer.tree_merge(stats_list)
    if merged is None:
        print("  round merged: <empty>")
        print()
        return

    phi_mean = merged.phi_sum / merged.count[:, None]
    gap_mean = merged.gap_sum / merged.count
    delta = trainer.b._eta * gap_mean[:, None] * phi_mean
    print("  round merged result")
    for idx in range(len(merged.keys)):
        src = int(merged.keys[idx] // trainer.K)
        tgt = int(merged.keys[idx] % trainer.K)
        print(
            f"    {format_slot(src, slot_words_map)} -> {format_slot(tgt, slot_words_map)} "
            f"count={int(merged.count[idx])} "
            f"phi_sum={np.array2string(merged.phi_sum[idx], precision=4, suppress_small=False)} "
            f"gap_sum={merged.gap_sum[idx]:.4f} "
            f"phi_mean={np.array2string(phi_mean[idx], precision=4, suppress_small=False)} "
            f"gap_mean={gap_mean[idx]:.4f} "
            f"delta={np.array2string(delta[idx], precision=4, suppress_small=False)}"
        )

    trainer.apply_merged_stats(merged)
    b_after = capture_b_edges(trainer.model)
    print("  round B before/after")
    for idx in range(len(merged.keys)):
        src = int(merged.keys[idx] // trainer.K)
        tgt = int(merged.keys[idx] % trainer.K)
        key = (src, tgt)
        before = b_before.get(key, np.zeros((trainer.D,), dtype=np.float32))
        after = b_after.get(key, np.zeros((trainer.D,), dtype=np.float32))
        print(
            f"    {format_slot(src, slot_words_map)} -> {format_slot(tgt, slot_words_map)} "
            f"before={np.array2string(before, precision=4, suppress_small=False)} "
            f"after={np.array2string(after, precision=4, suppress_small=False)} "
            f"delta={np.array2string(after - before, precision=4, suppress_small=False)}"
        )
    print()


def print_dream_generation_trace(label: str,
                                 dream_data: dict,
                                 all_slots: np.ndarray,
                                 slot_words_map: dict[int, list[str]],
                                 limit: int):
    print(label)
    debug_trace = dream_data.get("debug_trace")
    if not debug_trace:
        print("  <debug trace unavailable>")
        print()
        return
    print("  injected S")
    for step_idx, slot_counts in enumerate(debug_trace["slot_sequence"]):
        if slot_counts is None:
            print(f"    S[{step_idx}] <reset>")
            continue
        pretty = ", ".join(
            f"{format_slot(int(slot), slot_words_map)}:{float(value):.4f}"
            for slot, value in slot_counts.items()
        )
        print(f"    S[{step_idx}] {pretty}")
    print("  realized sliding-Q steps")
    for row_idx, item in enumerate(debug_trace["realized_steps"][:limit]):
        print(f"    dream_step[{row_idx}] source_step={item['step_idx']}")
        injected = ", ".join(
            f"{format_slot(int(slot), slot_words_map)}:{float(value):.4f}"
            for slot, value in item["slot_counts"].items()
        )
        print(f"      injected={injected}")
        for active_idx, slot in enumerate(item["active_slots_step"].tolist()):
            print(
                f"      active[{active_idx}] {format_slot(slot, slot_words_map)} "
                f"q={np.array2string(item['q_vals_step'][active_idx], precision=4, suppress_small=False)} "
                f"phi={np.array2string(item['phi_step'][active_idx], precision=4, suppress_small=False)}"
            )
        print("      teacher full local")
        for local_idx, slot in enumerate(all_slots.tolist()):
            print(f"        {format_slot(slot, slot_words_map)} -> {item['teacher_local'][local_idx]:.4f}")
        print("      candidate teacher logits")
        for local_idx, logit in zip(item["candidate_local"].tolist(), item["teacher_logits"].tolist()):
            print(f"        {format_slot(int(all_slots[int(local_idx)]), slot_words_map)} -> {float(logit):.4f}")
    print()


def print_dream_contexts(label: str,
                         model: MathBrain,
                         dream_data: dict,
                         all_slots: np.ndarray,
                         slot_words_map: dict[int, list[str]],
                         limit: int):
    print(label)
    n_ctx = int(min(limit, dream_data["active_local"].shape[0]))
    for idx in range(n_ctx):
        count = int(np.sum(dream_data["active_mask"][idx]))
        active_local = dream_data["active_local"][idx, :count].astype(np.int32, copy=False)
        phi_rows = dream_data["phi"][idx, :count].astype(np.float32, copy=False)
        candidate_local = dream_data["candidate_local"][idx].astype(np.int32, copy=False)
        teacher_logits = dream_data["teacher_logits"][idx].astype(np.float32, copy=False)
        student_logits = candidate_logits_from_a(model, all_slots, active_local, phi_rows, candidate_local)
        candidate_slots = all_slots[candidate_local]
        teacher_scores = np.zeros((model.config.K,), dtype=np.float32)
        teacher_scores[candidate_slots] = teacher_logits
        student_scores = np.zeros((model.config.K,), dtype=np.float32)
        student_scores[candidate_slots] = student_logits
        print(f"  dream[{idx}]")
        print("    active:", ", ".join(format_slot(slot, slot_words_map) for slot in all_slots[active_local].tolist()))
        print("    teacher logits:")
        for j in range(len(candidate_slots)):
            print(f"      {format_slot(candidate_slots[j], slot_words_map)} -> {teacher_logits[j]:.4f}")
        print("    student A logits:")
        for j in range(len(candidate_slots)):
            print(f"      {format_slot(candidate_slots[j], slot_words_map)} -> {student_logits[j]:.4f}")
        print("    teacher words:", slot_scores_to_word_scores(model, teacher_scores))
        print("    student words:", slot_scores_to_word_scores(model, student_scores))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cpu", "mlx"], default="cpu")
    parser.add_argument("--sleep-backend", choices=["mlx", "numpy"], default="mlx")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--d-phi", type=int, default=8)
    parser.add_argument("--cp-rank", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--lambda-wd", type=float, default=1e-4)
    parser.add_argument("--chaos-folds", type=int, default=3)
    parser.add_argument("--chaos-alpha", type=float, default=2.0)
    parser.add_argument("--sleep-global-decay", type=float, default=1.0)
    parser.add_argument("--gamma-decay", type=float, default=0.9)
    parser.add_argument("--sleep-lr", type=float, default=0.01)
    parser.add_argument("--sleep-epochs", type=int, default=40)
    parser.add_argument("--dream-samples", type=int, default=8)
    parser.add_argument("--dream-group", type=int, default=4)
    parser.add_argument("--dream-topk", type=int, default=64)
    parser.add_argument("--dream-waves", type=int, default=6)
    parser.add_argument("--dream-batch", type=int, default=8)
    parser.add_argument("--dream-temp", type=float, default=1.0)
    parser.add_argument("--dream-prox", type=float, default=1e-3)
    parser.add_argument("--dream-episode", type=int, default=12)
    parser.add_argument("--dream-seq-multiplier", type=float, default=1.0)
    parser.add_argument("--dream-uniform-mix", type=float, default=0.25)
    parser.add_argument("--dream-sequence-source", choices=["train", "synthetic"], default="train")
    parser.add_argument("--dream-print", type=int, default=4)
    parser.add_argument("--full-io", action="store_true")
    parser.add_argument("--trace-max-contexts", type=int, default=8)
    parser.add_argument("--corpus", nargs="*", default=DEFAULT_CORPUS)
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=False, linewidth=240)

    model = build_model(args)
    trainer = build_trainer(model, args.backend)
    corpus = list(args.corpus)
    ensure_vocab(model, corpus)
    dataset = trainer.build_dataset(corpus, verbose=False)
    slot_words_map = slot_to_words(model)
    positions = tokenize_positions(model, corpus)
    tracked_slots = np.asarray(sorted(slot_words_map.keys()), dtype=np.int32)

    print("Tiny corpus")
    for line in corpus:
        print(f"  {line}")
    print()
    print_word_slots(model, slot_words_map)
    print_position_inputs("Wake position inputs", positions, slot_words_map)

    for cycle_idx in range(args.cycles):
        cycle_no = cycle_idx + 1
        print("=" * 80)
        print(f"Cycle {cycle_no}")
        print("=" * 80)

        wake_before_eval = evaluate_parallel(model, dataset, trainer)
        print(f"Pre-wake accuracy: {wake_before_eval['correct']}/{wake_before_eval['total']} = {wake_before_eval['accuracy']:.2f}%")
        positive_rows = compute_positive_edge_table(model, positions)
        print_positive_edge_summary("Pre-wake positive edge gap summary", positive_rows, slot_words_map)
        print_positive_edge_table("Pre-wake positive edge gaps", positive_rows, slot_words_map)

        for round_no in range(1, args.rounds + 1):
            if args.full_io:
                print_parallel_wake_round(round_no, trainer, dataset, positions, slot_words_map, shard_size=4096)
            else:
                trainer.run_round(dataset, shard_size=4096, verbose=False)
        trainer.sync_to_hash()

        wake_after_eval = evaluate_parallel(model, dataset, trainer)
        print(f"Post-wake accuracy: {wake_after_eval['correct']}/{wake_after_eval['total']} = {wake_after_eval['accuracy']:.2f}%")
        print_wake_positions("Wake positions after wake", model, positions, slot_words_map, tracked_slots=tracked_slots)
        print_b_entries(model, slot_words_map)

        dream_data, dream_info = build_dream_bundle(
            model,
            corpus,
            n_dreams=args.dream_samples,
            topk=args.dream_topk,
            debug_trace=args.full_io,
        )
        if not dream_data:
            print(f"Dream unavailable: {dream_info}")
            continue
        all_slots = np.asarray(sorted({int(src) for src, _, _ in model.b.get_entries()} | {int(tgt) for _, tgt, _ in model.b.get_entries()} | {int(slot) for slots in model.word_to_slots.values() for slot in slots.tolist()}), dtype=np.int32)
        involved_slots = sorted({int(slot) for slot in all_slots.tolist()})
        a_before = capture_a_edges(model, involved_slots)
        print_a_factors("A factors before sleep", model, involved_slots, slot_words_map)
        print(f"Dream info: {dream_info}")
        print_dream_generation_trace("Dream generation trace", dream_data, all_slots, slot_words_map, limit=args.dream_print)
        print_dream_contexts("Dream contexts before sleep", model, dream_data, all_slots, slot_words_map, limit=args.dream_print)

        sleep_result = model.sleep(b_retention=0.0, sleep_backend=args.sleep_backend, dream_corpus=corpus)
        trainer.invalidate_dense_state()
        print(f"Sleep result: {sleep_result}")

        a_after = capture_a_edges(model, involved_slots)
        print_a_factors("A factors after sleep", model, involved_slots, slot_words_map)
        print_dream_contexts("Dream contexts after sleep", model, dream_data, all_slots, slot_words_map, limit=args.dream_print)
        print_a_edge_delta(a_before, a_after, slot_words_map)

        sleep_eval = evaluate_parallel(model, dataset, trainer)
        print(f"Post-sleep accuracy: {sleep_eval['correct']}/{sleep_eval['total']} = {sleep_eval['accuracy']:.2f}%")
        print_wake_positions("Wake positions after sleep", model, positions, slot_words_map, tracked_slots=tracked_slots)
        print_b_entries(model, slot_words_map, label="B entries after sleep")
        print(f"A stats: {model.a.stats()}")
        print(f"B stats: {model.b.stats()}")
        print()


if __name__ == "__main__":
    main()
