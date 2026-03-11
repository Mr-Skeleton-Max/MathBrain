#!/usr/bin/env python3
"""Inspect actual sleep learning targets from a saved snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.sleep_dream import prepare_dream_distillation_data
from mathbrain.wake_dataset import WakeDataset
from experiments.run_parallel_wake import load_corpus


def _infer_config(snapshot: np.lib.npyio.NpzFile) -> MathBrainConfig:
    cfg = MathBrainConfig()
    cfg.PHI_MODE = 'chaos'
    b_vec = snapshot['b_vec']
    if b_vec.ndim == 2 and b_vec.shape[1] > 0:
        cfg.D_PHI = int(b_vec.shape[1])
    a_e_src = snapshot['a_E_src']
    if a_e_src.ndim == 3 and a_e_src.shape[1] > 0:
        cfg.CP_RANK = int(a_e_src.shape[1])
    cfg._build_derived()
    return cfg


def _restore_model(snapshot_path: Path):
    snap = np.load(snapshot_path)
    cfg = _infer_config(snap)
    model = MathBrain(cfg)

    a_slots = snap['a_slots'].astype(np.int32, copy=False)
    a_e_src = snap['a_E_src'].astype(np.float32, copy=False)
    a_e_tgt = snap['a_E_tgt'].astype(np.float32, copy=False)
    if len(a_slots) > 0:
        model.a.E_src = {int(slot): a_e_src[i].copy() for i, slot in enumerate(a_slots.tolist())}
        model.a.E_tgt = {int(slot): a_e_tgt[i].copy() for i, slot in enumerate(a_slots.tolist())}
        model.a.has_knowledge = True
        model.a._build_cache()

    b_src = snap['b_src'].astype(np.int32, copy=False)
    b_tgt = snap['b_tgt'].astype(np.int32, copy=False)
    b_vec = snap['b_vec'].astype(np.float32, copy=False)
    b_time = snap['b_time'].astype(np.int32, copy=False)
    b = model.b
    b._capacity = max(1, len(b_src))
    b._srcs = b_src.copy()
    b._tgts = b_tgt.copy()
    b._vecs = b_vec.copy()
    b._times = b_time.copy()
    b._n_entries = len(b_src)
    b.n_b = len(b_src)
    b._hash_table = {
        (int(src), int(tgt)): idx
        for idx, (src, tgt) in enumerate(zip(b_src.tolist(), b_tgt.tolist()))
    }
    return model, snap


def _build_edge_dict(snap: np.lib.npyio.NpzFile) -> dict[tuple[int, int], np.ndarray]:
    edge_dict: dict[tuple[int, int], np.ndarray] = {}
    for src, tgt, vec in zip(snap['b_src'].tolist(), snap['b_tgt'].tolist(), snap['b_vec']):
        edge_dict[(int(src), int(tgt))] = np.asarray(vec, dtype=np.float32)
    return edge_dict


def _select_slots(snap: np.lib.npyio.NpzFile, top_slots: int) -> list[int]:
    score: dict[int, float] = {}
    for src, tgt, vec in zip(snap['b_src'].tolist(), snap['b_tgt'].tolist(), snap['b_vec']):
        norm = float(np.linalg.norm(vec))
        score[int(src)] = score.get(int(src), 0.0) + norm
        score[int(tgt)] = score.get(int(tgt), 0.0) + norm
    ordered = sorted(score.items(), key=lambda item: item[1], reverse=True)
    return [slot for slot, _ in ordered[:max(1, int(top_slots))]]


def _format_float(x: float, width: int = 8) -> str:
    return f"{x:{width}.3f}"


def _print_matrix(title: str, rows: list[int], cols: list[int], values: np.ndarray):
    print(title)
    header = "src\\tgt".ljust(12) + "".join(f"{col:>10d}" for col in cols)
    print(header)
    for row_idx, src in enumerate(rows):
        line = f"{src:<12d}"
        for col_idx, _ in enumerate(cols):
            line += _format_float(float(values[row_idx, col_idx]), width=10)
        print(line)
    print()


def _ensure_corpus_vocab(model: MathBrain, corpus: list[str]):
    seen: set[str] = set()
    for sentence in corpus:
        for word in WakeDataset._tokenize(sentence):
            if word not in seen:
                model._ensure_word(word)
                seen.add(word)


def _encode_phi_normalized(model: MathBrain, q_vals: np.ndarray) -> np.ndarray:
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


def _build_wake_contexts(model: MathBrain, corpus: list[str]) -> list[dict]:
    retina = model.retina
    rho = model.config.rho.astype(np.float32, copy=False)
    theta_q = float(model.config.THETA_Q)
    eps_q = float(model.config.EPS_Q)

    contexts: list[dict] = []
    for sent_idx, sentence in enumerate(corpus):
        words = WakeDataset._tokenize(sentence)
        if len(words) < 2:
            continue

        for word in words:
            model._ensure_word(word)

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

            target_slots = np.fromiter(encoded[pos + 1].keys(), dtype=np.int32)
            if len(target_slots) == 0:
                continue

            phi_hat = _encode_phi_normalized(model, q_by_pos[pos])
            contexts.append({
                'sent_idx': int(sent_idx),
                'pos_idx': int(pos),
                'context_words': words[:pos + 1],
                'target_word': words[pos + 1],
                'active_slots': active_slots.astype(np.int32, copy=True),
                'phi': phi_hat.astype(np.float32, copy=True),
                'gold_slots': target_slots.astype(np.int32, copy=True),
            })
    return contexts


def _sample_context_indices(n_total: int, n_contexts: int, offset: int = 0) -> np.ndarray:
    if n_total <= 0:
        return np.zeros((0,), dtype=np.int32)
    start = min(max(0, int(offset)), max(0, n_total - 1))
    remain = n_total - start
    count = min(max(1, int(n_contexts)), remain)
    if count >= remain:
        return np.arange(start, n_total, dtype=np.int32)
    idx = np.linspace(start, n_total - 1, num=count, dtype=np.int64)
    return np.unique(idx.astype(np.int32, copy=False))


def _print_edge_vectors(model: MathBrain,
                        edge_dict: dict[tuple[int, int], np.ndarray],
                        slots: list[int],
                        max_edges: int):
    print("Edge vectors (current sleep target semantics)")
    printed = 0
    for src in slots:
        for tgt in slots:
            b_vec = edge_dict.get((src, tgt))
            if b_vec is None:
                continue
            if model.a.has_knowledge:
                a_vec = model.a.compute_entry(src, tgt).astype(np.float32, copy=False)
                if model.config.GAMMA_ADAPTIVE:
                    gamma_src = model.a.get_adaptive_gamma(src, model.config.GAMMA_INITIAL, model.config.GAMMA_MIN, model.config.GAMMA_DECAY_RATE)
                    gamma_tgt = model.a.get_adaptive_gamma(tgt, model.config.GAMMA_INITIAL, model.config.GAMMA_MIN, model.config.GAMMA_DECAY_RATE)
                    gamma = max(gamma_src, gamma_tgt)
                else:
                    gamma = float(model.config.GAMMA_DECAY)
            else:
                a_vec = np.zeros_like(b_vec)
                gamma = float(model.config.GAMMA_DECAY)
            target_vec = (gamma * a_vec) + b_vec
            print(f"  {src}->{tgt}")
            print(f"    B      = {np.array2string(b_vec, precision=4, suppress_small=False)}")
            print(f"    A_old  = {np.array2string(a_vec, precision=4, suppress_small=False)}")
            print(f"    gamma  = {gamma:.4f}")
            print(f"    target = {np.array2string(target_vec, precision=4, suppress_small=False)}")
            printed += 1
            if printed >= max_edges:
                print()
                return
    print()


def _print_edge_target_summary(model: MathBrain,
                               snap: np.lib.npyio.NpzFile,
                               slots: list[int],
                               max_edges: int):
    edge_dict = _build_edge_dict(snap)
    n = len(slots)
    b_norm = np.zeros((n, n), dtype=np.float32)
    a_norm = np.zeros((n, n), dtype=np.float32)
    target_norm = np.zeros((n, n), dtype=np.float32)

    for i, src in enumerate(slots):
        for j, tgt in enumerate(slots):
            b_vec = edge_dict.get((src, tgt))
            if b_vec is None:
                continue
            b_norm[i, j] = float(np.linalg.norm(b_vec))
            if model.a.has_knowledge:
                a_vec = model.a.compute_entry(src, tgt).astype(np.float32, copy=False)
                a_norm[i, j] = float(np.linalg.norm(a_vec))
                if model.config.GAMMA_ADAPTIVE:
                    gamma_src = model.a.get_adaptive_gamma(src, model.config.GAMMA_INITIAL, model.config.GAMMA_MIN, model.config.GAMMA_DECAY_RATE)
                    gamma_tgt = model.a.get_adaptive_gamma(tgt, model.config.GAMMA_INITIAL, model.config.GAMMA_MIN, model.config.GAMMA_DECAY_RATE)
                    gamma = max(gamma_src, gamma_tgt)
                else:
                    gamma = float(model.config.GAMMA_DECAY)
            else:
                a_vec = np.zeros_like(b_vec)
                gamma = float(model.config.GAMMA_DECAY)
            target_norm[i, j] = float(np.linalg.norm((gamma * a_vec) + b_vec))

    _print_matrix("B edge norm matrix", slots, slots, b_norm)
    _print_matrix("A_old edge norm matrix", slots, slots, a_norm)
    _print_matrix("Current sleep target norm matrix", slots, slots, target_norm)
    _print_edge_vectors(model, edge_dict, slots, max_edges=max_edges)


def _global_slots_from_local(all_slots: np.ndarray, local_ids: np.ndarray) -> list[int]:
    return [int(all_slots[int(local_id)]) for local_id in local_ids.tolist()]


def _print_dream_targets(model: MathBrain,
                         snap: np.lib.npyio.NpzFile,
                         n_contexts: int,
                         dream_topk: int,
                         student_model: MathBrain | None = None,
                         dream_corpus: list[str] | None = None):
    entries = [
        (int(src), int(tgt), np.asarray(vec, dtype=np.float32))
        for src, tgt, vec in zip(snap['b_src'].tolist(), snap['b_tgt'].tolist(), snap['b_vec'])
    ]
    all_slots = sorted(set(int(src) for src in snap['b_src'].tolist()) | set(int(tgt) for tgt in snap['b_tgt'].tolist()))

    dream_data, info = prepare_dream_distillation_data(
        b_memory=model.b,
        entries=entries,
        all_slots=all_slots,
        config=model.config,
        a_knowledge=model.a,
        global_decay=1.0,
        retina=model.retina,
        vocab_words=[],
        n_dreams=max(1, int(n_contexts)),
        dream_size=int(getattr(model.config, 'SLEEP_DREAM_ACTIVE', 8)),
        candidate_topk=max(16, int(dream_topk)),
        n_waves=int(getattr(model.config, 'SLEEP_DREAM_PHASES', 8)),
        episode_len=int(getattr(model.config, 'SLEEP_DREAM_EPISODE_LEN', 12)),
        probe_count=int(getattr(model.config, 'SLEEP_DREAM_PROBE_COUNT', 3)),
        uniform_mix=float(getattr(model.config, 'SLEEP_DREAM_UNIFORM_MIX', 0.25)),
        rng_seed=0,
        dream_corpus=dream_corpus,
        verbose=False,
    )

    if not dream_data:
        print(f"Dream data unavailable: {info}")
        return

    print("Dream generator info")
    for key in (
        'n_dreams', 'sequence_len', 'ema_window', 'context_width',
        'raw_active_mean', 'raw_active_max', 'teacher_peak_mean',
        'group_size', 'n_waves'
    ):
        if key in info:
            print(f"  {key}: {info[key]}")
    print()

    all_slots_arr = np.asarray(all_slots, dtype=np.int32)
    contexts = min(int(n_contexts), int(dream_data['active_local'].shape[0]))
    union_targets: list[int] = []
    union_seen: set[int] = set()

    print("Dream contexts")
    for idx in range(contexts):
        count = int(np.sum(dream_data['active_mask'][idx]))
        active_local = dream_data['active_local'][idx, :count].astype(np.int32, copy=False)
        active_global = _global_slots_from_local(all_slots_arr, active_local)
        candidate_local = dream_data['candidate_local'][idx].astype(np.int32, copy=False)
        teacher_logits = dream_data['teacher_logits'][idx].astype(np.float32, copy=False)
        phi_rows = dream_data['phi'][idx, :count].astype(np.float32, copy=False)
        top_idx = np.argsort(teacher_logits)[::-1][:min(8, len(teacher_logits))]
        top_targets = [int(all_slots_arr[int(candidate_local[top])]) for top in top_idx.tolist()]
        top_logits = [float(teacher_logits[top]) for top in top_idx.tolist()]
        print(f"  ctx[{idx}] active_slots({len(active_global)}): {active_global}")
        print(f"    top_targets: {list(zip(top_targets, [round(v, 4) for v in top_logits]))}")
        if student_model is not None:
            student_scores = student_model.a.predict(np.asarray(active_global, dtype=np.int32), phi_rows)
            student_logits = student_scores[np.asarray(top_targets, dtype=np.int32)]
            print(f"    student_on_teacher_targets: {list(zip(top_targets, [round(float(v), 4) for v in student_logits.tolist()]))}")
            print(f"    teacher_logits_full: {np.array2string(teacher_logits, precision=4, suppress_small=False)}")
            candidate_slots = np.asarray([int(all_slots_arr[int(local)]) for local in candidate_local.tolist()], dtype=np.int32)
            student_same_candidates = student_scores[candidate_slots]
            print(f"    student_logits_full: {np.array2string(student_same_candidates, precision=4, suppress_small=False)}")
        for slot in top_targets[:min(8, len(top_targets))]:
            if slot not in union_seen:
                union_seen.add(slot)
                union_targets.append(slot)
    print()

    union_targets = union_targets[:min(12, len(union_targets))]
    if not union_targets:
        return

    teacher_matrix = np.zeros((contexts, len(union_targets)), dtype=np.float32)
    for idx in range(contexts):
        candidate_local = dream_data['candidate_local'][idx].astype(np.int32, copy=False)
        teacher_logits = dream_data['teacher_logits'][idx].astype(np.float32, copy=False)
        global_to_logit = {
            int(all_slots_arr[int(local)]): float(logit)
            for local, logit in zip(candidate_local.tolist(), teacher_logits.tolist())
        }
        for col, slot in enumerate(union_targets):
            teacher_matrix[idx, col] = float(global_to_logit.get(int(slot), 0.0))

    _print_matrix("Dream teacher logit matrix (rows=context, cols=target slot)", list(range(contexts)), union_targets, teacher_matrix)
    if student_model is not None:
        student_matrix = np.zeros((contexts, len(union_targets)), dtype=np.float32)
        diff_matrix = np.zeros((contexts, len(union_targets)), dtype=np.float32)
        for idx in range(contexts):
            count = int(np.sum(dream_data['active_mask'][idx]))
            active_local = dream_data['active_local'][idx, :count].astype(np.int32, copy=False)
            active_global = np.asarray(_global_slots_from_local(all_slots_arr, active_local), dtype=np.int32)
            phi_rows = dream_data['phi'][idx, :count].astype(np.float32, copy=False)
            student_scores = student_model.a.predict(active_global, phi_rows)
            for col, slot in enumerate(union_targets):
                student_matrix[idx, col] = float(student_scores[int(slot)])
                diff_matrix[idx, col] = float(student_matrix[idx, col] - teacher_matrix[idx, col])
        _print_matrix("Dream student logit matrix (rows=context, cols=target slot)", list(range(contexts)), union_targets, student_matrix)
        _print_matrix("Dream student-target diff matrix (rows=context, cols=target slot)", list(range(contexts)), union_targets, diff_matrix)


def _print_wake_targets(model: MathBrain,
                        corpus: list[str],
                        n_contexts: int,
                        wake_topk: int,
                        student_model: MathBrain | None = None,
                        wake_offset: int = 0):
    if not corpus:
        print("Wake data unavailable: empty corpus")
        return

    _ensure_corpus_vocab(model, corpus)
    if student_model is not None:
        _ensure_corpus_vocab(student_model, corpus)

    wake_contexts = _build_wake_contexts(model, corpus)
    if not wake_contexts:
        print("Wake data unavailable: no valid wake positions")
        return

    sample_idx = _sample_context_indices(len(wake_contexts), n_contexts, offset=wake_offset)
    sampled = [wake_contexts[int(idx)] for idx in sample_idx.tolist()]
    union_targets: list[int] = []
    union_seen: set[int] = set()

    print("Wake contexts")
    for local_idx, ctx in enumerate(sampled):
        active_slots = ctx['active_slots'].astype(np.int32, copy=False)
        phi = ctx['phi'].astype(np.float32, copy=False)
        gold_slots = ctx['gold_slots'].astype(np.int32, copy=False)

        teacher_b = model.b.predict(active_slots, phi)
        teacher_a = model.a.predict(active_slots, phi)
        teacher_total = teacher_a + teacher_b
        teacher_top = np.argsort(teacher_total)[::-1][:max(1, int(wake_topk))]

        gold_set = set(int(slot) for slot in gold_slots.tolist())
        print(
            f"  ctx[{local_idx}] wake_pos={int(sample_idx[local_idx])} "
            f"sent={ctx['sent_idx']} pos={ctx['pos_idx']} "
            f"context={' '.join(ctx['context_words'])} -> {ctx['target_word']}"
        )
        print(f"    active_slots({len(active_slots)}): {active_slots.tolist()}")
        print(f"    gold_slots: {gold_slots.tolist()}")

        teacher_rows = []
        for slot in teacher_top.tolist():
            teacher_rows.append(
                (
                    int(slot),
                    round(float(teacher_total[int(slot)]), 4),
                    round(float(teacher_a[int(slot)]), 4),
                    round(float(teacher_b[int(slot)]), 4),
                    int(slot) in gold_set,
                )
            )
        print(f"    teacher_top_targets(total,A,B,is_gold): {teacher_rows}")

        if student_model is not None:
            student_scores = student_model.a.predict(active_slots, phi)
            student_rows = [
                (
                    int(slot),
                    round(float(student_scores[int(slot)]), 4),
                    int(slot) in gold_set,
                )
                for slot in teacher_top.tolist()
            ]
            student_top = np.argsort(student_scores)[::-1][:max(1, int(wake_topk))]
            print(f"    student_on_teacher_targets(A_only,is_gold): {student_rows}")
            print(
                "    student_top_targets(A_only,is_gold): "
                f"{[(int(slot), round(float(student_scores[int(slot)]), 4), int(slot) in gold_set) for slot in student_top.tolist()]}"
            )
            print(
                f"    teacher_gold_logits(total): "
                f"{[(int(slot), round(float(teacher_total[int(slot)]), 4)) for slot in gold_slots.tolist()]}"
            )
            print(
                f"    student_gold_logits(A_only): "
                f"{[(int(slot), round(float(student_scores[int(slot)]), 4)) for slot in gold_slots.tolist()]}"
            )

        for slot in gold_slots.tolist():
            slot_i = int(slot)
            if slot_i not in union_seen:
                union_seen.add(slot_i)
                union_targets.append(slot_i)
        for slot in teacher_top[:min(8, len(teacher_top))].tolist():
            slot_i = int(slot)
            if slot_i not in union_seen:
                union_seen.add(slot_i)
                union_targets.append(slot_i)
    print()

    union_targets = union_targets[:min(16, len(union_targets))]
    if not union_targets:
        return

    rows = list(range(len(sampled)))
    teacher_total_matrix = np.zeros((len(sampled), len(union_targets)), dtype=np.float32)
    teacher_a_matrix = np.zeros_like(teacher_total_matrix)
    teacher_b_matrix = np.zeros_like(teacher_total_matrix)
    student_matrix = np.zeros_like(teacher_total_matrix) if student_model is not None else None
    diff_matrix = np.zeros_like(teacher_total_matrix) if student_model is not None else None

    for row_idx, ctx in enumerate(sampled):
        active_slots = ctx['active_slots'].astype(np.int32, copy=False)
        phi = ctx['phi'].astype(np.float32, copy=False)
        teacher_b = model.b.predict(active_slots, phi)
        teacher_a = model.a.predict(active_slots, phi)
        teacher_total = teacher_a + teacher_b
        if student_model is not None:
            student_scores = student_model.a.predict(active_slots, phi)
        for col_idx, slot in enumerate(union_targets):
            teacher_total_matrix[row_idx, col_idx] = float(teacher_total[int(slot)])
            teacher_a_matrix[row_idx, col_idx] = float(teacher_a[int(slot)])
            teacher_b_matrix[row_idx, col_idx] = float(teacher_b[int(slot)])
            if student_model is not None and student_matrix is not None and diff_matrix is not None:
                student_matrix[row_idx, col_idx] = float(student_scores[int(slot)])
                diff_matrix[row_idx, col_idx] = float(student_scores[int(slot)] - teacher_total[int(slot)])

    _print_matrix("Wake teacher total logit matrix (rows=context, cols=target slot)", rows, union_targets, teacher_total_matrix)
    _print_matrix("Wake teacher A logit matrix (rows=context, cols=target slot)", rows, union_targets, teacher_a_matrix)
    _print_matrix("Wake teacher B logit matrix (rows=context, cols=target slot)", rows, union_targets, teacher_b_matrix)
    if student_model is not None and student_matrix is not None and diff_matrix is not None:
        _print_matrix("Wake student A logit matrix (rows=context, cols=target slot)", rows, union_targets, student_matrix)
        _print_matrix("Wake student-teacher diff matrix (rows=context, cols=target slot)", rows, union_targets, diff_matrix)


def _build_tiny_model_from_slots(model: MathBrain,
                                 snap: np.lib.npyio.NpzFile,
                                 slots: list[int]) -> tuple[MathBrain, list[tuple[int, int, np.ndarray]], list[int]]:
    slot_set = set(int(slot) for slot in slots)
    tiny = MathBrain(model.config)

    if model.a.has_knowledge:
        tiny_src = {}
        tiny_tgt = {}
        for slot in slots:
            if int(slot) in model.a.E_src:
                tiny_src[int(slot)] = model.a.E_src[int(slot)].copy()
                tiny_tgt[int(slot)] = model.a.E_tgt[int(slot)].copy()
        if tiny_src:
            tiny.a.E_src = tiny_src
            tiny.a.E_tgt = tiny_tgt
            tiny.a.has_knowledge = True
            tiny.a._build_cache()

    entries: list[tuple[int, int, np.ndarray]] = []
    for src, tgt, vec in zip(snap['b_src'].tolist(), snap['b_tgt'].tolist(), snap['b_vec']):
        src_i = int(src)
        tgt_i = int(tgt)
        if src_i in slot_set and tgt_i in slot_set:
            entries.append((src_i, tgt_i, np.asarray(vec, dtype=np.float32).copy()))

    b = tiny.b
    b._capacity = max(1, len(entries))
    b._srcs = np.empty((b._capacity,), dtype=np.int32)
    b._tgts = np.empty((b._capacity,), dtype=np.int32)
    b._vecs = np.empty((b._capacity, b.D), dtype=np.float32)
    b._times = np.zeros((b._capacity,), dtype=np.int32)
    b._hash_table = {}
    for idx, (src, tgt, vec) in enumerate(entries):
        b._srcs[idx] = int(src)
        b._tgts[idx] = int(tgt)
        b._vecs[idx] = np.asarray(vec, dtype=np.float32)
        b._times[idx] = 0
        b._hash_table[(int(src), int(tgt))] = idx
    b._n_entries = len(entries)
    b.n_b = len(entries)
    return tiny, entries, [int(slot) for slot in slots]


def _print_linear_design_matrix(slots: list[int],
                                active_local: np.ndarray,
                                active_mask: np.ndarray,
                                phi: np.ndarray,
                                teacher_full: np.ndarray,
                                d_rank: int):
    slot_to_idx = {int(slot): idx for idx, slot in enumerate(slots)}
    n_ctx = int(active_local.shape[0])
    n_slots = len(slots)
    d_phi = int(phi.shape[2])
    n_rows = n_ctx * n_slots
    n_cols = n_slots * n_slots * d_phi
    X = np.zeros((n_rows, n_cols), dtype=np.float32)

    def edge_col(src_idx: int, tgt_idx: int, dim_idx: int) -> int:
        return ((src_idx * n_slots + tgt_idx) * d_phi) + dim_idx

    for ctx_idx in range(n_ctx):
        count = int(np.sum(active_mask[ctx_idx]))
        denom = float(max(1, count))
        active_ids = active_local[ctx_idx, :count].astype(np.int32, copy=False)
        phi_rows = phi[ctx_idx, :count].astype(np.float32, copy=False)
        for local_pos, local_id in enumerate(active_ids.tolist()):
            src_slot = int(slots[int(local_id)])
            src_idx = slot_to_idx[src_slot]
            for tgt_idx in range(n_slots):
                row = ctx_idx * n_slots + tgt_idx
                base = edge_col(src_idx, tgt_idx, 0)
                X[row, base:base + d_phi] += phi_rows[local_pos] / denom

    rank = int(np.linalg.matrix_rank(X))
    singular = np.linalg.svd(X, compute_uv=False)
    top_singular = singular[:min(8, len(singular))]
    nonzero = singular[singular > 1e-8]
    print("Tiny direct linear system")
    print(f"  design shape: {X.shape}, rank={rank}, null_dim={X.shape[1] - rank}")
    print(f"  singular values: {np.array2string(top_singular, precision=4, suppress_small=False)}")
    if len(nonzero) > 0:
        print(f"  nonzero sigma min/max: {float(np.min(nonzero)):.6f} / {float(np.max(nonzero)):.6f}")
    print()

    if X.shape[0] <= 24 and X.shape[1] <= 96:
        col_labels = []
        for src in slots:
            for tgt in slots:
                for dim_idx in range(d_phi):
                    col_labels.append(f"{src}->{tgt}[{dim_idx}]")
        print("Tiny design matrix X")
        header = "row".ljust(10) + "".join(f"{label:>14s}" for label in col_labels)
        print(header)
        for row_idx in range(X.shape[0]):
            line = f"{row_idx:<10d}" + "".join(f"{float(val):14.4f}" for val in X[row_idx])
            print(line)
        print()

    _print_matrix(
        "Tiny teacher matrix (rows=context, cols=target slot)",
        list(range(n_ctx)),
        slots,
        teacher_full,
    )


def _print_tiny_dream_targets(model: MathBrain,
                              snap: np.lib.npyio.NpzFile,
                              slots: list[int],
                              n_contexts: int):
    tiny_model, entries, tiny_slots = _build_tiny_model_from_slots(model, snap, slots)
    if not entries:
        print("Tiny-slot dream data unavailable: no intra-slot B edges")
        return

    dream_data, info = prepare_dream_distillation_data(
        b_memory=tiny_model.b,
        entries=entries,
        all_slots=tiny_slots,
        config=tiny_model.config,
        a_knowledge=tiny_model.a,
        global_decay=1.0,
        retina=tiny_model.retina,
        vocab_words=[],
        n_dreams=max(1, int(n_contexts)),
        dream_size=min(max(1, len(tiny_slots)), int(getattr(tiny_model.config, 'SLEEP_DREAM_ACTIVE', 8))),
        candidate_topk=len(tiny_slots),
        n_waves=int(getattr(tiny_model.config, 'SLEEP_DREAM_PHASES', 8)),
        episode_len=int(getattr(tiny_model.config, 'SLEEP_DREAM_EPISODE_LEN', 12)),
        probe_count=int(getattr(tiny_model.config, 'SLEEP_DREAM_PROBE_COUNT', 3)),
        uniform_mix=float(getattr(tiny_model.config, 'SLEEP_DREAM_UNIFORM_MIX', 0.25)),
        rng_seed=0,
        verbose=False,
    )
    if not dream_data:
        print(f"Tiny-slot dream data unavailable: {info}")
        return

    print("Tiny-slot dream info")
    for key in ('n_dreams', 'sequence_len', 'ema_window', 'context_width', 'teacher_peak_mean'):
        if key in info:
            print(f"  {key}: {info[key]}")
    print()

    contexts = min(int(n_contexts), int(dream_data['active_local'].shape[0]))
    teacher_full = np.zeros((contexts, len(tiny_slots)), dtype=np.float32)
    print("Tiny-slot dream contexts")
    for idx in range(contexts):
        count = int(np.sum(dream_data['active_mask'][idx]))
        active_local = dream_data['active_local'][idx, :count].astype(np.int32, copy=False)
        active_slots = [int(tiny_slots[int(local_id)]) for local_id in active_local.tolist()]
        candidate_local = dream_data['candidate_local'][idx].astype(np.int32, copy=False)
        teacher_logits = dream_data['teacher_logits'][idx].astype(np.float32, copy=False)
        print(f"  ctx[{idx}] active_slots: {active_slots}")
        for local_id, logit in zip(candidate_local.tolist(), teacher_logits.tolist()):
            teacher_full[idx, int(local_id)] = float(logit)
        print(f"    logits: {np.array2string(teacher_full[idx], precision=4, suppress_small=False)}")
    print()

    _print_linear_design_matrix(
        slots=tiny_slots,
        active_local=dream_data['active_local'][:contexts],
        active_mask=dream_data['active_mask'][:contexts],
        phi=dream_data['phi'][:contexts],
        teacher_full=teacher_full,
        d_rank=int(getattr(tiny_model.a, 'd', 1)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, type=str)
    parser.add_argument('--student-snapshot', type=str, default='')
    parser.add_argument('--dream-corpus', type=str, default='')
    parser.add_argument('--wake-corpus', type=str, default='')
    parser.add_argument('--top-slots', type=int, default=6)
    parser.add_argument('--edge-vectors', type=int, default=12)
    parser.add_argument('--dream-contexts', type=int, default=6)
    parser.add_argument('--dream-topk', type=int, default=64)
    parser.add_argument('--wake-contexts', type=int, default=6)
    parser.add_argument('--wake-topk', type=int, default=16)
    parser.add_argument('--wake-offset', type=int, default=0)
    parser.add_argument('--tiny-slots', type=int, default=4)
    parser.add_argument('--tiny-contexts', type=int, default=6)
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    model, snap = _restore_model(snapshot_path)
    student_model = None
    if args.student_snapshot:
        student_model, _ = _restore_model(Path(args.student_snapshot))
    dream_corpus = load_corpus(args.dream_corpus) if args.dream_corpus else None
    wake_corpus = load_corpus(args.wake_corpus) if args.wake_corpus else None

    print(f"Snapshot: {snapshot_path}")
    print(f"A slots: {len(snap['a_slots'])}, B edges: {len(snap['b_src'])}")
    print(f"Config: D_PHI={model.config.D_PHI}, CP_RANK={model.config.CP_RANK}, GAMMA_DECAY={model.config.GAMMA_DECAY}")
    print()

    slots = _select_slots(snap, top_slots=args.top_slots)
    print(f"Selected slots: {slots}")
    print()

    _print_edge_target_summary(model, snap, slots, max_edges=max(1, int(args.edge_vectors)))
    _print_dream_targets(
        model,
        snap,
        n_contexts=max(1, int(args.dream_contexts)),
        dream_topk=max(16, int(args.dream_topk)),
        student_model=student_model,
        dream_corpus=dream_corpus,
    )
    print()
    if wake_corpus is not None:
        _print_wake_targets(
            model,
            wake_corpus,
            n_contexts=max(1, int(args.wake_contexts)),
            wake_topk=max(1, int(args.wake_topk)),
            student_model=student_model,
            wake_offset=max(0, int(args.wake_offset)),
        )
        print()
    tiny_slots = slots[:max(1, int(args.tiny_slots))]
    _print_tiny_dream_targets(model, snap, tiny_slots, n_contexts=max(1, int(args.tiny_contexts)))


if __name__ == '__main__':
    main()
