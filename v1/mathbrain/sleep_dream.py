"""Dream sleep helper: build long synthetic slot-group sequences and replay EMA-Q."""

from __future__ import annotations

import numpy as np

from .config import MathBrainConfig
from .phi_encoder import create_phi_encoder
from .q_state import QState
from .wake_dataset import WakeDataset


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"_topk_indices expects 1D values, got {values.shape}")
    if len(values) == 0:
        return np.zeros((0,), dtype=np.int32)
    k = max(1, min(int(k), len(values)))
    if k == len(values):
        return np.argsort(values)[::-1].astype(np.int32, copy=False)
    part = np.argpartition(values, -k)[-k:]
    return part[np.argsort(values[part])[::-1]].astype(np.int32, copy=False)


def _softmax_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if logits.ndim != 1:
        raise ValueError(f"_softmax_probabilities expects 1D logits, got {logits.shape}")
    temperature = max(float(temperature), 1e-3)
    shifted = (logits - float(np.max(logits))) / temperature
    probs = np.exp(np.clip(shifted, -30.0, 30.0), dtype=np.float64)
    total = float(np.sum(probs))
    if total <= 1e-12 or not np.isfinite(total):
        return np.ones((len(logits),), dtype=np.float64) / max(1, len(logits))
    return probs / total


def _build_probe_steps(episode_len: int, probe_count: int) -> np.ndarray:
    episode_len = max(1, int(episode_len))
    probe_count = max(1, min(int(probe_count), episode_len))
    steps = np.linspace(1, episode_len, num=probe_count, dtype=np.int32) - 1
    return np.unique(np.clip(steps, 0, episode_len - 1)).astype(np.int32, copy=False)


def _build_slot_strength(entries: list) -> dict[int, float]:
    slot_strength: dict[int, float] = {}
    for src, _, vec in entries:
        vec_np = np.nan_to_num(np.asarray(vec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(np.linalg.norm(vec_np))
        if norm <= 1e-8:
            continue
        slot_strength[int(src)] = slot_strength.get(int(src), 0.0) + norm
    return slot_strength


def _estimate_ema_window(config: MathBrainConfig,
                         amplitude: float = 1.0,
                         max_steps: int = 4096) -> int:
    rho = np.asarray(config.rho, dtype=np.float32)
    threshold = max(float(getattr(config, 'THETA_Q', 0.0)), float(getattr(config, 'EPS_Q', 0.0)))
    threshold = max(threshold, 1e-8)
    amplitude = max(float(amplitude), 1e-6)
    for step in range(1, max_steps + 1):
        q_vec = amplitude * (rho ** step)
        if float(np.linalg.norm(q_vec)) <= threshold:
            return step
    return max_steps


def _build_slot_distribution(entries: list,
                             all_slots: np.ndarray,
                             uniform_mix: float) -> np.ndarray:
    strength_map = _build_slot_strength(entries)
    weights = np.asarray([strength_map.get(int(slot), 0.0) for slot in all_slots.tolist()], dtype=np.float64)
    if float(np.sum(weights)) <= 1e-12:
        weights = np.ones_like(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    uniform_mix = min(max(float(uniform_mix), 0.0), 1.0)
    weights = (1.0 - uniform_mix) * weights + (uniform_mix / len(weights))
    return weights / np.sum(weights)


def _sample_slot_group(all_slots: np.ndarray,
                       slot_weights: np.ndarray,
                       group_size: int,
                       rng: np.random.RandomState,
                       exclude: np.ndarray | None = None) -> np.ndarray:
    if exclude is not None and len(exclude) > 0:
        mask = ~np.isin(all_slots, exclude)
        candidate_slots = all_slots[mask]
        candidate_weights = slot_weights[mask]
    else:
        candidate_slots = all_slots
        candidate_weights = slot_weights

    if len(candidate_slots) == 0:
        return np.zeros((0,), dtype=np.int32)
    if len(candidate_slots) <= group_size:
        return np.asarray(candidate_slots, dtype=np.int32, copy=True)

    candidate_weights = np.asarray(candidate_weights, dtype=np.float64)
    if float(np.sum(candidate_weights)) <= 1e-12:
        candidate_weights = np.ones_like(candidate_weights, dtype=np.float64)
    candidate_weights = candidate_weights / np.sum(candidate_weights)

    idx = rng.choice(len(candidate_slots), size=group_size, replace=False, p=candidate_weights)
    return candidate_slots[idx].astype(np.int32, copy=False)


def _build_slot_sequence(all_slots: np.ndarray,
                         entries: list,
                         seq_len: int,
                         group_size: int,
                         n_waves: int,
                         uniform_mix: float,
                         rng: np.random.RandomState) -> list[dict[int, float]]:
    n_slots = len(all_slots)
    if n_slots == 0:
        return []

    group_size = max(1, min(int(group_size), n_slots))
    n_waves = max(1, int(n_waves))
    slot_weights = _build_slot_distribution(entries, all_slots=all_slots, uniform_mix=uniform_mix)
    weight_scale = slot_weights / (np.mean(slot_weights) + 1e-12)
    slot_bias = 0.15 * (weight_scale.astype(np.float32, copy=False) - 1.0)

    carrier_periods = np.geomspace(3.0, max(8.0, float(seq_len) / 2.0), num=n_waves).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_waves, n_slots)).astype(np.float32)
    amplitudes = rng.uniform(0.5, 1.0, size=(n_waves, 1)).astype(np.float32)

    envelope_cycles = max(1, n_waves // 2)
    envelope_period = max(32.0, float(seq_len) / float(envelope_cycles))
    delta_min = 0.15
    delta_max = max(2.0, float(group_size))
    sample_pos = 0.0

    sequence: list[dict[int, float]] = []
    for step in range(int(seq_len)):
        phase = (2.0 * np.pi * float(step)) / envelope_period
        freq_norm = 0.5 * (1.0 + np.sin(phase))
        sample_pos += delta_min + (delta_max - delta_min) * freq_norm

        step_signal = np.zeros((n_slots,), dtype=np.float32)
        for wave_idx in range(n_waves):
            omega = (2.0 * np.pi * sample_pos) / float(carrier_periods[wave_idx])
            step_signal += amplitudes[wave_idx, 0] * np.cos(omega + phases[wave_idx])
        step_signal += slot_bias
        step_signal += 0.03 * rng.randn(n_slots).astype(np.float32)

        group_size_t = max(1, min(int(round(1.0 + (float(group_size) - 1.0) * freq_norm)), n_slots))
        temp_t = 0.35 + (1.65 * freq_norm)
        probs = _softmax_probabilities(step_signal, temperature=temp_t)
        sampled_local = rng.choice(n_slots, size=group_size_t, replace=False, p=probs)
        sequence.append({int(all_slots[int(local_idx)]): 1.0 for local_idx in sampled_local.tolist()})

    return sequence


def _build_slot_sequence_from_corpus(corpus: list[str] | None,
                                     retina,
                                     n_dreams: int,
                                     ema_window: int) -> tuple[list[dict[int, float] | None], dict]:
    if retina is None or not corpus:
        return [], {'reason': 'no_corpus'}

    sentence_sequences: list[list[dict[int, float]]] = []
    raw_steps_per_pass = 0
    contexts_per_pass = 0

    for sentence in corpus:
        words = WakeDataset._tokenize(sentence)
        if len(words) < 2:
            continue
        sentence_steps = [retina.encode(word) for word in words[:-1]]
        sentence_steps = [step for step in sentence_steps if step]
        if not sentence_steps:
            continue
        sentence_sequences.append(sentence_steps)
        raw_steps_per_pass += len(sentence_steps)
        contexts_per_pass += max(0, len(sentence_steps) - ema_window + 1)

    if not sentence_sequences:
        return [], {'reason': 'empty_corpus_steps'}
    if contexts_per_pass <= 0:
        return [], {
            'reason': 'insufficient_sentence_length',
            'raw_steps_per_pass': int(raw_steps_per_pass),
            'contexts_per_pass': int(contexts_per_pass),
            'ema_window': int(ema_window),
        }

    passes = max(1, int(np.ceil(float(max(1, n_dreams)) / float(contexts_per_pass))))
    sequence: list[dict[int, float] | None] = []
    for _ in range(passes):
        for sentence_steps in sentence_sequences:
            sequence.extend(sentence_steps)
            sequence.append(None)

    return sequence, {
        'corpus_sentences': int(len(sentence_sequences)),
        'raw_steps_per_pass': int(raw_steps_per_pass),
        'contexts_per_pass': int(contexts_per_pass),
        'corpus_passes': int(passes),
        'reset_count': int(passes * len(sentence_sequences)),
    }


def _pad_contexts(active_local_list: list[np.ndarray],
                  phi_list: list[np.ndarray],
                  phi_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ctx = len(active_local_list)
    width = max((len(active) for active in active_local_list), default=0)
    active_padded = np.zeros((n_ctx, width), dtype=np.int32)
    phi_padded = np.zeros((n_ctx, width, phi_dim), dtype=np.float32)
    active_mask = np.zeros((n_ctx, width), dtype=np.float32)

    for idx, (active_local, phi_rows) in enumerate(zip(active_local_list, phi_list)):
        count = len(active_local)
        if count == 0:
            continue
        active_padded[idx, :count] = active_local
        phi_padded[idx, :count] = phi_rows
        active_mask[idx, :count] = 1.0

    return active_padded, phi_padded, active_mask


def _subsample_context_items(items: list, n_keep: int) -> list:
    if len(items) <= n_keep:
        return items
    keep_idx = np.unique(np.linspace(0, len(items) - 1, num=n_keep, dtype=np.int64)).astype(np.int32, copy=False)
    return [items[int(idx)] for idx in keep_idx.tolist()]


def prepare_dream_distillation_data(b_memory,
                                    entries: list,
                                    all_slots: list[int],
                                    config: MathBrainConfig,
                                    a_knowledge,
                                    global_decay: float,
                                    retina,
                                    vocab_words: list[str],
                                    n_dreams: int = 512,
                                    dream_size: int = 8,
                                    candidate_topk: int = 32,
                                    n_waves: int = 0,
                                    episode_len: int = 12,
                                    probe_count: int = 3,
                                    uniform_mix: float = 0.25,
                                    rng_seed: int = 0,
                                    dream_corpus: list[str] | None = None,
                                    verbose: bool = True,
                                    debug_trace: bool = False) -> tuple[dict, dict]:
    """Construct dream probes from one long synthetic slot-group stream with sliding EMA-Q."""
    del global_decay, vocab_words, probe_count

    if not entries:
        return {}, {'n_dreams': 0, 'reason': 'no_entries'}
    if not all_slots:
        return {}, {'n_dreams': 0, 'reason': 'no_slots'}
    slot_to_local = {int(slot): idx for idx, slot in enumerate(all_slots)}
    local_slot_arr = np.asarray(all_slots, dtype=np.int32)
    candidate_topk_raw = int(candidate_topk)
    if candidate_topk_raw <= 0:
        candidate_topk = len(local_slot_arr)
        full_scope_candidates = True
    else:
        candidate_topk = max(1, min(candidate_topk_raw, len(local_slot_arr)))
        full_scope_candidates = candidate_topk == len(local_slot_arr)
    n_dreams = max(1, int(n_dreams))
    episode_len = max(1, int(episode_len))
    ema_window = _estimate_ema_window(config)
    seq_multiplier = max(1.0, float(getattr(config, 'SLEEP_DREAM_SEQ_MULTIPLIER', 1.0)))
    base_seq_len = max(int(episode_len), int(np.ceil(float(ema_window) * seq_multiplier)))
    total_steps = base_seq_len + n_dreams - 1

    q_state = QState(config)
    phi_encoder = create_phi_encoder(config)
    phi_dim = int(config.d)
    rng = np.random.RandomState(int(rng_seed))
    requested_source = str(getattr(config, 'SLEEP_DREAM_SEQUENCE_SOURCE', 'train')).lower()
    sequence_source = requested_source
    source_info: dict = {}
    slot_sequence: list[dict[int, float] | None] = []

    if requested_source == 'train':
        slot_sequence, source_info = _build_slot_sequence_from_corpus(
            corpus=dream_corpus,
            retina=retina,
            n_dreams=n_dreams,
            ema_window=ema_window,
        )
        if not slot_sequence:
            sequence_source = 'synthetic_fallback'

    if requested_source != 'train' or not slot_sequence:
        slot_sequence = _build_slot_sequence(
            all_slots=local_slot_arr,
            entries=entries,
            seq_len=total_steps,
            group_size=dream_size,
            n_waves=n_waves,
            uniform_mix=uniform_mix,
            rng=rng,
        )
        if requested_source != 'train':
            source_info = {}
    if not slot_sequence:
        return {}, {'n_dreams': 0, 'reason': 'empty_slot_sequence'}

    active_local_list: list[np.ndarray] = []
    phi_list: list[np.ndarray] = []
    candidate_local_list: list[np.ndarray] = []
    teacher_logits_list: list[np.ndarray] = []
    debug_steps = [] if debug_trace else None

    teacher_peaks = []
    teacher_energy_ratios = []
    raw_active_counts = []
    kept_active_counts = []
    realized_steps = 0
    zero_teacher_count = 0
    empty_after_scope_count = 0

    q_state.reset()
    steps_since_reset = 0
    min_steps_for_context = 1 if sequence_source == 'train' else ema_window
    for step_idx, slot_counts in enumerate(slot_sequence):
        if slot_counts is None:
            q_state.reset()
            steps_since_reset = 0
            continue
        q_state.update(slot_counts)
        steps_since_reset += 1
        if steps_since_reset < min_steps_for_context:
            continue

        active_slots_step, q_vals_step = q_state.get_active()
        if len(active_slots_step) == 0:
            continue

        phi_step = phi_encoder.encode(q_vals_step)
        active_slots_step = active_slots_step.astype(np.int32, copy=False)

        teacher_scores = b_memory.predict(active_slots_step, phi_step)
        if a_knowledge is not None and getattr(a_knowledge, 'has_knowledge', False):
            teacher_scores += a_knowledge.predict(active_slots_step, phi_step)

        teacher_local = np.nan_to_num(
            teacher_scores[local_slot_arr],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
        if float(np.max(np.abs(teacher_local))) <= 1e-8:
            zero_teacher_count += 1
            continue

        active_local_rows = []
        phi_rows = []
        for src_idx, src in enumerate(active_slots_step.tolist()):
            local_idx = slot_to_local.get(int(src))
            if local_idx is None:
                continue
            active_local_rows.append(int(local_idx))
            phi_rows.append(
                np.nan_to_num(phi_step[src_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
            )

        raw_active_counts.append(int(len(active_slots_step)))
        kept_active_counts.append(int(len(active_local_rows)))
        if not active_local_rows:
            empty_after_scope_count += 1
            continue

        active_local_arr = np.asarray(active_local_rows, dtype=np.int32)
        phi_arr = np.stack(phi_rows, axis=0).astype(np.float32, copy=False)
        candidate_local = _topk_indices(teacher_local, candidate_topk)
        teacher_logits = teacher_local[candidate_local].astype(np.float32, copy=False)

        full_energy = float(np.sum(teacher_local * teacher_local))
        topk_energy = float(np.sum(teacher_logits * teacher_logits))
        teacher_energy_ratios.append(topk_energy / (full_energy + 1e-8))

        active_local_list.append(active_local_arr)
        phi_list.append(phi_arr)
        candidate_local_list.append(candidate_local)
        teacher_logits_list.append(teacher_logits)
        teacher_peaks.append(float(np.max(np.abs(teacher_logits))))
        if debug_trace:
            debug_steps.append({
                'step_idx': int(step_idx),
                'slot_counts': {int(slot): float(value) for slot, value in slot_counts.items()},
                'active_slots_step': active_slots_step.astype(np.int32, copy=True),
                'q_vals_step': q_vals_step.astype(np.float32, copy=True),
                'phi_step': phi_step.astype(np.float32, copy=True),
                'teacher_local': teacher_local.astype(np.float32, copy=True),
                'candidate_local': candidate_local.astype(np.int32, copy=True),
                'teacher_logits': teacher_logits.astype(np.float32, copy=True),
            })
        realized_steps += 1

        if sequence_source != 'train' and len(active_local_list) >= n_dreams:
            break

    if not active_local_list:
        return {}, {
            'n_dreams': 0,
            'reason': 'all_sliding_q_zero_teacher',
            'zero_teacher_count': int(zero_teacher_count),
            'empty_after_scope_count': int(empty_after_scope_count),
        }

    if len(active_local_list) > n_dreams:
        active_local_list = _subsample_context_items(active_local_list, n_dreams)
        phi_list = _subsample_context_items(phi_list, n_dreams)
        candidate_local_list = _subsample_context_items(candidate_local_list, n_dreams)
        teacher_logits_list = _subsample_context_items(teacher_logits_list, n_dreams)
        teacher_peaks = _subsample_context_items(teacher_peaks, n_dreams)
        teacher_energy_ratios = _subsample_context_items(teacher_energy_ratios, n_dreams)
        if debug_trace:
            debug_steps = _subsample_context_items(debug_steps, n_dreams)

    active_padded, phi_padded, active_mask = _pad_contexts(active_local_list, phi_list, phi_dim=phi_dim)
    data = {
        'active_local': active_padded,
        'phi': phi_padded,
        'active_mask': active_mask,
        'candidate_local': np.stack(candidate_local_list, axis=0).astype(np.int32, copy=False),
        'teacher_logits': np.stack(teacher_logits_list, axis=0).astype(np.float32, copy=False),
    }
    if debug_trace:
        data['debug_trace'] = {
            'slot_sequence': [
                None if step is None else {int(slot): float(value) for slot, value in step.items()}
                for step in slot_sequence
            ],
            'realized_steps': debug_steps,
        }

    raw_active_mean = float(np.mean(raw_active_counts)) if raw_active_counts else 0.0
    kept_active_mean = float(np.mean(kept_active_counts)) if kept_active_counts else 0.0
    injected_counts = [len(slot_counts) for slot_counts in slot_sequence if slot_counts is not None]
    info = {
        'n_dreams': int(data['active_local'].shape[0]),
        'n_episodes': 1,
        'episode_len': int(total_steps),
        'base_seq_len': int(base_seq_len),
        'probe_count': int(data['active_local'].shape[0]),
        'probe_steps': ['sliding_q'],
        'candidate_topk': int(candidate_topk),
        'candidate_topk_raw': int(candidate_topk_raw),
        'full_scope_candidates': bool(full_scope_candidates),
        'context_width': int(data['active_local'].shape[1]),
        'sequence_source': str(sequence_source),
        'requested_source': str(requested_source),
        'raw_active_mean': raw_active_mean,
        'raw_active_max': int(np.max(raw_active_counts)) if raw_active_counts else 0,
        'kept_active_mean': kept_active_mean,
        'kept_active_max': int(np.max(kept_active_counts)) if kept_active_counts else 0,
        'scope_keep_ratio_mean': float(kept_active_mean / (raw_active_mean + 1e-8)),
        'teacher_peak_mean': float(np.mean(teacher_peaks)) if teacher_peaks else 0.0,
        'teacher_peak_max': float(np.max(teacher_peaks)) if teacher_peaks else 0.0,
        'teacher_topk_energy_mean': float(np.mean(teacher_energy_ratios)) if teacher_energy_ratios else 0.0,
        'word_pool_size': 0,
        'word_coverage': 0.0,
        'uniform_mix': float(uniform_mix),
        'zero_teacher_count': int(zero_teacher_count),
        'empty_after_scope_count': int(empty_after_scope_count),
        'ema_window': int(ema_window),
        'seq_multiplier': float(seq_multiplier),
        'group_size': int(max(1, min(int(dream_size), len(local_slot_arr)))),
        'injected_group_mean': float(np.mean(injected_counts)) if injected_counts else 0.0,
        'injected_group_min': int(np.min(injected_counts)) if injected_counts else 0,
        'injected_group_max': int(np.max(injected_counts)) if injected_counts else 0,
        'n_waves': int(max(1, int(n_waves))),
        'realized_steps': int(realized_steps),
        'sequence_len': int(sum(1 for step in slot_sequence if step is not None)),
        'sequence_events': int(len(slot_sequence)),
    }
    info.update(source_info)
    if verbose:
        print(
            f"  Dream data: contexts={info['n_dreams']}, mode=sliding_q, source={info['sequence_source']}, "
            f"seq={info['sequence_len']} (base={info['base_seq_len']}, ema={info['ema_window']}, mult={info['seq_multiplier']:.2f}), "
            f"group={info['injected_group_mean']:.2f}[{info['injected_group_min']},{info['injected_group_max']}], "
            f"waves={info['n_waves']}, width={info['context_width']}, "
            f"cand={info['candidate_topk']}{'(full)' if info['full_scope_candidates'] else ''}, "
            f"active={info['raw_active_mean']:.2f}->{info['kept_active_mean']:.2f} "
            f"(keep={info['scope_keep_ratio_mean']:.1%}), "
            f"topk_energy={info['teacher_topk_energy_mean']:.1%}, "
            f"zero_teacher={info['zero_teacher_count']}, peak_mean={info['teacher_peak_mean']:.4f}"
        )
    return data, info
