#!/usr/bin/env python3
"""独立实验脚本：树形 / 语料级并行 Wake 与 Wake-Sleep。"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.inference_fast import SparseMatrixDecoder
from mathbrain.wake_tree_trainer import TreeWakeTrainer


def load_corpus(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def evaluate_recall_serial(model, corpus):
    def tokenize(text: str):
        return [w.strip() for w in text.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').split() if w.strip()]

    correct = 0
    total = 0
    for sent in corpus:
        words = tokenize(sent)
        if len(words) < 2:
            continue
        for i in range(len(words) - 1):
            preds = model.predict_next(words[:i + 1], k=1)
            pred = preds[0][0] if preds else '?'
            total += 1
            if pred == words[i + 1]:
                correct += 1
    accuracy = correct / total * 100 if total else 0.0
    return accuracy, correct, total


def evaluate_recall_parallel(model, dataset):
    if model._decoder_dirty or model._decoder is None:
        model._decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)
        model._decoder_dirty = False

    decoder = model._decoder
    correct = 0
    total = 0
    K = model.config.K

    active_offsets = dataset.active_offsets
    target_offsets = dataset.target_offsets
    active_slots = dataset.active_slots
    phi_hat = dataset.phi_hat
    target_slots = dataset.target_slots

    for pos in range(dataset.total_positions):
        a0, a1 = int(active_offsets[pos]), int(active_offsets[pos + 1])
        t0, t1 = int(target_offsets[pos]), int(target_offsets[pos + 1])
        pos_active_slots = active_slots[a0:a1]
        pos_phi_hat = phi_hat[a0:a1]
        pos_target_slots = target_slots[t0:t1]

        scores_b = model.b.predict(pos_active_slots, pos_phi_hat)
        scores_a = model.a.predict(pos_active_slots, pos_phi_hat)
        merged = scores_b + scores_a
        pred_word = decoder.decode_top_k(merged, 1)[0][0]

        target_score = np.zeros((K,), dtype=np.float32)
        target_score[pos_target_slots] = 1.0
        gold_word = decoder.decode_top_k(target_score, 1)[0][0]

        total += 1
        if pred_word == gold_word:
            correct += 1

    accuracy = correct / total * 100 if total else 0.0
    return accuracy, correct, total


def evaluate_model(model, corpus, dataset, trainer, backend, eval_mode):
    if eval_mode in ('parallel', 'sparse'):
        if backend == 'mlx' and hasattr(dataset, 'slot_universe'):
            from mathbrain.evaluator_mlx import DenseEvaluatorMLX, SparseEvaluatorMLX
            evaluator_cls = SparseEvaluatorMLX if eval_mode == 'sparse' else DenseEvaluatorMLX
            result = evaluator_cls(model, dataset.slot_universe, trainer=trainer).evaluate_dataset(dataset)
            return {
                'accuracy': result.accuracy,
                'correct': result.correct,
                'total': result.total,
            }
        if hasattr(dataset, 'active_offsets'):
            acc, correct, total = evaluate_recall_parallel(model, dataset)
            return {
                'accuracy': acc,
                'correct': correct,
                'total': total,
            }

    acc, correct, total = evaluate_recall_serial(model, corpus)
    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
    }


def collect_a_stats(model):
    stats = model.a.stats()
    if not stats.get('has_knowledge'):
        return {
            'has_knowledge': False,
            'n_embeddings': 0,
            'E_src_norm_mean': 0.0,
            'E_tgt_norm_mean': 0.0,
        }
    return {
        'has_knowledge': True,
        'n_embeddings': int(stats['n_embeddings']),
        'E_src_norm_mean': float(stats['E_src_norm_mean']),
        'E_tgt_norm_mean': float(stats['E_tgt_norm_mean']),
    }


def current_b_stats(model, trainer):
    if hasattr(trainer, 'dense_stats') and getattr(trainer, 'b_dense', None) is not None:
        return trainer.dense_stats()
    return model.get_stats()['b']


def print_eval(label, result):
    print(f"{label}: {result['correct']}/{result['total']} = {result['accuracy']:.2f}%")


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(values)))
    if k >= len(values):
        return np.argsort(values)[::-1].astype(np.int32, copy=False)
    part = np.argpartition(values, -k)[-k:]
    return part[np.argsort(values[part])[::-1]].astype(np.int32, copy=False)


def build_sleep_analysis_snapshot(model, dataset, global_decay: float, max_positions: int = 512, topk: int = 16):
    del global_decay
    if getattr(dataset, 'total_positions', 0) <= 0:
        return {'summary': {'n_positions': 0}, 'contexts': []}

    if hasattr(dataset, 'active_offsets'):
        active_offsets = dataset.active_offsets
        active_slots_flat = dataset.active_slots
        phi_hat_flat = dataset.phi_hat
        target_offsets = dataset.target_offsets
        target_slots_flat = dataset.target_slots
    elif hasattr(dataset, 'slot_universe') and hasattr(dataset, 'active_pos_mx'):
        slot_universe = dataset.slot_universe.astype(np.int32, copy=False)
        active_pos = np.array(dataset.active_pos_mx, copy=False).astype(np.int32, copy=False)
        active_src_compact = np.array(dataset.active_src_mx, copy=False).astype(np.int32, copy=False)
        phi_hat_flat = np.array(dataset.active_phi_mx, copy=False).astype(np.float32, copy=False)
        active_slots_flat = slot_universe[active_src_compact]
        active_counts = np.bincount(active_pos, minlength=int(dataset.total_positions))
        active_offsets = np.concatenate([[0], np.cumsum(active_counts, dtype=np.int64)]).astype(np.int64, copy=False)
        target_offsets = dataset.target_offsets_np.astype(np.int64, copy=False)
        target_slots_flat = slot_universe[dataset.target_compact_np.astype(np.int32, copy=False)]
    else:
        return {'summary': {'n_positions': 0}, 'contexts': []}

    if model._decoder_dirty or model._decoder is None:
        model._decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)
        model._decoder_dirty = False
    decoder = model._decoder

    sample_n = min(max(1, int(max_positions)), int(dataset.total_positions))
    sampled_pos = np.linspace(0, dataset.total_positions - 1, num=sample_n, dtype=np.int64)
    sampled_pos = np.unique(sampled_pos).astype(np.int64, copy=False)

    contexts = []
    teacher_top1_gold = 0
    teacher_topk_gold = 0
    teacher_word_correct = 0
    a_top1_gold = 0
    a_word_correct = 0

    K = model.config.K
    for pos in sampled_pos.tolist():
        a0, a1 = int(active_offsets[pos]), int(active_offsets[pos + 1])
        t0, t1 = int(target_offsets[pos]), int(target_offsets[pos + 1])
        active = active_slots_flat[a0:a1].astype(np.int32, copy=True)
        phi = phi_hat_flat[a0:a1].astype(np.float32, copy=True)
        gold_slots = target_slots_flat[t0:t1].astype(np.int32, copy=True)
        if len(active) == 0 or len(gold_slots) == 0:
            continue

        scores_b = model.b.predict(active, phi)
        scores_a = model.a.predict(active, phi)
        teacher_scores = scores_b + scores_a
        teacher_topk = _topk_indices(teacher_scores, topk)
        a_topk = _topk_indices(scores_a, topk)

        teacher_top1_gold += int(int(teacher_topk[0]) in set(gold_slots.tolist()))
        teacher_topk_gold += int(np.intersect1d(teacher_topk, gold_slots, assume_unique=False).size > 0)
        a_top1_gold += int(int(a_topk[0]) in set(gold_slots.tolist()))

        target_score = np.zeros((K,), dtype=np.float32)
        target_score[gold_slots] = 1.0
        gold_word = decoder.decode_top_k(target_score, 1)[0][0]
        teacher_word = decoder.decode_top_k(teacher_scores, 1)[0][0]
        a_word = decoder.decode_top_k(scores_a, 1)[0][0]
        teacher_word_correct += int(teacher_word == gold_word)
        a_word_correct += int(a_word == gold_word)

        contexts.append({
            'active': active,
            'phi': phi,
            'gold_slots': gold_slots,
            'gold_word': gold_word,
            'teacher_top1': int(teacher_topk[0]),
            'teacher_topk': teacher_topk.astype(np.int32, copy=False),
            'teacher_word': teacher_word,
        })

    n = len(contexts)
    if n == 0:
        return {'summary': {'n_positions': 0}, 'contexts': []}

    return {
        'summary': {
            'n_positions': int(n),
            'teacher_slot_top1_gold': float(teacher_top1_gold / n),
            'teacher_slot_topk_gold': float(teacher_topk_gold / n),
            'teacher_word_acc': float(teacher_word_correct / n),
            'a_before_slot_top1_gold': float(a_top1_gold / n),
            'a_before_word_acc': float(a_word_correct / n),
            'topk': int(topk),
        },
        'contexts': contexts,
    }


def evaluate_student_against_snapshot(model, snapshot, topk: int = 16):
    contexts = snapshot.get('contexts', [])
    if not contexts:
        return {'n_positions': 0}

    if model._decoder_dirty or model._decoder is None:
        model._decoder = SparseMatrixDecoder(model.vocab, model.word_to_slots, model.config.K)
        model._decoder_dirty = False
    decoder = model._decoder

    top1_match = 0
    topk_overlap = 0.0
    teacher_rank_sum = 0.0
    slot_top1_gold = 0
    slot_topk_gold = 0
    word_acc = 0

    for ctx in contexts:
        scores_a = model.a.predict(ctx['active'], ctx['phi'])
        student_topk = _topk_indices(scores_a, topk)
        top1_match += int(int(student_topk[0]) == int(ctx['teacher_top1']))
        topk_overlap += float(np.intersect1d(student_topk, ctx['teacher_topk'], assume_unique=False).size / max(1, topk))
        teacher_rank_sum += float(1 + np.sum(scores_a > scores_a[int(ctx['teacher_top1'])]))
        slot_top1_gold += int(int(student_topk[0]) in set(ctx['gold_slots'].tolist()))
        slot_topk_gold += int(np.intersect1d(student_topk, ctx['gold_slots'], assume_unique=False).size > 0)
        word_acc += int(decoder.decode_top_k(scores_a, 1)[0][0] == ctx['gold_word'])

    n = len(contexts)
    return {
        'n_positions': int(n),
        'student_slot_top1_gold': float(slot_top1_gold / n),
        'student_slot_topk_gold': float(slot_topk_gold / n),
        'student_word_acc': float(word_acc / n),
        'teacher_top1_match': float(top1_match / n),
        'teacher_topk_overlap': float(topk_overlap / n),
        'teacher_top1_rank_mean': float(teacher_rank_sum / n),
        'topk': int(topk),
    }


def print_sleep_analysis(label, metrics):
    if not metrics or metrics.get('n_positions', 0) == 0:
        print(f"{label}: unavailable")
        return
    if 'teacher_slot_top1_gold' in metrics:
        print(
            f"{label}: n={metrics['n_positions']}, teacher_slot@1={metrics['teacher_slot_top1_gold']:.3f}, "
            f"teacher_slot@{metrics['topk']}={metrics['teacher_slot_topk_gold']:.3f}, "
            f"teacher_word={metrics['teacher_word_acc']:.3f}, A_before_word={metrics['a_before_word_acc']:.3f}"
        )
    else:
        print(
            f"{label}: n={metrics['n_positions']}, A_slot@1={metrics['student_slot_top1_gold']:.3f}, "
            f"A_slot@{metrics['topk']}={metrics['student_slot_topk_gold']:.3f}, A_word={metrics['student_word_acc']:.3f}, "
            f"teacher_top1_match={metrics['teacher_top1_match']:.3f}, overlap={metrics['teacher_topk_overlap']:.3f}, "
            f"teacher_rank_mean={metrics['teacher_top1_rank_mean']:.1f}"
        )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _extract_a_snapshot(model):
    a = model.a
    if not getattr(a, 'has_knowledge', False) or not getattr(a, 'E_src', None):
        return {
            'a_slots': np.empty((0,), dtype=np.int32),
            'a_E_src': np.empty((0, 0, 0), dtype=np.float32),
            'a_E_tgt': np.empty((0, 0, 0), dtype=np.float32),
        }

    slots = np.array(sorted(a.E_src.keys()), dtype=np.int32)
    E_src = np.stack([a.E_src[int(slot)].astype(np.float32, copy=False) for slot in slots], axis=0)
    E_tgt = np.stack([a.E_tgt[int(slot)].astype(np.float32, copy=False) for slot in slots], axis=0)
    return {
        'a_slots': slots,
        'a_E_src': E_src,
        'a_E_tgt': E_tgt,
    }


def _extract_b_snapshot(model):
    b = model.b
    n_entries = int(getattr(b, '_n_entries', 0))
    if n_entries <= 0:
        return {
            'b_src': np.empty((0,), dtype=np.int32),
            'b_tgt': np.empty((0,), dtype=np.int32),
            'b_vec': np.empty((0, getattr(b, 'D', 0)), dtype=np.float32),
            'b_time': np.empty((0,), dtype=np.int32),
        }

    return {
        'b_src': np.array(b._srcs[:n_entries], copy=True),
        'b_tgt': np.array(b._tgts[:n_entries], copy=True),
        'b_vec': np.array(b._vecs[:n_entries], copy=True),
        'b_time': np.array(b._times[:n_entries], copy=True),
    }


def initialize_random_a(model, std: float = 0.01, seed: int = 0):
    slots = sorted(int(slot) for slot in model._all_vocab_slots)
    if not slots:
        return {
            'initialized': False,
            'reason': 'no_vocab_slots',
            'n_slots': 0,
            'std': float(std),
            'seed': int(seed),
        }

    rng = np.random.RandomState(int(seed))
    d = int(model.a.d)
    D = int(model.a.D)
    model.a.E_src = {
        int(slot): (rng.randn(d, D).astype(np.float32) * float(std))
        for slot in slots
    }
    model.a.E_tgt = {
        int(slot): (rng.randn(d, D).astype(np.float32) * float(std))
        for slot in slots
    }
    model.a.has_knowledge = True
    model.a._build_cache()

    return {
        'initialized': True,
        'n_slots': int(len(slots)),
        'std': float(std),
        'seed': int(seed),
    }


def save_cycle_snapshot(snapshot_dir: Path, cycle_no: int, stage: str, model, extra: dict | None = None):
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stage_tag = stage.replace(' ', '_')
    stem = f"cycle{cycle_no:02d}_{stage_tag}"

    arrays = {}
    arrays.update(_extract_a_snapshot(model))
    arrays.update(_extract_b_snapshot(model))
    arrays['meta_cycle'] = np.array([cycle_no], dtype=np.int32)
    arrays['meta_stage'] = np.array([stage_tag])

    np.savez_compressed(snapshot_dir / f"{stem}.npz", **arrays)

    meta = {
        'cycle': int(cycle_no),
        'stage': stage_tag,
        'a_stats': collect_a_stats(model),
        'b_stats': model.get_stats()['b'],
    }
    if extra:
        meta.update(_jsonable(extra))

    with open(snapshot_dir / f"{stem}.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--cycles', type=int, default=1)
    parser.add_argument('--wake-sleep', action='store_true')
    parser.add_argument('--shard-size', type=int, default=4096)
    parser.add_argument('--phi-mode', type=str, default='chaos', choices=['chaos'])
    parser.add_argument('--d-phi', type=int, default=8)
    parser.add_argument('--phi-sigma', type=float, default=0.5)
    parser.add_argument('--chaos-n-folds', type=int, default=3)
    parser.add_argument('--chaos-alpha', type=float, default=2.0)
    parser.add_argument('--cp-rank', type=int, default=64)
    parser.add_argument('--eta', type=float, default=0.5)
    parser.add_argument('--lambda-wd', type=float, default=1e-4)
    parser.add_argument('--sleep-global-decay', type=float, default=0.85)
    parser.add_argument('--gamma-decay', type=float, default=0.9)
    parser.add_argument('--no-gamma-adaptive', action='store_true')
    parser.add_argument('--gamma-initial', type=float, default=0.98)
    parser.add_argument('--gamma-min', type=float, default=0.9)
    parser.add_argument('--gamma-decay-rate', type=float, default=0.9)
    parser.add_argument('--sleep-lr', type=float, default=0.01)
    parser.add_argument('--sleep-max-epochs', type=int, default=1000)
    parser.add_argument('--sleep-min-epochs', type=int, default=0)
    parser.add_argument('--sleep-patience', type=int, default=200)
    parser.add_argument('--sleep-rel-tol', type=float, default=0.0001)
    parser.add_argument('--sleep-memory-limit-gb', type=float, default=20.0)
    parser.add_argument('--sleep-memory-utilization', type=float, default=0.9)
    parser.add_argument('--sleep-loss', type=str, default='mse', choices=['mse', 'huber'])
    parser.add_argument('--sleep-huber-delta', type=float, default=1.0)
    parser.add_argument('--sleep-solver', type=str, default='adamw', choices=['adamw', 'als', 'dream'])
    parser.add_argument('--sleep-als-iters', type=int, default=6)
    parser.add_argument('--sleep-als-ridge', type=float, default=1e-3)
    parser.add_argument('--sleep-als-prox', type=float, default=1.0)
    parser.add_argument('--sleep-dream-samples', type=int, default=512)
    parser.add_argument('--sleep-dream-active', type=int, default=8)
    parser.add_argument('--sleep-dream-topk', type=int, default=0)
    parser.add_argument('--sleep-dream-phases', type=int, default=8)
    parser.add_argument('--sleep-dream-max-proto', type=int, default=4)
    parser.add_argument('--sleep-dream-batch-size', type=int, default=128)
    parser.add_argument('--sleep-dream-epochs', type=int, default=200)
    parser.add_argument('--sleep-dream-temperature', type=float, default=1.5)
    parser.add_argument('--sleep-dream-prox', type=float, default=1e-3)
    parser.add_argument('--sleep-dream-logit-weight', type=float, default=1.0)
    parser.add_argument('--sleep-dream-kl-weight', type=float, default=0.25)
    parser.add_argument('--sleep-dream-episode-len', type=int, default=12)
    parser.add_argument('--sleep-dream-seq-multiplier', type=float, default=1.0)
    parser.add_argument('--sleep-dream-probe-count', type=int, default=3)
    parser.add_argument('--sleep-dream-uniform-mix', type=float, default=0.25)
    parser.add_argument('--sleep-dream-sequence-source', type=str, default='train', choices=['train', 'synthetic'])
    parser.add_argument('--sleep-dream-warmstart-als-iters', type=int, default=0)
    parser.add_argument('--b-retention', type=float, default=0.0)
    parser.add_argument('--prune-ratio', type=float, default=0.8)
    parser.add_argument('--prune-max', type=int, default=200000)
    parser.add_argument('--no-prune-before-sleep', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--backend', choices=['cpu', 'mlx'], default='cpu')
    parser.add_argument('--eval-mode', choices=['parallel', 'sparse', 'serial', 'none'], default='parallel')
    parser.add_argument('--sleep-analysis-positions', type=int, default=512)
    parser.add_argument('--sleep-analysis-topk', type=int, default=16)
    parser.add_argument('--snapshot-dir', type=str, default='')
    parser.add_argument('--init-random-a', action='store_true')
    parser.add_argument('--init-random-a-std', type=float, default=0.01)
    parser.add_argument('--init-random-a-seed', type=int, default=0)
    args = parser.parse_args()

    cfg = MathBrainConfig()
    cfg.PHI_MODE = args.phi_mode
    cfg.D_PHI = args.d_phi
    cfg.PHI_SIGMA = args.phi_sigma
    cfg.CHAOS_N_FOLDS = args.chaos_n_folds
    cfg.CHAOS_ALPHA = args.chaos_alpha
    cfg.ETA = args.eta
    cfg.CP_RANK = args.cp_rank
    cfg.LAMBDA_WD = args.lambda_wd
    cfg.SLEEP_GLOBAL_DECAY = args.sleep_global_decay
    cfg.GAMMA_DECAY = args.gamma_decay
    cfg.GAMMA_ADAPTIVE = not args.no_gamma_adaptive
    cfg.GAMMA_INITIAL = args.gamma_initial
    cfg.GAMMA_MIN = args.gamma_min
    cfg.GAMMA_DECAY_RATE = args.gamma_decay_rate
    cfg.SLEEP_LR = args.sleep_lr
    cfg.SLEEP_MAX_EPOCHS = args.sleep_max_epochs
    cfg.SLEEP_MIN_EPOCHS = args.sleep_min_epochs
    cfg.SLEEP_PATIENCE = args.sleep_patience
    cfg.SLEEP_REL_TOL = args.sleep_rel_tol
    cfg.SLEEP_MINIBATCH_MEMORY_LIMIT_GB = args.sleep_memory_limit_gb
    cfg.SLEEP_MINIBATCH_UTILIZATION = args.sleep_memory_utilization
    cfg.SLEEP_LOSS = args.sleep_loss
    cfg.SLEEP_HUBER_DELTA = args.sleep_huber_delta
    cfg.SLEEP_SOLVER = args.sleep_solver
    cfg.SLEEP_ALS_ITERS = args.sleep_als_iters
    cfg.SLEEP_ALS_RIDGE = args.sleep_als_ridge
    cfg.SLEEP_ALS_PROX = args.sleep_als_prox
    cfg.SLEEP_DREAM_SAMPLES = args.sleep_dream_samples
    cfg.SLEEP_DREAM_ACTIVE = args.sleep_dream_active
    cfg.SLEEP_DREAM_TOPK = args.sleep_dream_topk
    cfg.SLEEP_DREAM_PHASES = args.sleep_dream_phases
    cfg.SLEEP_DREAM_MAX_PROTO = args.sleep_dream_max_proto
    cfg.SLEEP_DREAM_BATCH_SIZE = args.sleep_dream_batch_size
    cfg.SLEEP_DREAM_EPOCHS = args.sleep_dream_epochs
    cfg.SLEEP_DREAM_TEMPERATURE = args.sleep_dream_temperature
    cfg.SLEEP_DREAM_PROX = args.sleep_dream_prox
    cfg.SLEEP_DREAM_LOGIT_WEIGHT = args.sleep_dream_logit_weight
    cfg.SLEEP_DREAM_KL_WEIGHT = args.sleep_dream_kl_weight
    cfg.SLEEP_DREAM_EPISODE_LEN = args.sleep_dream_episode_len
    cfg.SLEEP_DREAM_SEQ_MULTIPLIER = args.sleep_dream_seq_multiplier
    cfg.SLEEP_DREAM_PROBE_COUNT = args.sleep_dream_probe_count
    cfg.SLEEP_DREAM_UNIFORM_MIX = args.sleep_dream_uniform_mix
    cfg.SLEEP_DREAM_SEQUENCE_SOURCE = args.sleep_dream_sequence_source
    cfg.SLEEP_DREAM_WARMSTART_ALS_ITERS = args.sleep_dream_warmstart_als_iters
    cfg._build_derived()

    model = MathBrain(cfg)
    if hasattr(model.b, '_prune_keep_ratio'):
        model.b._prune_keep_ratio = args.prune_ratio
    if hasattr(model.b, '_prune_max_entries'):
        model.b._prune_max_entries = args.prune_max

    corpus = load_corpus(args.corpus)

    if args.backend == 'mlx':
        from mathbrain.wake_tree_trainer_mlx import TreeWakeTrainerMLX
        trainer = TreeWakeTrainerMLX(model)
    else:
        trainer = TreeWakeTrainer(model)

    t_build = time.time()
    try:
        dataset = trainer.build_dataset(corpus, verbose=not args.quiet, shard_size=args.shard_size)
    except TypeError:
        dataset = trainer.build_dataset(corpus, verbose=not args.quiet)
    build_ms = (time.time() - t_build) * 1000
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None

    init_a_info = None
    if args.init_random_a:
        init_a_info = initialize_random_a(
            model,
            std=args.init_random_a_std,
            seed=args.init_random_a_seed,
        )
        if snapshot_dir is not None:
            save_cycle_snapshot(
                snapshot_dir,
                cycle_no=0,
                stage='post_random_a_init',
                model=model,
                extra={
                    'random_a_init': init_a_info,
                    'a_stats': collect_a_stats(model),
                    'b_stats': current_b_stats(model, trainer),
                },
            )

    if not args.wake_sleep:
        need_sync = args.eval_mode not in ('parallel', 'sparse')
        total_rounds = max(1, args.cycles) * args.rounds

        t_train = time.time()
        history = trainer.run_corpus_parallel(
            dataset,
            rounds=total_rounds,
            shard_size=args.shard_size,
            verbose=not args.quiet,
            sync_at_end=need_sync,
        )
        train_ms = (time.time() - t_train) * 1000

        sync_ms = 0.0
        if need_sync and hasattr(trainer, 'sync_to_hash'):
            t_sync = time.time()
            trainer.sync_to_hash()
            sync_ms = (time.time() - t_sync) * 1000
        elif snapshot_dir is not None and hasattr(trainer, 'sync_to_hash'):
            t_sync = time.time()
            trainer.sync_to_hash()
            sync_ms = (time.time() - t_sync) * 1000

        eval_ms = 0.0
        result = None
        if args.eval_mode != 'none':
            t_eval = time.time()
            result = evaluate_model(model, corpus, dataset, trainer, args.backend, args.eval_mode)
            eval_ms = (time.time() - t_eval) * 1000

        if snapshot_dir is not None:
            save_cycle_snapshot(
                snapshot_dir,
                cycle_no=1,
                stage='post_wake_only',
                model=model,
                extra={
                    'rounds': int(total_rounds),
                    'wake_eval': result,
                    'train_ms': train_ms,
                    'sync_ms': sync_ms,
                    'eval_ms': eval_ms,
                    'history': history,
                    'random_a_init': init_a_info,
                    'b_stats': current_b_stats(model, trainer),
                    'a_stats': collect_a_stats(model),
                },
            )

        print(f"build_ms: {build_ms:.1f}")
        print(f"train_ms: {train_ms:.1f}")
        print(f"sync_ms: {sync_ms:.1f}")
        print(f"eval_ms: {eval_ms:.1f}")
        if result is not None:
            print_eval('parallel wake', result)
        if hasattr(trainer, 'profile'):
            print(f"profile: {trainer.profile}")
        print(f"history: {history}")
        print(f"b_stats: {current_b_stats(model, trainer)}")
        print(f"a_stats: {collect_a_stats(model)}")
        return

    prune_before_sleep = not args.no_prune_before_sleep
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    cycle_history = []
    totals = {
        'train_ms': 0.0,
        'sync_ms': 0.0,
        'prune_ms': 0.0,
        'sleep_ms': 0.0,
        'eval_ms': 0.0,
    }
    final_result = None

    for cycle_idx in range(max(1, args.cycles)):
        cycle_no = cycle_idx + 1
        if not args.quiet:
            print('=' * 70)
            print(f'Cycle {cycle_no}/{max(1, args.cycles)}')
            print('=' * 70)

        t_wake = time.time()
        wake_history = trainer.run_corpus_parallel(
            dataset,
            rounds=args.rounds,
            shard_size=args.shard_size,
            verbose=not args.quiet,
            sync_at_end=False,
        )
        wake_train_ms = (time.time() - t_wake) * 1000
        totals['train_ms'] += wake_train_ms

        cycle_info = {
            'cycle': cycle_no,
            'wake_train_ms': wake_train_ms,
            'wake_history': wake_history,
            'a_wake': collect_a_stats(model),
        }

        if args.eval_mode in ('parallel', 'sparse'):
            t_eval = time.time()
            wake_eval = evaluate_model(model, corpus, dataset, trainer, args.backend, args.eval_mode)
            wake_eval_ms = (time.time() - t_eval) * 1000
            totals['eval_ms'] += wake_eval_ms
            cycle_info['wake_eval'] = wake_eval
            cycle_info['wake_eval_ms'] = wake_eval_ms
            if not args.quiet:
                print_eval('  Wake', wake_eval)

        t_sync = time.time()
        trainer.sync_to_hash()
        sync_ms = (time.time() - t_sync) * 1000
        totals['sync_ms'] += sync_ms
        cycle_info['sync_ms'] = sync_ms

        if args.eval_mode not in ('parallel', 'sparse') and args.eval_mode != 'none':
            t_eval = time.time()
            wake_eval = evaluate_model(model, corpus, dataset, trainer, args.backend, args.eval_mode)
            wake_eval_ms = (time.time() - t_eval) * 1000
            totals['eval_ms'] += wake_eval_ms
            cycle_info['wake_eval'] = wake_eval
            cycle_info['wake_eval_ms'] = wake_eval_ms
            if not args.quiet:
                print_eval('  Wake', wake_eval)

        cycle_info['b_before_prune'] = model.get_stats()['b']
        if snapshot_dir is not None:
            save_cycle_snapshot(
                snapshot_dir,
                cycle_no,
                'post_sync',
                model,
                extra={
                    'wake_eval': cycle_info.get('wake_eval'),
                    'sync_ms': sync_ms,
                    'b_before_prune': cycle_info['b_before_prune'],
                },
            )

        prune_ms = 0.0
        n_pruned = 0
        if prune_before_sleep and hasattr(model.b, 'prune_by_norm'):
            t_prune = time.time()
            n_pruned = model.b.prune_by_norm(keep_ratio=args.prune_ratio, max_entries=args.prune_max)
            prune_ms = (time.time() - t_prune) * 1000
            totals['prune_ms'] += prune_ms

        cycle_info['n_pruned'] = int(n_pruned)
        cycle_info['prune_ms'] = prune_ms
        cycle_info['b_after_prune'] = model.get_stats()['b']
        if snapshot_dir is not None:
            save_cycle_snapshot(
                snapshot_dir,
                cycle_no,
                'post_prune',
                model,
                extra={
                    'wake_eval': cycle_info.get('wake_eval'),
                    'n_pruned': cycle_info['n_pruned'],
                    'prune_ms': prune_ms,
                    'b_after_prune': cycle_info['b_after_prune'],
                },
            )

        sleep_snapshot = build_sleep_analysis_snapshot(
            model, dataset,
            global_decay=cfg.SLEEP_GLOBAL_DECAY,
            max_positions=args.sleep_analysis_positions,
            topk=args.sleep_analysis_topk,
        )
        cycle_info['sleep_teacher_snapshot'] = sleep_snapshot['summary']
        if not args.quiet:
            print_sleep_analysis('  Sleep teacher', sleep_snapshot['summary'])

        t_sleep = time.time()
        sleep_result = model.sleep(b_retention=args.b_retention, dream_corpus=corpus)
        sleep_ms = (time.time() - t_sleep) * 1000
        totals['sleep_ms'] += sleep_ms
        trainer.invalidate_dense_state()

        sleep_transfer = evaluate_student_against_snapshot(
            model, sleep_snapshot, topk=args.sleep_analysis_topk
        )
        cycle_info['sleep_transfer'] = sleep_transfer
        if not args.quiet:
            print_sleep_analysis('  Sleep transfer', sleep_transfer)

        cycle_info['sleep'] = sleep_result
        cycle_info['sleep_ms'] = sleep_ms
        cycle_info['a_sleep'] = collect_a_stats(model)
        cycle_info['b_after_sleep'] = model.get_stats()['b']
        if snapshot_dir is not None:
            save_cycle_snapshot(
                snapshot_dir,
                cycle_no,
                'post_sleep',
                model,
                extra={
                    'wake_eval': cycle_info.get('wake_eval'),
                    'sleep': sleep_result,
                    'sleep_ms': sleep_ms,
                    'sleep_transfer': sleep_transfer,
                    'sleep_teacher_snapshot': cycle_info['sleep_teacher_snapshot'],
                },
            )

        if args.eval_mode != 'none':
            t_eval = time.time()
            sleep_eval = evaluate_model(model, corpus, dataset, trainer, args.backend, args.eval_mode)
            sleep_eval_ms = (time.time() - t_eval) * 1000
            totals['eval_ms'] += sleep_eval_ms
            cycle_info['sleep_eval'] = sleep_eval
            cycle_info['sleep_eval_ms'] = sleep_eval_ms
            final_result = sleep_eval
            if not args.quiet:
                print_eval('  Sleep', sleep_eval)
            if snapshot_dir is not None:
                save_cycle_snapshot(
                    snapshot_dir,
                    cycle_no,
                    'post_sleep_eval',
                    model,
                    extra={
                        'wake_eval': cycle_info.get('wake_eval'),
                        'sleep_eval': sleep_eval,
                        'sleep_eval_ms': sleep_eval_ms,
                        'sleep': sleep_result,
                        'sleep_transfer': sleep_transfer,
                    },
                )

        cycle_history.append(cycle_info)

    print(f"build_ms: {build_ms:.1f}")
    print(f"train_ms: {totals['train_ms']:.1f}")
    print(f"sync_ms: {totals['sync_ms']:.1f}")
    print(f"prune_ms: {totals['prune_ms']:.1f}")
    print(f"sleep_ms: {totals['sleep_ms']:.1f}")
    print(f"eval_ms: {totals['eval_ms']:.1f}")
    if final_result is not None:
        print_eval('parallel wake-sleep', final_result)
    if hasattr(trainer, 'profile'):
        print(f"profile: {trainer.profile}")
    print(f"history: {cycle_history}")
    print(f"b_stats: {current_b_stats(model, trainer)}")
    print(f"a_stats: {collect_a_stats(model)}")


if __name__ == '__main__':
    main()
