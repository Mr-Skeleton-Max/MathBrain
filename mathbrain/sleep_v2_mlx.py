"""Sleep 巩固 v2 MLX: Slice-wise 矩阵分解 + MLX 加速

核心改进：
1. 使用 MLX 替代 PyTorch，针对 Apple Silicon 优化
2. E_src/E_tgt 形状为 (n_slots, D, d)，sum 在最后维度（内存连续）
3. 用 AdamW weight_decay 替代正交约束
4. 移除随机采样，依赖 B 裁剪控制样本数

数学公式：
  对于每个 B[src,tgt] 条目 (D 维向量):
    A[tgt, src] = Σ_r E_tgt[tgt,:,r] * E_src[src,:,r]  → (D,)
    目标：A_new ≈ γ·A_old + B
"""

from dataclasses import dataclass
import numpy as np

try:
    import mlx.core as mx
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from .config import MathBrainConfig
from .sleep_als import solve_sleep_als
from .sleep_dream import prepare_dream_distillation_data


@dataclass
class SleepResultV2:
    n_new: int = 0
    n_rehearsal: int = 0
    epochs: int = 0
    final_loss: float = 0.0
    success: bool = False


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return (numerator + denominator - 1) // denominator


def _resolve_sleep_global_decay(raw_decay: float) -> tuple[float, bool]:
    decay = float(raw_decay)
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"SLEEP_GLOBAL_DECAY must be in [0, 1], got {decay}")
    if decay == 0.0:
        return 1.0, True
    return decay, False


def _reseed_zero_bilinear_factors(E_src_np: np.ndarray,
                                  E_tgt_np: np.ndarray,
                                  scale: float = 1e-2) -> tuple[np.ndarray, np.ndarray, bool]:
    if E_src_np.size == 0 or E_tgt_np.size == 0:
        return E_src_np, E_tgt_np, False
    src_zero = float(np.max(np.abs(E_src_np))) <= 1e-12
    tgt_zero = float(np.max(np.abs(E_tgt_np))) <= 1e-12
    if not (src_zero or tgt_zero):
        return E_src_np, E_tgt_np, False

    E_src_np = E_src_np + (np.random.randn(*E_src_np.shape).astype(np.float32) * scale)
    E_tgt_np = E_tgt_np + (np.random.randn(*E_tgt_np.shape).astype(np.float32) * scale)
    return E_src_np, E_tgt_np, True


def _resolve_mlx_memory_budget_bytes(cfg: MathBrainConfig) -> tuple[int, dict]:
    device_info = {}
    try:
        device_info = mx.device_info()
    except Exception:
        try:
            device_info = mx.metal.device_info()
        except Exception:
            device_info = {}

    device_limit_bytes = int(
        device_info.get('max_recommended_working_set_size')
        or device_info.get('memory_size')
        or 0
    )
    configured_limit_gb = max(float(getattr(cfg, 'SLEEP_MINIBATCH_MEMORY_LIMIT_GB', 0.0)), 0.0)
    configured_limit_bytes = int(configured_limit_gb * (1000 ** 3)) if configured_limit_gb > 0 else 0

    if configured_limit_bytes > 0 and device_limit_bytes > 0:
        return min(configured_limit_bytes, device_limit_bytes), device_info
    if configured_limit_bytes > 0:
        return configured_limit_bytes, device_info
    return device_limit_bytes, device_info


def _get_mlx_active_memory_bytes() -> int:
    try:
        return int(mx.get_active_memory())
    except Exception:
        try:
            return int(mx.metal.get_active_memory())
        except Exception:
            return 0


def _plan_sleep_batches(n_new: int,
                        n_rehearsal: int,
                        sample_bytes: int,
                        active_memory_bytes: int,
                        memory_limit_bytes: int,
                        utilization: float) -> dict:
    total_samples = n_new + n_rehearsal
    full_data_bytes = total_samples * sample_bytes
    target_working_set_bytes = int(memory_limit_bytes * utilization) if memory_limit_bytes > 0 else 0
    full_working_set_bytes = active_memory_bytes + full_data_bytes

    plan = {
        'use_minibatch': False,
        'new_batch_size': n_new,
        'rehearsal_batch_size': n_rehearsal,
        'n_batches': 1,
        'full_working_set_bytes': full_working_set_bytes,
        'batch_working_set_bytes': full_working_set_bytes,
        'target_working_set_bytes': target_working_set_bytes,
        'active_memory_bytes': active_memory_bytes,
        'memory_limit_bytes': memory_limit_bytes,
        'hard_limit_exceeded': memory_limit_bytes > 0 and full_working_set_bytes > memory_limit_bytes,
        'strict_utilization_met': full_working_set_bytes >= target_working_set_bytes,
        'reason': 'full_batch',
    }

    if total_samples == 0 or sample_bytes <= 0 or memory_limit_bytes <= 0:
        return plan

    if full_working_set_bytes <= target_working_set_bytes:
        plan['reason'] = 'below_activation_threshold'
        return plan

    max_batch_data_bytes = memory_limit_bytes - active_memory_bytes
    if max_batch_data_bytes <= 0:
        plan['reason'] = 'no_available_headroom'
        return plan

    n_batches = max(2, _ceil_div(full_data_bytes, max_batch_data_bytes))
    new_batch_size = 0
    rehearsal_batch_size = 0
    batch_working_set_bytes = full_working_set_bytes

    while True:
        new_batch_size = _ceil_div(n_new, n_batches) if n_new > 0 else 0
        rehearsal_batch_size = _ceil_div(n_rehearsal, n_batches) if n_rehearsal > 0 else 0
        batch_data_bytes = (new_batch_size + rehearsal_batch_size) * sample_bytes
        batch_working_set_bytes = active_memory_bytes + batch_data_bytes
        if batch_working_set_bytes <= memory_limit_bytes or n_batches >= total_samples:
            break
        n_batches += 1

    strict_utilization_met = batch_working_set_bytes >= target_working_set_bytes
    hard_limit_exceeded = full_working_set_bytes > memory_limit_bytes

    if not hard_limit_exceeded and not strict_utilization_met:
        plan['reason'] = 'underfilled_batches'
        return plan

    plan.update({
        'use_minibatch': n_batches > 1,
        'new_batch_size': new_batch_size,
        'rehearsal_batch_size': rehearsal_batch_size,
        'n_batches': n_batches,
        'batch_working_set_bytes': batch_working_set_bytes,
        'hard_limit_exceeded': hard_limit_exceeded,
        'strict_utilization_met': strict_utilization_met,
        'reason': 'forced_by_hard_limit' if hard_limit_exceeded and not strict_utilization_met else 'full_batch_above_activation',
    })
    return plan


def consolidate_v2_mlx(b_memory, a_knowledge, active_slots: set,
                       config: MathBrainConfig = None,
                       b_retention: float = 0.0,
                       cp_rank: int = 384,
                       lambda_wd: float = 1e-4,
                       retina=None,
                       vocab_words: list[str] | None = None,
                       dream_corpus: list[str] | None = None) -> SleepResultV2:
    """Sleep 巩固 v2 MLX: slice-wise 矩阵分解 + MLX 加速

    Args:
        b_memory: B Memory 实例
        a_knowledge: AKnowledgeV2 实例
        active_slots: 活跃槽位集合
        config: 配置
        b_retention: B Memory 保留比例 (0-1)
        cp_rank: E_src/E_tgt 的 rank 维度 d
        lambda_wd: AdamW weight decay (替代正交约束)
    """
    cfg = config or MathBrainConfig()
    result = SleepResultV2()
    trace_full_io = bool(getattr(cfg, 'SLEEP_TRACE_FULL_IO', False))
    trace_max_contexts = max(1, int(getattr(cfg, 'SLEEP_TRACE_MAX_CONTEXTS', 8)))

    if not HAS_MLX:
        print("  Sleep v2 MLX: MLX 不可用，回退到 PyTorch 版本")
        # 回退到 PyTorch 版本
        from .sleep_v2 import consolidate_v2
        return consolidate_v2(
            b_memory, a_knowledge, active_slots, config,
            b_retention=b_retention, cp_rank=cp_rank, lambda_wd=lambda_wd,
            retina=retina, vocab_words=vocab_words, dream_corpus=dream_corpus,
        )

    # 收集 B 样本（不做随机采样，依赖 prune_by_norm 控制）
    entries = b_memory.get_entries()
    if not entries:
        print("  Sleep v2 MLX: 无新知识样本")
        return result

    involved_slots = set()
    for src, tgt, _ in entries:
        involved_slots.add(src)
        involved_slots.add(tgt)

    active_slot_set = {int(slot) for slot in active_slots}
    overlap_slots = involved_slots & active_slot_set
    active_only_slots = active_slot_set - involved_slots
    involved_only_slots = involved_slots - active_slot_set
    legacy_scope_slots = involved_slots | active_slot_set

    sleep_solver = str(getattr(cfg, 'SLEEP_SOLVER', 'adamw')).lower()

    n_new = len(entries)
    all_slots = sorted(legacy_scope_slots if sleep_solver == 'dream' else involved_slots)
    slot_to_idx = {s: i for i, s in enumerate(all_slots)}
    n_slots = len(all_slots)
    d = cp_rank
    D = a_knowledge.D    # phi 维度

    print(f"  Sleep v2 MLX: {n_new} 样本, {n_slots} 槽位, d={d}, D={D}")
    print(
        f"  [诊断] Sleep scope: involved={len(involved_slots)}, active_arg={len(active_slot_set)}, "
        f"overlap={len(overlap_slots)}, active_only={len(active_only_slots)}, "
        f"involved_only={len(involved_only_slots)}, legacy_union={len(legacy_scope_slots)}, "
        f"current_scope={n_slots}"
    )

    # 初始化参数: E_src (n_slots, D, d), E_tgt (n_slots, D, d)
    # 注意：使用 (D, d) 布局，sum 在最后维度更快
    if not a_knowledge.has_knowledge:
        # 首次 Sleep: 随机初始化 * 0.01
        E_src_np = np.random.randn(n_slots, D, d).astype(np.float32) * 0.01
        E_tgt_np = np.random.randn(n_slots, D, d).astype(np.float32) * 0.01
    else:
        # 后续 Sleep: 从已有知识加载
        E_src_np = np.zeros((n_slots, D, d), dtype=np.float32)
        E_tgt_np = np.zeros((n_slots, D, d), dtype=np.float32)
        for slot, idx in slot_to_idx.items():
            if slot in a_knowledge.E_src:
                # 原始存储是 (d, D)，需要转置为 (D, d)
                E_src_np[idx] = a_knowledge.E_src[slot].T
                E_tgt_np[idx] = a_knowledge.E_tgt[slot].T
            else:
                E_src_np[idx] = np.random.randn(D, d).astype(np.float32) * 0.01
                E_tgt_np[idx] = np.random.randn(D, d).astype(np.float32) * 0.01

    raw_global_decay = float(getattr(cfg, 'SLEEP_GLOBAL_DECAY', 1.0))
    global_decay, decay_disabled = _resolve_sleep_global_decay(raw_global_decay)
    if a_knowledge.has_knowledge and global_decay < 1.0:
        factor_decay = float(np.sqrt(global_decay))
        E_src_np *= factor_decay
        E_tgt_np *= factor_decay
        print(f"  Sleep global decay: edge={global_decay:.6f}, factor={factor_decay:.6f}")
    elif decay_disabled:
        print("  Sleep global decay: disabled (raw=0 interpreted as no decay)")

    if sleep_solver == 'dream':
        E_src_np, E_tgt_np, reseeded = _reseed_zero_bilinear_factors(E_src_np, E_tgt_np)
        if reseeded:
            print("  Sleep init safeguard: reseeded zero A factors to escape bilinear zero-gradient trap")

    # 转换为 MLX arrays
    params = {
        'E_src': mx.array(E_src_np),
        'E_tgt': mx.array(E_tgt_np),
    }

    # 准备 B 样本数据 (D 维向量)
    src_indices, tgt_indices, b_vecs_np = [], [], []
    for src, tgt, b_vec in entries:
        src_indices.append(slot_to_idx[src])
        tgt_indices.append(slot_to_idx[tgt])
        b_vecs_np.append(b_vec)

    src_indices = np.array(src_indices, dtype=np.int32)
    tgt_indices = np.array(tgt_indices, dtype=np.int32)
    b_vecs_np = np.array(b_vecs_np, dtype=np.float32)

    # --- 诊断: B 向量统计 ---
    b_norms = np.linalg.norm(b_vecs_np, axis=1)
    print(f"  [诊断] B向量: norm mean={np.mean(b_norms):.4f}, "
          f"max={np.max(b_norms):.4f}, min={np.min(b_norms):.6f}, "
          f"std={np.std(b_norms):.4f}")

    # 转换为 MLX arrays
    src_indices_mx = mx.array(src_indices)
    tgt_indices_mx = mx.array(tgt_indices)
    b_vecs_mx = mx.array(b_vecs_np)

    # === Target = γ·A_old + B (预计算，固定，全程 MLX) ===
    # 计算 A_old
    E_src_indexed = params['E_src'][src_indices_mx]  # (n_new, D, d)
    E_tgt_indexed = params['E_tgt'][tgt_indices_mx]  # (n_new, D, d)
    a_old_mx = mx.sum(E_tgt_indexed * E_src_indexed, axis=2)  # (n_new, D)

    if cfg.GAMMA_ADAPTIVE and a_knowledge.has_knowledge:
        src_slots = np.array([all_slots[idx] for idx in src_indices], dtype=np.int32)
        tgt_slots = np.array([all_slots[idx] for idx in tgt_indices], dtype=np.int32)
        gamma_src = a_knowledge.get_adaptive_gamma_batch(
            src_slots, cfg.GAMMA_INITIAL, cfg.GAMMA_MIN, cfg.GAMMA_DECAY_RATE)
        gamma_tgt = a_knowledge.get_adaptive_gamma_batch(
            tgt_slots, cfg.GAMMA_INITIAL, cfg.GAMMA_MIN, cfg.GAMMA_DECAY_RATE)
        gamma_vec_np = np.maximum(gamma_src, gamma_tgt)[:, None]  # (n_new, 1)
        gamma_vec_mx = mx.array(gamma_vec_np)
        target_mx = gamma_vec_mx * a_old_mx + b_vecs_mx
    else:
        gamma = cfg.GAMMA_DECAY
        target_mx = gamma * a_old_mx + b_vecs_mx  # 全程 MLX

    # --- 诊断: target / A_old / B 范数 ---
    mx.eval(a_old_mx, target_mx)
    a_old_np = np.array(a_old_mx)
    target_np = np.array(target_mx)
    a_old_norms = np.linalg.norm(a_old_np, axis=1)
    target_norms = np.linalg.norm(target_np, axis=1)
    print(f"  [诊断] A_old: norm mean={np.mean(a_old_norms):.4f}, "
          f"max={np.max(a_old_norms):.4f}, std={np.std(a_old_norms):.4f}")
    print(f"  [诊断] Target: norm mean={np.mean(target_norms):.4f}, "
          f"max={np.max(target_norms):.4f}, std={np.std(target_norms):.4f}")
    # A_old 中有多少异常大的值？
    a_old_abs_max = np.max(np.abs(a_old_np))
    a_old_above_2 = np.sum(np.max(np.abs(a_old_np), axis=1) > 2.0)
    a_old_above_5 = np.sum(np.max(np.abs(a_old_np), axis=1) > 5.0)
    print(f"  [诊断] A_old abs_max={a_old_abs_max:.4f}, "
          f"|A|>2: {a_old_above_2}/{n_new}, |A|>5: {a_old_above_5}/{n_new}")
    if trace_full_io:
        print("  [TRACE] Sleep edge targets")
        for edge_idx in range(n_new):
            src_slot = int(all_slots[int(src_indices[edge_idx])])
            tgt_slot = int(all_slots[int(tgt_indices[edge_idx])])
            print(f"    edge[{edge_idx}] src={src_slot} tgt={tgt_slot}")
            print(f"      A_old ={np.array2string(a_old_np[edge_idx], precision=4, suppress_small=False)}")
            print(f"      B     ={np.array2string(b_vecs_np[edge_idx], precision=4, suppress_small=False)}")
            print(f"      target={np.array2string(target_np[edge_idx], precision=4, suppress_small=False)}")

    # === 不做 rehearsal：Sleep 仅吸收 B 中记录的增量目标 ===
    n_rehearsal = 0
    rehearsal_src_mx = None
    rehearsal_tgt_mx = None
    rehearsal_targets_mx = None

    # --- 诊断: 排练 ---
    if n_rehearsal > 0:
        mx.eval(rehearsal_targets_mx)
        reh_np = np.array(rehearsal_targets_mx)
        reh_norms = np.linalg.norm(reh_np, axis=1)
        print(f"  [诊断] 排练: {n_rehearsal} 样本, target norm mean={np.mean(reh_norms):.4f}, "
              f"max={np.max(reh_norms):.4f}")
    else:
        print(f"  [诊断] 排练: 0 样本 (disabled)")

    # --- 诊断: 初始 E_src/E_tgt 范数 ---
    E_src_init_norms = np.linalg.norm(E_src_np.reshape(n_slots, -1), axis=1)
    E_tgt_init_norms = np.linalg.norm(E_tgt_np.reshape(n_slots, -1), axis=1)
    print(f"  [诊断] E_src初始: norm mean={np.mean(E_src_init_norms):.4f}, "
          f"max={np.max(E_src_init_norms):.4f}")
    print(f"  [诊断] E_tgt初始: norm mean={np.mean(E_tgt_init_norms):.4f}, "
          f"max={np.max(E_tgt_init_norms):.4f}")

    # === 求解器选择: AdamW 或 ALS ===
    use_huber = cfg.SLEEP_LOSS == "huber"
    huber_delta = cfg.SLEEP_HUBER_DELTA

    if sleep_solver == 'als':
        als_iters = max(1, int(getattr(cfg, 'SLEEP_ALS_ITERS', 6)))
        als_ridge = max(0.0, float(getattr(cfg, 'SLEEP_ALS_RIDGE', 1e-3)))
        als_prox = max(0.0, float(getattr(cfg, 'SLEEP_ALS_PROX', 1.0)))
        if use_huber:
            print(f"  Sleep ALS: Huber unsupported, fallback to MSE surrogate (delta={huber_delta})")
        print(f"  Sleep solver: ALS, iters={als_iters}, ridge={als_ridge:.4g}, prox={als_prox:.4g}")
        E_src_np, E_tgt_np, als_info = solve_sleep_als(
            E_src_np,
            E_tgt_np,
            src_indices,
            tgt_indices,
            target_np,
            max_iters=als_iters,
            ridge=als_ridge,
            prox=als_prox,
            rel_tol=cfg.SLEEP_REL_TOL,
            verbose=True,
        )
        best_loss = float(als_info['final_loss'])
        epoch = int(als_info['iterations']) - 1
    elif sleep_solver == 'dream':
        dream_samples = max(1, int(getattr(cfg, 'SLEEP_DREAM_SAMPLES', 512)))
        dream_size = max(1, int(getattr(cfg, 'SLEEP_DREAM_ACTIVE', 8)))
        dream_topk = int(getattr(cfg, 'SLEEP_DREAM_TOPK', 0))
        dream_waves = max(1, int(getattr(cfg, 'SLEEP_DREAM_PHASES', 6)))
        dream_batch_size = max(1, int(getattr(cfg, 'SLEEP_DREAM_BATCH_SIZE', 128)))
        dream_epochs = max(1, int(getattr(cfg, 'SLEEP_DREAM_EPOCHS', 200)))
        dream_temp = max(1e-3, float(getattr(cfg, 'SLEEP_DREAM_TEMPERATURE', 1.5)))
        dream_prox = max(0.0, float(getattr(cfg, 'SLEEP_DREAM_PROX', 1e-3)))
        dream_logit_weight = max(0.0, float(getattr(cfg, 'SLEEP_DREAM_LOGIT_WEIGHT', 1.0)))
        dream_kl_weight = max(0.0, float(getattr(cfg, 'SLEEP_DREAM_KL_WEIGHT', 0.25)))
        dream_episode_len = max(1, int(getattr(cfg, 'SLEEP_DREAM_EPISODE_LEN', 12)))
        dream_probe_count = max(1, int(getattr(cfg, 'SLEEP_DREAM_PROBE_COUNT', 3)))
        dream_uniform_mix = min(max(float(getattr(cfg, 'SLEEP_DREAM_UNIFORM_MIX', 0.25)), 0.0), 1.0)
        dream_warmstart_als_iters = max(0, int(getattr(cfg, 'SLEEP_DREAM_WARMSTART_ALS_ITERS', 0)))

        print(
            f"  Sleep solver: DREAM, episodes={dream_samples}, topk={'full' if dream_topk <= 0 else dream_topk}, "
            f"T={dream_episode_len}, probes={dream_probe_count}, uniform={dream_uniform_mix:.2f}, "
            f"batch={dream_batch_size}, epochs={dream_epochs}, temp={dream_temp:.3f}, prox={dream_prox:.4g}, "
            f"logit_w={dream_logit_weight:.3f}, kl_w={dream_kl_weight:.3f}"
        )

        if dream_warmstart_als_iters > 0:
            als_ridge = max(0.0, float(getattr(cfg, 'SLEEP_ALS_RIDGE', 1e-3)))
            als_prox = max(0.0, float(getattr(cfg, 'SLEEP_ALS_PROX', 1.0)))
            print(
                f"  Dream warmstart: ALS iters={dream_warmstart_als_iters}, "
                f"ridge={als_ridge:.4g}, prox={als_prox:.4g}"
            )
            E_src_np, E_tgt_np, _ = solve_sleep_als(
                E_src_np,
                E_tgt_np,
                src_indices,
                tgt_indices,
                target_np,
                max_iters=dream_warmstart_als_iters,
                ridge=als_ridge,
                prox=als_prox,
                rel_tol=cfg.SLEEP_REL_TOL,
                verbose=False,
            )
            params['E_src'] = mx.array(E_src_np)
            params['E_tgt'] = mx.array(E_tgt_np)

        dream_data, dream_info = prepare_dream_distillation_data(
            b_memory=b_memory,
            entries=entries,
            all_slots=all_slots,
            config=cfg,
            a_knowledge=a_knowledge,
            global_decay=global_decay,
            retina=retina,
            vocab_words=list(vocab_words or []),
            n_dreams=dream_samples,
            dream_size=dream_size,
            candidate_topk=dream_topk,
            n_waves=dream_waves,
            episode_len=dream_episode_len,
            probe_count=dream_probe_count,
            uniform_mix=dream_uniform_mix,
            rng_seed=0,
            dream_corpus=dream_corpus,
            verbose=True,
            debug_trace=trace_full_io,
        )
        if not dream_data:
            raise ValueError(f"Dream sleep data unavailable: {dream_info}")

        dream_active_np = dream_data['active_local']
        dream_phi_np = dream_data['phi']
        dream_mask_np = dream_data['active_mask']
        dream_candidate_np = dream_data['candidate_local']
        dream_teacher_np = dream_data['teacher_logits']
        dream_active_mx = mx.array(dream_active_np)
        dream_phi_mx = mx.array(dream_phi_np)
        dream_mask_mx = mx.array(dream_mask_np)
        dream_candidate_mx = mx.array(dream_candidate_np)
        dream_teacher_mx = mx.array(dream_teacher_np)
        init_src_mx = mx.array(E_src_np)
        init_tgt_mx = mx.array(E_tgt_np)

        def _dream_logits_numpy(E_src_diag, E_tgt_diag):
            phi_norm = np.sqrt(np.sum(dream_phi_np * dream_phi_np, axis=2, keepdims=True)) + 1e-8
            phi_hat = dream_phi_np / phi_norm
            active_weight = np.expand_dims(dream_mask_np, axis=(2, 3))
            src_emb = E_src_diag[dream_active_np]
            ctx = np.sum(src_emb * np.expand_dims(phi_hat, axis=3) * active_weight, axis=1)
            tgt_emb = E_tgt_diag[dream_candidate_np]
            logits_raw = np.sum(tgt_emb * np.expand_dims(ctx, axis=1), axis=(2, 3))
            active_count = np.sum(dream_mask_np, axis=1, keepdims=True) + 1e-8
            return logits_raw / active_count

        def _print_dream_trace(label, logits_np):
            print(label)
            n_ctx = min(trace_max_contexts, int(logits_np.shape[0]))
            for idx in range(n_ctx):
                count = int(np.sum(dream_mask_np[idx]))
                active_local = dream_active_np[idx, :count].astype(np.int32, copy=False)
                candidate_local = dream_candidate_np[idx].astype(np.int32, copy=False)
                candidate_slots = [int(all_slots[int(local)]) for local in candidate_local.tolist()]
                active_slots_trace = [int(all_slots[int(local)]) for local in active_local.tolist()]
                print(f"    ctx[{idx}] active_slots={active_slots_trace}")
                print(f"      phi={np.array2string(dream_phi_np[idx, :count], precision=4, suppress_small=False)}")
                print(f"      candidate_slots={candidate_slots}")
                print(f"      teacher={np.array2string(dream_teacher_np[idx], precision=4, suppress_small=False)}")
                print(f"      student={np.array2string(logits_np[idx], precision=4, suppress_small=False)}")

        def _dream_diag_numpy(E_src_diag, E_tgt_diag):
            logits = _dream_logits_numpy(E_src_diag, E_tgt_diag)

            teacher_top1 = np.argmax(dream_teacher_np, axis=1)
            student_top1 = np.argmax(logits, axis=1)
            top1_match = float(np.mean(teacher_top1 == student_top1))

            topn = min(8, logits.shape[1])
            teacher_topn = np.argsort(dream_teacher_np, axis=1)[:, -topn:]
            student_topn = np.argsort(logits, axis=1)[:, -topn:]
            overlap = []
            ranks = []
            cosine = []
            rmse = []
            for idx in range(logits.shape[0]):
                overlap.append(len(np.intersect1d(teacher_topn[idx], student_topn[idx], assume_unique=False)) / max(1, topn))
                ranks.append(1 + int(np.sum(logits[idx] > logits[idx, teacher_top1[idx]])))
                denom = (np.linalg.norm(logits[idx]) * np.linalg.norm(dream_teacher_np[idx])) + 1e-8
                cosine.append(float(np.dot(logits[idx], dream_teacher_np[idx]) / denom))
                rmse.append(float(np.sqrt(np.mean((logits[idx] - dream_teacher_np[idx]) ** 2))))

            return {
                'top1_match': top1_match,
                'top8_overlap': float(np.mean(overlap)),
                'teacher_rank_mean': float(np.mean(ranks)),
                'logit_cosine': float(np.mean(cosine)),
                'logit_rmse': float(np.mean(rmse)),
                'student_peak_mean': float(np.mean(np.max(logits, axis=1))),
                'teacher_peak_mean': float(np.mean(np.max(dream_teacher_np, axis=1))),
                'student_norm_mean': float(np.mean(np.linalg.norm(logits, axis=1))),
                'teacher_norm_mean': float(np.mean(np.linalg.norm(dream_teacher_np, axis=1))),
            }

        dream_diag_before = _dream_diag_numpy(E_src_np, E_tgt_np)
        print(
            f"  Dream diag before: top1_match={dream_diag_before['top1_match']:.3f}, "
            f"overlap8={dream_diag_before['top8_overlap']:.3f}, teacher_rank={dream_diag_before['teacher_rank_mean']:.1f}, "
            f"cos={dream_diag_before['logit_cosine']:.3f}, rmse={dream_diag_before['logit_rmse']:.3f}, "
            f"peak_s={dream_diag_before['student_peak_mean']:.3f}, peak_t={dream_diag_before['teacher_peak_mean']:.3f}, "
            f"norm_s={dream_diag_before['student_norm_mean']:.3f}, norm_t={dream_diag_before['teacher_norm_mean']:.3f}"
        )
        if trace_full_io:
            _print_dream_trace("  [TRACE] Dream optimizer inputs / student before", _dream_logits_numpy(E_src_np, E_tgt_np))

        total_dreams = int(dream_data['active_local'].shape[0])
        n_batches = _ceil_div(total_dreams, dream_batch_size)
        lr_schedule = optim.cosine_decay(init=cfg.SLEEP_LR, decay_steps=max(1, dream_epochs * n_batches))
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=lambda_wd)

        best_loss = float('inf')
        patience_counter = 0
        loss_history = []
        best_src_np = None
        best_tgt_np = None

        def _log_softmax(logits):
            shifted = logits - mx.max(logits, axis=1, keepdims=True)
            return shifted - mx.log(mx.sum(mx.exp(shifted), axis=1, keepdims=True) + 1e-8)

        def _dream_element_loss(diff):
            if use_huber:
                abs_diff = mx.abs(diff)
                quadratic = mx.minimum(abs_diff, huber_delta)
                linear = abs_diff - quadratic
                return mx.mean(0.5 * quadratic ** 2 + huber_delta * linear)
            return mx.mean(diff ** 2)

        def _dream_loss(params, batch_active, batch_phi, batch_mask, batch_candidate, batch_teacher):
            phi_norm = mx.sqrt(mx.sum(batch_phi * batch_phi, axis=2, keepdims=True)) + 1e-8
            phi_hat = batch_phi / phi_norm
            active_weight = mx.expand_dims(mx.expand_dims(batch_mask, axis=2), axis=3)
            src_emb = params['E_src'][batch_active]
            ctx = mx.sum(src_emb * mx.expand_dims(phi_hat, axis=3) * active_weight, axis=1)
            tgt_emb = params['E_tgt'][batch_candidate]
            logits_raw = mx.sum(tgt_emb * mx.expand_dims(ctx, axis=1), axis=(2, 3))
            active_count = mx.sum(batch_mask, axis=1, keepdims=True) + 1e-8
            logits = logits_raw / active_count

            teacher_logp = _log_softmax(batch_teacher / dream_temp)
            student_logp = _log_softmax(logits / dream_temp)
            teacher_prob = mx.exp(teacher_logp)
            kl = mx.mean(mx.sum(teacher_prob * (teacher_logp - student_logp), axis=1)) * (dream_temp * dream_temp)
            logit_loss = _dream_element_loss(logits - batch_teacher)
            prox = dream_prox * (
                mx.mean((params['E_src'] - init_src_mx) ** 2) +
                mx.mean((params['E_tgt'] - init_tgt_mx) ** 2)
            )
            total = (dream_logit_weight * logit_loss) + (dream_kl_weight * kl) + prox
            return total

        loss_and_grad_fn = mx.value_and_grad(_dream_loss)

        for epoch in range(dream_epochs):
            check_epoch = trace_full_io or (epoch % 10 == 0) or (epoch == dream_epochs - 1)
            epoch_loss_acc = 0.0
            perm = np.random.permutation(total_dreams).astype(np.int32, copy=False)

            for batch_idx in range(n_batches):
                start = batch_idx * dream_batch_size
                end = min(start + dream_batch_size, total_dreams)
                batch_idx_np = perm[start:end]
                batch_idx_mx = mx.array(batch_idx_np)
                batch_active = dream_active_mx[batch_idx_mx]
                batch_phi = dream_phi_mx[batch_idx_mx]
                batch_mask = dream_mask_mx[batch_idx_mx]
                batch_candidate = dream_candidate_mx[batch_idx_mx]
                batch_teacher = dream_teacher_mx[batch_idx_mx]

                loss, grads = loss_and_grad_fn(params, batch_active, batch_phi, batch_mask, batch_candidate, batch_teacher)
                optimizer.update(params, grads)

                if check_epoch:
                    mx.eval(loss, params)
                    epoch_loss_acc += float(loss)
                else:
                    mx.eval(params)

            if check_epoch:
                lv = epoch_loss_acc / max(1, n_batches)
                loss_history.append(lv)
                if lv < best_loss:
                    best_loss = lv
                    patience_counter = 0
                    mx.eval(params)
                    best_src_np = np.array(params['E_src'])
                    best_tgt_np = np.array(params['E_tgt'])
                else:
                    patience_counter += 1

                if epoch >= cfg.SLEEP_MIN_EPOCHS:
                    if patience_counter >= cfg.SLEEP_PATIENCE:
                        break
                    if len(loss_history) >= 10:
                        recent = loss_history[-10:]
                        rel = abs(recent[0] - recent[-1]) / (abs(recent[0]) + 1e-8)
                        if rel < cfg.SLEEP_REL_TOL:
                            break
                if trace_full_io:
                    mx.eval(params)
                    curr_src_np = np.array(params['E_src'])
                    curr_tgt_np = np.array(params['E_tgt'])
                    curr_logits_np = _dream_logits_numpy(curr_src_np, curr_tgt_np)
                    curr_src_norms = np.linalg.norm(curr_src_np.reshape(n_slots, -1), axis=1)
                    curr_tgt_norms = np.linalg.norm(curr_tgt_np.reshape(n_slots, -1), axis=1)
                    print(
                        f"  [TRACE] Dream epoch {epoch + 1}: loss={lv:.6f}, "
                        f"E_src_mean={np.mean(curr_src_norms):.4f}, E_tgt_mean={np.mean(curr_tgt_norms):.4f}"
                    )
                    _print_dream_trace("    [TRACE] Student logits", curr_logits_np)

        if best_src_np is None or best_tgt_np is None:
            mx.eval(params)
            E_src_np = np.array(params['E_src'])
            E_tgt_np = np.array(params['E_tgt'])
        else:
            E_src_np = best_src_np
            E_tgt_np = best_tgt_np
        dream_diag_after = _dream_diag_numpy(E_src_np, E_tgt_np)
        print(
            f"  Dream diag after: top1_match={dream_diag_after['top1_match']:.3f}, "
            f"overlap8={dream_diag_after['top8_overlap']:.3f}, teacher_rank={dream_diag_after['teacher_rank_mean']:.1f}, "
            f"cos={dream_diag_after['logit_cosine']:.3f}, rmse={dream_diag_after['logit_rmse']:.3f}, "
            f"peak_s={dream_diag_after['student_peak_mean']:.3f}, peak_t={dream_diag_after['teacher_peak_mean']:.3f}, "
            f"norm_s={dream_diag_after['student_norm_mean']:.3f}, norm_t={dream_diag_after['teacher_norm_mean']:.3f}"
        )
        if trace_full_io:
            _print_dream_trace("  [TRACE] Dream optimizer outputs / student after", _dream_logits_numpy(E_src_np, E_tgt_np))
        epoch = epoch
    elif sleep_solver == 'adamw':
        def _element_loss(diff):
            """逐元素损失: MSE 或 Huber"""
            if use_huber:
                abs_diff = mx.abs(diff)
                quadratic = mx.minimum(abs_diff, huber_delta)
                linear = abs_diff - quadratic
                return mx.mean(0.5 * quadratic ** 2 + huber_delta * linear)
            else:
                return mx.mean(diff ** 2)

        if use_huber:
            print(f"  Sleep loss: Huber (delta={huber_delta})")

        estimated_sample_bytes = D * d * 8
        active_memory_bytes = _get_mlx_active_memory_bytes()
        memory_limit_bytes, memory_info = _resolve_mlx_memory_budget_bytes(cfg)
        utilization = min(max(float(getattr(cfg, 'SLEEP_MINIBATCH_UTILIZATION', 0.9)), 0.0), 1.0)
        batch_plan = _plan_sleep_batches(
            n_new,
            n_rehearsal,
            estimated_sample_bytes,
            active_memory_bytes,
            memory_limit_bytes,
            utilization,
        )

        use_minibatch = batch_plan['use_minibatch']
        new_batch_size = batch_plan['new_batch_size']
        rehearsal_batch_size = batch_plan['rehearsal_batch_size']
        n_batches = batch_plan['n_batches']

        mode_label = 'mini-batch' if use_minibatch else 'full-batch'
        active_gb = batch_plan['active_memory_bytes'] / (1000 ** 3)
        batch_gb = batch_plan['batch_working_set_bytes'] / (1000 ** 3)
        full_gb = batch_plan['full_working_set_bytes'] / (1000 ** 3)
        limit_gb = batch_plan['memory_limit_bytes'] / (1000 ** 3) if batch_plan['memory_limit_bytes'] > 0 else 0.0
        target_gb = batch_plan['target_working_set_bytes'] / (1000 ** 3) if batch_plan['target_working_set_bytes'] > 0 else 0.0
        device_name = memory_info.get('device_name', 'unknown') if isinstance(memory_info, dict) else 'unknown'
        print(
            f"  Sleep optimize: {mode_label}, new_batch={new_batch_size}, reh_batch={rehearsal_batch_size or 0}, "
            f"steps/epoch={n_batches}, active={active_gb:.2f}GB, batch={batch_gb:.2f}GB, "
            f"full={full_gb:.2f}GB, target={target_gb:.2f}GB, limit={limit_gb:.2f}GB, "
            f"device={device_name}, reason={batch_plan['reason']}"
        )

        total_steps = max(1, cfg.SLEEP_MAX_EPOCHS * n_batches)
        lr_schedule = optim.cosine_decay(init=cfg.SLEEP_LR, decay_steps=total_steps)
        optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=lambda_wd)

        best_loss = float('inf')
        patience_counter = 0
        loss_history = []

        def _batch_loss(params, batch_src, batch_tgt, batch_target):
            pred = mx.sum(params['E_tgt'][batch_tgt] * params['E_src'][batch_src], axis=2)
            return _element_loss(pred - batch_target)

        def _combined_loss(params, batch_new_src, batch_new_tgt, batch_new_target,
                           batch_reh_src=None, batch_reh_tgt=None, batch_reh_target=None):
            loss = _batch_loss(params, batch_new_src, batch_new_tgt, batch_new_target)
            if batch_reh_src is not None:
                loss = loss + _batch_loss(params, batch_reh_src, batch_reh_tgt, batch_reh_target)
            return loss

        loss_and_grad_fn = mx.value_and_grad(_combined_loss)

        def _batch_indices(perm_np, start, batch_size):
            if len(perm_np) == 0:
                return None
            start_mod = start % len(perm_np)
            end_mod = start_mod + batch_size
            if end_mod <= len(perm_np):
                return perm_np[start_mod:end_mod]
            return np.concatenate([perm_np[start_mod:], perm_np[:end_mod - len(perm_np)]])

        for epoch in range(cfg.SLEEP_MAX_EPOCHS):
            check_epoch = (epoch % 10 == 0) or (epoch == cfg.SLEEP_MAX_EPOCHS - 1)
            epoch_loss_acc = 0.0

            if use_minibatch:
                perm_new = np.random.permutation(n_new).astype(np.int32, copy=False)
                perm_reh = None if n_rehearsal == 0 else np.random.permutation(n_rehearsal).astype(np.int32, copy=False)

                for batch_idx in range(n_batches):
                    new_idx_np = _batch_indices(perm_new, batch_idx * new_batch_size, new_batch_size)
                    new_idx_mx = mx.array(new_idx_np)
                    batch_new_src = src_indices_mx[new_idx_mx]
                    batch_new_tgt = tgt_indices_mx[new_idx_mx]
                    batch_new_target = target_mx[new_idx_mx]

                    if n_rehearsal > 0:
                        reh_idx_np = _batch_indices(perm_reh, batch_idx * rehearsal_batch_size, rehearsal_batch_size)
                        reh_idx_mx = mx.array(reh_idx_np)
                        batch_reh_src = rehearsal_src_mx[reh_idx_mx]
                        batch_reh_tgt = rehearsal_tgt_mx[reh_idx_mx]
                        batch_reh_target = rehearsal_targets_mx[reh_idx_mx]
                    else:
                        batch_reh_src = batch_reh_tgt = batch_reh_target = None

                    loss, grads = loss_and_grad_fn(
                        params,
                        batch_new_src,
                        batch_new_tgt,
                        batch_new_target,
                        batch_reh_src,
                        batch_reh_tgt,
                        batch_reh_target,
                    )
                    optimizer.update(params, grads)

                    if check_epoch:
                        mx.eval(loss, params)
                        epoch_loss_acc += float(loss)
                    else:
                        mx.eval(params)
            else:
                loss, grads = loss_and_grad_fn(
                    params,
                    src_indices_mx,
                    tgt_indices_mx,
                    target_mx,
                    rehearsal_src_mx,
                    rehearsal_tgt_mx,
                    rehearsal_targets_mx,
                )
                optimizer.update(params, grads)
                if check_epoch:
                    mx.eval(loss, params)
                    epoch_loss_acc = float(loss)
                else:
                    mx.eval(params)

            if check_epoch:
                lv = epoch_loss_acc / n_batches if use_minibatch else epoch_loss_acc
                loss_history.append(lv)

                if lv < best_loss:
                    best_loss = lv
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch >= cfg.SLEEP_MIN_EPOCHS:
                    if patience_counter >= cfg.SLEEP_PATIENCE:
                        break
                    if len(loss_history) >= 10:
                        recent = loss_history[-10:]
                        rel = abs(recent[0] - recent[-1]) / (abs(recent[0]) + 1e-8)
                        if rel < cfg.SLEEP_REL_TOL:
                            break

        mx.eval(params)
        E_src_np = np.array(params['E_src'])  # (n_slots, D, d)
        E_tgt_np = np.array(params['E_tgt'])  # (n_slots, D, d)
    else:
        raise ValueError(f"Unknown sleep solver: {sleep_solver}")

    # --- 诊断: 训练后 E_src/E_tgt 范数 ---
    E_src_final_norms = np.linalg.norm(E_src_np.reshape(n_slots, -1), axis=1)
    E_tgt_final_norms = np.linalg.norm(E_tgt_np.reshape(n_slots, -1), axis=1)
    print(f"  [诊断] E_src最终: norm mean={np.mean(E_src_final_norms):.4f}, "
          f"max={np.max(E_src_final_norms):.4f}, "
          f"变化={np.mean(E_src_final_norms)/np.mean(E_src_init_norms):.2f}x")
    print(f"  [诊断] E_tgt最终: norm mean={np.mean(E_tgt_final_norms):.4f}, "
          f"max={np.max(E_tgt_final_norms):.4f}, "
          f"变化={np.mean(E_tgt_final_norms)/np.mean(E_tgt_init_norms):.2f}x")

    # --- 诊断: 训练后 A 预测范数 ---
    a_new_pred = np.sum(E_tgt_np[tgt_indices] * E_src_np[src_indices], axis=2)  # (n_new, D)
    a_new_norms = np.linalg.norm(a_new_pred, axis=1)
    residual = a_new_pred - target_np
    residual_norms = np.linalg.norm(residual, axis=1)
    print(f"  [诊断] A_new预测: norm mean={np.mean(a_new_norms):.4f}, "
          f"max={np.max(a_new_norms):.4f}")
    print(f"  [诊断] 残差: norm mean={np.mean(residual_norms):.4f}, "
          f"max={np.max(residual_norms):.4f}")

    E_src_dict = {slot: E_src_np[idx].T.astype(np.float32) for slot, idx in slot_to_idx.items()}  # 转置为 (d, D)
    E_tgt_dict = {slot: E_tgt_np[idx].T.astype(np.float32) for slot, idx in slot_to_idx.items()}  # 转置为 (d, D)

    updated_slots = involved_slots if cfg.GAMMA_ADAPTIVE else None
    a_knowledge.update_from_sleep(E_src_dict, E_tgt_dict, updated_slots=updated_slots)

    # 对 B 进行衰减（或清空）
    if hasattr(b_memory, 'decay'):
        b_memory.decay(b_retention)

    result.n_new = n_new
    result.n_rehearsal = n_rehearsal
    result.epochs = epoch + 1
    result.final_loss = best_loss
    result.success = True

    print(f"  Sleep MLX 完成: {epoch+1} epochs, loss={best_loss:.6f}")

    return result
