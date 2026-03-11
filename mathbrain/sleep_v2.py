"""Sleep 巩固 v2: Slice-wise 矩阵分解 + Mini-batch 优化

核心改进：
1. E_src/E_tgt 形状为 (d, D)，d 和 D 解耦
2. 用 AdamW weight_decay 替代正交约束
3. Mini-batch: 稀疏抽取相关 slots，避免显存爆炸
4. 移除随机采样，依赖 B 裁剪控制样本数

数学公式：
  对于每个 B[src,tgt] 条目 (D 维向量):
    A[tgt, src] = Σ_r E_tgt[tgt,r,:] * E_src[src,r,:]  → (D,)
    目标：A_new ≈ γ·A_old + B
"""

from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.optim as optim
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False

from .config import MathBrainConfig
from .sleep_als import solve_sleep_als


@dataclass
class SleepResultV2:
    n_new: int = 0
    n_rehearsal: int = 0
    epochs: int = 0
    final_loss: float = 0.0
    success: bool = False


def _resolve_sleep_global_decay(raw_decay: float) -> tuple[float, bool]:
    decay = float(raw_decay)
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"SLEEP_GLOBAL_DECAY must be in [0, 1], got {decay}")
    if decay == 0.0:
        return 1.0, True
    return decay, False


def consolidate_v2(b_memory, a_knowledge, active_slots: set,
                   config: MathBrainConfig = None, use_mps: bool = True,
                   b_retention: float = 0.0,
                   cp_rank: int = 384,
                   lambda_wd: float = 1e-4,
                   retina=None,
                   vocab_words: list[str] | None = None,
                   dream_corpus: list[str] | None = None) -> SleepResultV2:
    """Sleep 巩固 v2: slice-wise 矩阵分解 + mini-batch

    Args:
        b_memory: B Memory 实例
        a_knowledge: AKnowledgeV2 实例
        active_slots: 活跃槽位集合
        config: 配置
        use_mps: 是否使用 MPS 加速
        b_retention: B Memory 保留比例 (0-1)
        cp_rank: E_src/E_tgt 的 rank 维度 d
        lambda_wd: AdamW weight decay (替代正交约束)
    """
    del retina, vocab_words, dream_corpus

    cfg = config or MathBrainConfig()
    result = SleepResultV2()

    if not HAS_TORCH:
        print("  Sleep v2: PyTorch 不可用")
        return result

    # 收集 B 样本（不做随机采样，依赖 prune_by_norm 控制）
    entries = b_memory.get_entries()
    if not entries:
        print("  Sleep v2: 无新知识样本")
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

    n_new = len(entries)
    all_slots = sorted(involved_slots)
    slot_to_idx = {s: i for i, s in enumerate(all_slots)}
    n_slots = len(all_slots)
    d = cp_rank
    D = a_knowledge.D    # phi 维度

    print(f"  Sleep v2: {n_new} 样本, {n_slots} 槽位, d={d}, D={D}, batch_size={cfg.SLEEP_BATCH_SIZE}")
    print(
        f"  [诊断] Sleep scope: involved={len(involved_slots)}, active_arg={len(active_slot_set)}, "
        f"overlap={len(overlap_slots)}, active_only={len(active_only_slots)}, "
        f"involved_only={len(involved_only_slots)}, legacy_union={len(legacy_scope_slots)}, "
        f"current_scope={n_slots}"
    )

    # 设备
    device = torch.device("mps") if (use_mps and HAS_MPS) else torch.device("cpu")

    # 初始化参数: E_src (n_slots, d, D), E_tgt (n_slots, d, D)
    if not a_knowledge.has_knowledge:
        # 首次 Sleep: 随机初始化 * 0.01
        E_src_np = np.random.randn(n_slots, d, D).astype(np.float32) * 0.01
        E_tgt_np = np.random.randn(n_slots, d, D).astype(np.float32) * 0.01
    else:
        # 后续 Sleep: 从已有知识加载
        E_src_np = np.zeros((n_slots, d, D), dtype=np.float32)
        E_tgt_np = np.zeros((n_slots, d, D), dtype=np.float32)
        for slot, idx in slot_to_idx.items():
            if slot in a_knowledge.E_src:
                E_src_np[idx] = a_knowledge.E_src[slot]
                E_tgt_np[idx] = a_knowledge.E_tgt[slot]
            else:
                E_src_np[idx] = np.random.randn(d, D).astype(np.float32) * 0.01
                E_tgt_np[idx] = np.random.randn(d, D).astype(np.float32) * 0.01

    raw_global_decay = float(getattr(cfg, 'SLEEP_GLOBAL_DECAY', 1.0))
    global_decay, decay_disabled = _resolve_sleep_global_decay(raw_global_decay)
    if a_knowledge.has_knowledge and global_decay < 1.0:
        factor_decay = float(np.sqrt(global_decay))
        E_src_np *= factor_decay
        E_tgt_np *= factor_decay
        print(f"  Sleep global decay: edge={global_decay:.6f}, factor={factor_decay:.6f}")
    elif decay_disabled:
        print("  Sleep global decay: disabled (raw=0 interpreted as no decay)")

    E_src = torch.tensor(E_src_np, dtype=torch.float32, device=device, requires_grad=True)
    E_tgt = torch.tensor(E_tgt_np, dtype=torch.float32, device=device, requires_grad=True)

    # 准备 B 样本数据 (D 维向量) - 预先转换为 tensor
    src_indices, tgt_indices, b_vecs_np = [], [], []
    for src, tgt, b_vec in entries:
        src_indices.append(slot_to_idx[src])
        tgt_indices.append(slot_to_idx[tgt])
        b_vecs_np.append(b_vec)

    src_indices = np.array(src_indices, dtype=np.int64)
    tgt_indices = np.array(tgt_indices, dtype=np.int64)
    b_vecs_np = np.array(b_vecs_np, dtype=np.float32)

    # 预先转换为 tensor（避免训练循环中重复创建）
    src_indices_t = torch.tensor(src_indices, dtype=torch.long, device=device)
    tgt_indices_t = torch.tensor(tgt_indices, dtype=torch.long, device=device)

    # 预先转换 B 向量为 tensor（在 GPU 上）
    b_vecs_t = torch.tensor(b_vecs_np, dtype=torch.float32, device=device)

    # === Target = γ·A_old + B (预计算，固定，全程 GPU) ===
    with torch.no_grad():
        # 分批计算 A_old 避免显存爆炸
        batch_size = cfg.SLEEP_BATCH_SIZE
        n_batches = (n_new + batch_size - 1) // batch_size
        a_old_list = []

        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_new)
            src_batch = src_indices_t[start:end]  # 复用预创建的 tensor
            tgt_batch = tgt_indices_t[start:end]
            # A_old: Σ_r E_tgt[tgt,r,:] * E_src[src,r,:] → (batch, D)
            a_old_batch = torch.sum(E_tgt[tgt_batch] * E_src[src_batch], dim=1)
            a_old_list.append(a_old_batch)  # 保持在 GPU

        a_old_fixed_t = torch.cat(a_old_list, dim=0)  # (n_new, D) on GPU

        if cfg.GAMMA_ADAPTIVE and a_knowledge.has_knowledge:
            src_slots = np.array([all_slots[idx] for idx in src_indices], dtype=np.int32)
            tgt_slots = np.array([all_slots[idx] for idx in tgt_indices], dtype=np.int32)
            gamma_src = a_knowledge.get_adaptive_gamma_batch(
                src_slots, cfg.GAMMA_INITIAL, cfg.GAMMA_MIN, cfg.GAMMA_DECAY_RATE)
            gamma_tgt = a_knowledge.get_adaptive_gamma_batch(
                tgt_slots, cfg.GAMMA_INITIAL, cfg.GAMMA_MIN, cfg.GAMMA_DECAY_RATE)
            gamma_vec_np = np.maximum(gamma_src, gamma_tgt)[:, None]  # (n_new, 1)
            gamma_vec_t = torch.tensor(gamma_vec_np, dtype=torch.float32, device=device)
            target_new_t = gamma_vec_t * a_old_fixed_t + b_vecs_t
        else:
            gamma = cfg.GAMMA_DECAY
            target_new_t = gamma * a_old_fixed_t + b_vecs_t  # 全程 GPU

    # === 不做 rehearsal：Sleep 仅吸收 B 中记录的增量目标 ===
    n_rehearsal = 0
    rehearsal_src_t = None
    rehearsal_tgt_t = None
    rehearsal_targets_t = None

    sleep_solver = str(getattr(cfg, 'SLEEP_SOLVER', 'adamw')).lower()
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
            src_indices.astype(np.int32, copy=False),
            tgt_indices.astype(np.int32, copy=False),
            target_new_t.detach().cpu().numpy(),
            max_iters=als_iters,
            ridge=als_ridge,
            prox=als_prox,
            rel_tol=cfg.SLEEP_REL_TOL,
            verbose=True,
        )
        best_loss = float(als_info['final_loss'])
        epoch = int(als_info['iterations']) - 1
    elif sleep_solver == 'dream':
        raise ValueError('Dream sleep solver is currently implemented only for the MLX backend')
    elif sleep_solver == 'adamw':
        optimizer = optim.AdamW([E_src, E_tgt], lr=cfg.SLEEP_LR, weight_decay=lambda_wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SLEEP_MAX_EPOCHS)

        best_loss = float('inf')
        patience_counter = 0
        loss_history = []

        if use_huber:
            print(f"  Sleep loss: Huber (delta={huber_delta})")

        for epoch in range(cfg.SLEEP_MAX_EPOCHS):
            optimizer.zero_grad()

            a_new_pred = torch.sum(E_tgt[tgt_indices_t] * E_src[src_indices_t], dim=1)
            if use_huber:
                loss_new = torch.nn.functional.huber_loss(
                    a_new_pred, target_new_t, reduction='mean', delta=huber_delta)
            else:
                loss_new = torch.mean((a_new_pred - target_new_t) ** 2)

            if n_rehearsal > 0:
                reh_pred = torch.sum(E_tgt[rehearsal_tgt_t] * E_src[rehearsal_src_t], dim=1)
                if use_huber:
                    loss_rehearsal = torch.nn.functional.huber_loss(
                        reh_pred, rehearsal_targets_t, reduction='mean', delta=huber_delta)
                else:
                    loss_rehearsal = torch.mean((reh_pred - rehearsal_targets_t) ** 2)
                loss = loss_new + loss_rehearsal
            else:
                loss = loss_new

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 10 == 0 or epoch == cfg.SLEEP_MAX_EPOCHS - 1:
                lv = loss.item()
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

        E_src_np = E_src.detach().cpu().numpy()
        E_tgt_np = E_tgt.detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown sleep solver: {sleep_solver}")

    E_src_dict = {slot: E_src_np[idx].astype(np.float32) for slot, idx in slot_to_idx.items()}
    E_tgt_dict = {slot: E_tgt_np[idx].astype(np.float32) for slot, idx in slot_to_idx.items()}

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

    print(f"  Sleep 完成: {epoch+1} epochs, loss={best_loss:.6f}")

    return result
