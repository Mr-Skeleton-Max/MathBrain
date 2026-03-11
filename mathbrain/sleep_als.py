"""Sleep ALS helper: direct low-rank fitting without SGD."""

from __future__ import annotations

import numpy as np


def _build_edge_groups(slot_indices: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for edge_idx, slot_idx in enumerate(slot_indices.tolist()):
        groups.setdefault(int(slot_idx), []).append(edge_idx)
    return {slot_idx: np.asarray(edge_ids, dtype=np.int32) for slot_idx, edge_ids in groups.items()}


def _solve_slot_rows(counterparty: np.ndarray,
                     targets: np.ndarray,
                     init_rows: np.ndarray,
                     ridge: float,
                     prox: float) -> np.ndarray:
    """Solve one slot's D rows via independent ridge systems.

    Args:
        counterparty: (m, D, d)
        targets: (m, D)
        init_rows: (D, d)
    Returns:
        Updated rows: (D, d)
    """
    _, n_dims, rank = counterparty.shape
    out = np.empty((n_dims, rank), dtype=np.float32)
    base_diag = float(ridge + prox) + 1e-6

    for dim_idx in range(n_dims):
        vectors = np.nan_to_num(counterparty[:, dim_idx, :], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        target = np.nan_to_num(targets[:, dim_idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        with np.errstate(all='ignore'):
            gram = vectors.T @ vectors
            rhs = vectors.T @ target
        gram.flat[::rank + 1] += base_diag
        if prox > 0.0:
            rhs = rhs + float(prox) * init_rows[dim_idx].astype(np.float64, copy=False)
        out[dim_idx] = np.linalg.solve(gram, rhs).astype(np.float32)

    return out


def _compute_mse(E_src: np.ndarray,
                 E_tgt: np.ndarray,
                 src_indices: np.ndarray,
                 tgt_indices: np.ndarray,
                 targets: np.ndarray) -> float:
    pred = np.sum(E_tgt[tgt_indices] * E_src[src_indices], axis=2)
    diff = pred - targets
    return float(np.mean(diff * diff))


def solve_sleep_als(E_src_init: np.ndarray,
                    E_tgt_init: np.ndarray,
                    src_indices: np.ndarray,
                    tgt_indices: np.ndarray,
                    targets: np.ndarray,
                    max_iters: int = 6,
                    ridge: float = 1e-3,
                    prox: float = 1.0,
                    rel_tol: float = 1e-4,
                    verbose: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit sleep targets with alternating least squares.

    Shapes:
      E_src/E_tgt: (n_slots, D, d)
      src_indices/tgt_indices: (n_edges,)
      targets: (n_edges, D)
    """
    if max_iters <= 0:
        raise ValueError(f'max_iters must be positive, got {max_iters}')

    E_src = np.array(E_src_init, copy=True)
    E_tgt = np.array(E_tgt_init, copy=True)
    src_groups = _build_edge_groups(src_indices)
    tgt_groups = _build_edge_groups(tgt_indices)

    history: list[float] = []
    best_loss = float('inf')
    best_src = E_src.copy()
    best_tgt = E_tgt.copy()

    for iteration in range(max_iters):
        for src_slot, edge_ids in src_groups.items():
            counterparty = E_tgt[tgt_indices[edge_ids]]
            target_block = targets[edge_ids]
            E_src[src_slot] = _solve_slot_rows(counterparty, target_block, E_src[src_slot], ridge, prox)

        for tgt_slot, edge_ids in tgt_groups.items():
            counterparty = E_src[src_indices[edge_ids]]
            target_block = targets[edge_ids]
            E_tgt[tgt_slot] = _solve_slot_rows(counterparty, target_block, E_tgt[tgt_slot], ridge, prox)

        loss = _compute_mse(E_src, E_tgt, src_indices, tgt_indices, targets)
        history.append(loss)
        if verbose:
            print(f'  Sleep ALS iter {iteration + 1}/{max_iters}: loss={loss:.6f}')
        if loss < best_loss:
            best_loss = loss
            best_src = E_src.copy()
            best_tgt = E_tgt.copy()
        if len(history) >= 2:
            prev = history[-2]
            rel = abs(prev - loss) / (abs(prev) + 1e-8)
            if rel < rel_tol:
                break

    return best_src, best_tgt, {
        'iterations': len(history),
        'final_loss': best_loss,
        'loss_history': history,
    }
