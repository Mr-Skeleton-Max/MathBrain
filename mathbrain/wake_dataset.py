"""Wake 数据集：为并行 / 树形 Wake 实验准备全局扁平数据。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class WakeShard:
    position_start: int
    position_end: int
    active_offsets: np.ndarray
    active_slots: np.ndarray
    phi_hat: np.ndarray
    target_offsets: np.ndarray
    target_slots: np.ndarray

    @property
    def n_positions(self) -> int:
        return int(self.position_end - self.position_start)


class WakeDataset:
    """全局扁平 Wake 数据集。"""

    def __init__(
        self,
        active_offsets: np.ndarray,
        active_slots: np.ndarray,
        phi_hat: np.ndarray,
        target_offsets: np.ndarray,
        target_slots: np.ndarray,
        sentences_kept: int,
        total_positions: int,
    ):
        self.active_offsets = active_offsets.astype(np.int64, copy=False)
        self.active_slots = active_slots.astype(np.int32, copy=False)
        self.phi_hat = phi_hat.astype(np.float32, copy=False)
        self.target_offsets = target_offsets.astype(np.int64, copy=False)
        self.target_slots = target_slots.astype(np.int32, copy=False)
        self.sentences_kept = int(sentences_kept)
        self.total_positions = int(total_positions)

    @staticmethod
    def _tokenize(sentence: str) -> List[str]:
        return [
            w.strip()
            for w in sentence.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').split()
            if w.strip()
        ]

    @staticmethod
    def _build_sentence_q(encoded_prefix: List[dict], rho: np.ndarray, theta_q: float, eps_q: float):
        t_steps = len(encoded_prefix)
        if t_steps == 0:
            return [], [], 0

        slot_lists = [np.fromiter(item.keys(), dtype=np.int32) for item in encoded_prefix]
        count_lists = [np.fromiter(item.values(), dtype=np.float32) for item in encoded_prefix]
        nonempty_slots = [slots for slots in slot_lists if len(slots) > 0]
        if not nonempty_slots:
            return [], [], t_steps

        all_slots = np.concatenate(nonempty_slots, axis=0)
        _, first_idx = np.unique(all_slots, return_index=True)
        unique_slots = all_slots[np.sort(first_idx)].astype(np.int32, copy=False)
        n_slots = len(unique_slots)
        n_scales = int(rho.shape[0])
        slot_to_idx = {int(slot): idx for idx, slot in enumerate(unique_slots.tolist())}

        x = np.zeros((t_steps, n_slots), dtype=np.float32)
        for t, (slots, counts) in enumerate(zip(slot_lists, count_lists)):
            if len(slots) == 0:
                continue
            cols = np.fromiter((slot_to_idx[int(slot)] for slot in slots), dtype=np.int32, count=len(slots))
            x[t, cols] = counts

        q_state = np.zeros((n_slots, n_scales), dtype=np.float32)
        active_slots_parts = []
        q_vals_parts = []
        rho_row = rho.astype(np.float32, copy=False)[None, :]

        for t in range(t_steps):
            if n_slots > 0:
                q_state *= rho_row
                alive = np.max(np.abs(q_state), axis=1) >= eps_q
                if not np.all(alive):
                    q_state[~alive] = 0.0

            counts_t = x[t]
            active_write = counts_t != 0
            if np.any(active_write):
                q_state[active_write] += counts_t[active_write, None]

            norms = np.linalg.norm(q_state, axis=1)
            mask_t = norms > theta_q
            if not np.any(mask_t):
                active_slots_parts.append(np.zeros((0,), dtype=np.int32))
                q_vals_parts.append(np.zeros((0, n_scales), dtype=np.float32))
            else:
                active_slots_parts.append(unique_slots[mask_t].astype(np.int32, copy=False))
                q_vals_parts.append(q_state[mask_t].astype(np.float32, copy=True))

        return active_slots_parts, q_vals_parts, t_steps

    @classmethod
    def from_corpus(cls, model, corpus: List[str], verbose: bool = True) -> "WakeDataset":
        retina = model.retina
        phi_encoder = model.phi_encoder
        rho = model.config.rho.astype(np.float32, copy=False)
        theta_q = float(model.config.THETA_Q)
        eps_q = float(model.config.EPS_Q)

        active_offsets = [0]
        target_offsets = [0]
        active_slots_parts = []
        q_vals_parts = []
        target_slots_parts = []

        kept_sentences = 0
        total_positions = 0

        for sentence in corpus:
            words = cls._tokenize(sentence)
            if len(words) < 2:
                continue

            for word in words:
                model._ensure_word(word)

            encoded = [retina.encode(word) for word in words]
            active_by_pos, q_by_pos, t_steps = cls._build_sentence_q(encoded[:-1], rho, theta_q, eps_q)
            if t_steps == 0:
                continue

            kept_sentences += 1
            for pos in range(t_steps):
                active_slots = active_by_pos[pos]
                if len(active_slots) == 0:
                    continue

                tgt_slots = np.fromiter(encoded[pos + 1].keys(), dtype=np.int32)
                if len(tgt_slots) == 0:
                    continue

                active_slots_parts.append(active_slots)
                q_vals_parts.append(q_by_pos[pos])
                target_slots_parts.append(tgt_slots.astype(np.int32, copy=False))
                active_offsets.append(active_offsets[-1] + len(active_slots))
                target_offsets.append(target_offsets[-1] + len(tgt_slots))
                total_positions += 1

        if active_slots_parts:
            active_slots_flat = np.concatenate(active_slots_parts, axis=0)
            q_vals_flat = np.concatenate(q_vals_parts, axis=0)
            if hasattr(phi_encoder, 'encode_normalized_mlx'):
                phi_hat_flat = phi_encoder.encode_normalized_mlx(q_vals_flat)
            elif hasattr(phi_encoder, 'encode_normalized'):
                phi_hat_flat = phi_encoder.encode_normalized(q_vals_flat)
            else:
                phi = phi_encoder.encode(q_vals_flat)
                norms = np.linalg.norm(phi, axis=1, keepdims=True)
                phi_hat_flat = (phi / (norms + 1e-8)).astype(np.float32, copy=False)
            target_slots_flat = np.concatenate(target_slots_parts, axis=0)
        else:
            d = model.config.d
            active_slots_flat = np.zeros((0,), dtype=np.int32)
            phi_hat_flat = np.zeros((0, d), dtype=np.float32)
            target_slots_flat = np.zeros((0,), dtype=np.int32)

        if verbose:
            print(
                f"WakeDataset: {kept_sentences} 句, {total_positions} positions, "
                f"{len(active_slots_flat)} active rows"
            )

        return cls(
            active_offsets=np.asarray(active_offsets, dtype=np.int64),
            active_slots=active_slots_flat,
            phi_hat=phi_hat_flat,
            target_offsets=np.asarray(target_offsets, dtype=np.int64),
            target_slots=target_slots_flat,
            sentences_kept=kept_sentences,
            total_positions=total_positions,
        )

    def iter_shards(self, shard_size: int) -> Iterable[WakeShard]:
        if shard_size <= 0:
            shard_size = self.total_positions

        for start in range(0, self.total_positions, shard_size):
            end = min(start + shard_size, self.total_positions)

            active_lo = int(self.active_offsets[start])
            active_hi = int(self.active_offsets[end])
            target_lo = int(self.target_offsets[start])
            target_hi = int(self.target_offsets[end])

            yield WakeShard(
                position_start=start,
                position_end=end,
                active_offsets=self.active_offsets[start:end + 1] - active_lo,
                active_slots=self.active_slots[active_lo:active_hi],
                phi_hat=self.phi_hat[active_lo:active_hi],
                target_offsets=self.target_offsets[start:end + 1] - target_lo,
                target_slots=self.target_slots[target_lo:target_hi],
            )
