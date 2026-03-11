"""MLX 评估器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import time

try:
    import mlx.core as mx
    HAS_MLX = True
except Exception:
    HAS_MLX = False
    mx = None


@dataclass
class EvalResult:
    accuracy: float
    correct: int
    total: int


class DenseEvaluatorMLX:
    def __init__(self, model, slot_universe: np.ndarray, trainer=None):
        if not HAS_MLX:
            raise RuntimeError("MLX is not available")
        self.model = model
        self.b = model.b
        self.a = model.a
        self.K = model.config.K
        self.D = model.config.d
        self.slot_universe = slot_universe.astype(np.int32, copy=False)
        self.slot_to_compact = {int(slot): idx for idx, slot in enumerate(self.slot_universe.tolist())}
        self.b_dense = None
        self.e_v = None
        self.e_q = None
        self.e_tgt_flat = None
        self._a_mode = None
        self.word_slot_compact = None
        self.word_slot_compact_t = None
        self.gold_word_by_target_tuple = None
        self.a_word = None
        self.b_word = None
        self.decoder = None
        self.trainer = trainer

    def _ensure_decoder(self):
        from .inference_fast import SparseMatrixDecoder
        if self.model._decoder_dirty or self.model._decoder is None:
            self.model._decoder = SparseMatrixDecoder(self.model.vocab, self.model.word_to_slots, self.K)
            self.model._decoder_dirty = False
        self.decoder = self.model._decoder

    def _prepare_dense_b(self):
        s = len(self.slot_universe)
        b_dense_np = np.zeros((s, s, self.D), dtype=np.float32)
        n = int(self.b._n_entries)
        if n > 0:
            for idx in range(n):
                src = int(self.b._srcs[idx])
                tgt = int(self.b._tgts[idx])
                src_compact = self.slot_to_compact.get(src)
                tgt_compact = self.slot_to_compact.get(tgt)
                if src_compact is not None and tgt_compact is not None:
                    b_dense_np[src_compact, tgt_compact] = self.b._vecs[idx]
        self.b_dense = mx.array(b_dense_np)

    def _prepare_dense_a(self):
        s = len(self.slot_universe)
        if not self.a.has_knowledge:
            self.e_v = None
            self.e_q = None
            self.e_tgt_flat = None
            self._a_mode = None
            return
        self.b._precompute_a_embeddings(self.a)
        cache = self.b._a_cache
        if cache is None:
            self.e_v = None
            self.e_q = None
            self.e_tgt_flat = None
            self._a_mode = None
            return
        if 'E_V' in cache and 'E_Q' in cache:
            e_v_np = np.zeros((s, self.D), dtype=np.float32)
            e_q_np = np.zeros((s, self.D), dtype=np.float32)
            for compact, slot in enumerate(self.slot_universe):
                idx = cache['slot_to_idx'][slot] if slot <= cache['max_slot'] else -1
                if idx >= 0:
                    e_v_np[compact] = cache['E_V'][idx]
                    e_q_np[compact] = cache['E_Q'][idx]
            self.e_v = mx.array(e_v_np)
            self.e_q = mx.array(e_q_np)
            self.e_tgt_flat = None
            self._a_mode = 'legacy'
            return
        if 'E_src' in cache and 'E_tgt' in cache:
            d_rank = int(cache.get('d', cache['E_src'].shape[1]))
            e_v_np = np.zeros((s, d_rank, self.D), dtype=np.float32)
            e_q_np = np.zeros((s, d_rank, self.D), dtype=np.float32)
            for compact, slot in enumerate(self.slot_universe):
                idx = cache['slot_to_idx'][slot] if slot <= cache['max_slot'] else -1
                if idx >= 0:
                    e_v_np[compact] = cache['E_src'][idx]
                    e_q_np[compact] = cache['E_tgt'][idx]
            self.e_v = mx.array(e_v_np)
            self.e_q = mx.array(e_q_np)
            self.e_tgt_flat = mx.reshape(self.e_q, (s, d_rank * self.D))
            self._a_mode = 'v2'
            return
        self.e_v = None
        self.e_q = None
        self.e_tgt_flat = None
        self._a_mode = None

    def _prepare_word_projection(self):
        self._ensure_decoder()
        vocab_list = self.decoder.word_list
        v = len(vocab_list)
        s = len(self.slot_universe)
        proj = np.zeros((v, s), dtype=np.float32)
        gold = {}
        for word_idx, word in enumerate(vocab_list):
            slots = self.model.word_to_slots[word]
            compact_slots = sorted(self.slot_to_compact[int(slot)] for slot in slots if int(slot) in self.slot_to_compact)
            if compact_slots:
                weight = 1.0 / len(compact_slots)
                proj[word_idx, compact_slots] = weight
                gold[tuple(compact_slots)] = word_idx
        self.word_slot_compact = mx.array(proj)
        self.word_slot_compact_t = mx.transpose(self.word_slot_compact)
        self.gold_word_by_target_tuple = gold

    def _setup(self):
        if self.trainer is not None and getattr(self.trainer, 'b_dense', None) is not None:
            self.b_dense = self.trainer.b_dense
            if getattr(self.trainer, 'e_v', None) is None and self.a.has_knowledge:
                self.trainer._prepare_a_dense()
            self.e_v = self.trainer.e_v
            self.e_q = self.trainer.e_q
            self._a_mode = getattr(self.trainer, '_a_mode', None)
            if self._a_mode == 'v2' and self.e_q is not None:
                shape = self.e_q.shape
                self.e_tgt_flat = mx.reshape(self.e_q, (int(shape[0]), int(shape[1]) * int(shape[2])))
            else:
                self.e_tgt_flat = None
        else:
            self._prepare_dense_b()
            self._prepare_dense_a()
        self._prepare_word_projection()
        b_src_d_t = mx.transpose(self.b_dense, (0, 2, 1))
        self.b_word = mx.matmul(b_src_d_t, self.word_slot_compact_t)
        if self.e_q is not None and self._a_mode == 'legacy':
            self.a_word = mx.matmul(mx.transpose(self.e_q), self.word_slot_compact_t)
        else:
            self.a_word = None

    def _gold_word_idx(self, dataset) -> np.ndarray:
        target_offsets = dataset.target_offsets_np
        target_compact = dataset.target_compact_np
        gold = np.full((dataset.total_positions,), -1, dtype=np.int32)
        for pos in range(dataset.total_positions):
            t0, t1 = int(target_offsets[pos]), int(target_offsets[pos + 1])
            gold[pos] = self.gold_word_by_target_tuple.get(tuple(sorted(target_compact[t0:t1].tolist())), -1)
        return gold

    def evaluate_dataset(self, dataset) -> EvalResult:
        t0 = time.time()
        self._setup()
        t1 = time.time()
        p = int(dataset.total_positions)
        v = int(self.word_slot_compact.shape[0])
        active_counts_np = np.bincount(
            np.array(dataset.active_pos_sorted_mx, copy=False).astype(np.int32, copy=False),
            minlength=p
        ).astype(np.float32, copy=False)
        active_counts_np = np.maximum(active_counts_np, 1.0)
        active_counts_mx = mx.array(active_counts_np[:, None])

        word_scores_b = mx.zeros((p, v), dtype=mx.float32)
        ctx = mx.zeros((p, self.D), dtype=mx.float32) if self._a_mode == 'legacy' else None
        word_scores_a = mx.zeros((p, v), dtype=mx.float32)

        for idx, src in enumerate(dataset.active_src_unique_np.tolist()):
            start_i = int(np.array(dataset.active_src_start_mx[idx:idx + 1])[0])
            end_i = int(np.array(dataset.active_src_end_mx[idx:idx + 1])[0])
            pos_rows = dataset.active_pos_sorted_mx[start_i:end_i]
            phi_rows = dataset.active_phi_sorted_mx[start_i:end_i]

            row_word_scores = mx.matmul(phi_rows, self.b_word[int(src)])
            word_scores_b = word_scores_b.at[pos_rows].add(row_word_scores)

            if self.e_v is not None and self._a_mode == 'legacy':
                ctx_rows = self.e_v[int(src)][None, :] * phi_rows
                ctx = ctx.at[pos_rows].add(ctx_rows)
            elif self.e_v is not None and self._a_mode == 'v2':
                n_rows = int(phi_rows.shape[0])
                src_embed = self.e_v[int(src)][None, :, :]
                ctx_flat_rows = mx.reshape(src_embed * phi_rows[:, None, :], (n_rows, int(self.e_tgt_flat.shape[1])))
                slot_scores_rows = mx.matmul(ctx_flat_rows, mx.transpose(self.e_tgt_flat))
                word_scores_rows_a = mx.matmul(slot_scores_rows, self.word_slot_compact_t)
                word_scores_a = word_scores_a.at[pos_rows].add(word_scores_rows_a)

        if self.a_word is not None:
            word_scores_a = mx.matmul(ctx, self.a_word)

        word_scores_b = word_scores_b / active_counts_mx
        if self._a_mode == 'v2' and self.e_q is not None:
            word_scores_a = word_scores_a / active_counts_mx
        elif self.a_word is not None:
            word_scores_a = word_scores_a / (
                active_counts_mx * float(max(1, getattr(self.a, 'd', 1)))
            )

        t2 = time.time()
        _lam = getattr(self, 'namida', None)
        if _lam is not None:
            pred = mx.argmax(_lam * word_scores_b + (1 - _lam) * word_scores_a, axis=1)
        else:
            pred = mx.argmax(word_scores_b + word_scores_a, axis=1)
        mx.eval(pred)
        pred_np = np.array(pred)
        t3 = time.time()

        gold_np = self._gold_word_idx(dataset)
        t4 = time.time()
        valid = gold_np >= 0
        correct = int(np.sum(pred_np[valid] == gold_np[valid]))
        total = int(np.sum(valid))
        return EvalResult(accuracy=correct / total * 100 if total else 0.0, correct=correct, total=total)


SparseEvaluatorMLX = DenseEvaluatorMLX
