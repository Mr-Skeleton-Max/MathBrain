"""MathBrain A Trainer — PyTorch 实现

高效的 gold-teacher dream sleep 训练器，支持 CUDA/MPS/CPU。

用法：
    from mathbrain import MathBrain, MathBrainConfig
    from mathbrain.trainer import MathBrainTrainer

    model = MathBrain(MathBrainConfig())
    trainer = MathBrainTrainer(model)
    trainer.fit(corpus, epochs=320)
    trainer.save('model.pt')
    print(trainer.evaluate(corpus))
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F

from .config import MathBrainConfig
from .wake_dataset import WakeDataset
from .inference_fast import SparseMatrixDecoder


def _resolve_device(device: str) -> torch.device:
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device)


class _SleepModule(torch.nn.Module):
    """前向传播模块。

    用 (S, d, D) 3D 存储，但用 einsum 避免显式 tile/expand。
    """

    def __init__(self, S, d_rank, D, word_proj_t):
        super().__init__()
        self.d_rank = d_rank
        self.D = D
        self.E_src = torch.nn.Parameter(torch.randn(S, d_rank, D) * 0.01)
        self.E_tgt = torch.nn.Parameter(torch.randn(S, d_rank, D) * 0.01)
        self.register_buffer('word_proj_t', word_proj_t)

    def forward(self, b_active, b_phi, b_mask, b_target):
        src_emb = self.E_src[b_active]                            # (B, M, d, D)
        # einsum: 对 D 维做逐元素乘，对 M 维求和（带 mask）
        # ctx[b, r, m] = Σ_j src_emb[b,j,r,m] * phi[b,j,m] * mask[b,j]
        weighted = torch.einsum('bmrd,bmd,bm->brd', src_emb, b_phi, b_mask)
        # weighted: (B, d, D)

        B_size = weighted.shape[0]
        ctx_flat = weighted.reshape(B_size, -1)                   # (B, d*D)

        E_tgt_flat = self.E_tgt.reshape(-1, self.d_rank * self.D) # (S, d*D)
        slot_scores = ctx_flat @ E_tgt_flat.t()                   # (B, S)

        active_count = b_mask.sum(1, keepdim=True).clamp(min=1.0)
        slot_scores = slot_scores / active_count

        word_scores = slot_scores @ self.word_proj_t              # (B, V)
        loss = F.mse_loss(word_scores, b_target)
        return loss


class MathBrainTrainer:
    """MathBrain A 模块训练器。

    直接用 gold teacher 做 dream sleep，不需要 B wake。
    """

    def __init__(self, model, device: str = 'auto'):
        self.model = model
        self.config = model.config
        self.device = _resolve_device(device)

        self._slot_universe = None
        self._slot_to_compact = None
        self._word_proj = None
        self._vocab_list = None

    # ------------------------------------------------------------------
    # 数据构建
    # ------------------------------------------------------------------

    def _build_positions(self, corpus: List[str]):
        retina = self.model.retina
        positions = []

        for sentence in corpus:
            words = WakeDataset._tokenize(sentence)
            if len(words) < 2:
                continue
            encoded = [retina.encode(w) for w in words]
            self.model.q.reset()

            for i in range(len(words) - 1):
                self.model.q.update(encoded[i])
                active_slots, Q_vals = self.model.q.get_active()
                if len(active_slots) == 0:
                    continue

                phi = self.model.phi_encoder.encode(Q_vals)
                phi_norms = np.linalg.norm(phi, axis=1, keepdims=True)
                phi_hat = (phi / (phi_norms + 1e-8)).astype(np.float32)

                gold_word = words[i + 1]
                gold_slots = np.fromiter(encoded[i + 1].keys(), dtype=np.int32)

                positions.append((active_slots.copy(), phi_hat.copy(),
                                  gold_slots, gold_word))

        return positions

    def _build_tensors(self, positions):
        # slot universe
        all_slots = set()
        for act, _, gs, gw in positions:
            all_slots.update(act.tolist())
            all_slots.update(gs.tolist())
        for word in self.model.vocab:
            if word in self.model.word_to_slots:
                all_slots.update(int(s) for s in self.model.word_to_slots[word])
        self._slot_universe = np.array(sorted(all_slots), dtype=np.int32)
        self._slot_to_compact = {int(s): i for i, s in enumerate(self._slot_universe.tolist())}
        S = len(self._slot_universe)

        # decoder + vocab
        decoder = SparseMatrixDecoder(
            self.model.vocab, self.model.word_to_slots, self.config.K)
        self._vocab_list = decoder.word_list
        V = len(self._vocab_list)
        word_to_idx = {w: i for i, w in enumerate(self._vocab_list)}

        # word projection
        word_proj = np.zeros((V, S), dtype=np.float32)
        for wi, w in enumerate(self._vocab_list):
            if w in self.model.word_to_slots:
                slots = self.model.word_to_slots[w]
                compact = [self._slot_to_compact[int(sl)]
                           for sl in slots if int(sl) in self._slot_to_compact]
                if compact:
                    word_proj[wi, compact] = 1.0 / len(compact)
        self._word_proj = word_proj

        # pad positions
        n_pos = len(positions)
        D_phi = positions[0][1].shape[1]
        max_active = max(len(p[0]) for p in positions)

        active_local = np.zeros((n_pos, max_active), dtype=np.int64)
        phi_padded = np.zeros((n_pos, max_active, D_phi), dtype=np.float32)
        active_mask = np.zeros((n_pos, max_active), dtype=np.float32)
        teacher_labels = np.zeros(n_pos, dtype=np.int64)

        for i, (act, phi_h, _, gw) in enumerate(positions):
            n = len(act)
            local_ids = [self._slot_to_compact[int(s)] for s in act]
            active_local[i, :n] = local_ids
            phi_padded[i, :n] = phi_h
            active_mask[i, :n] = 1.0
            teacher_labels[i] = word_to_idx.get(gw, 0)

        one_hot = np.zeros((n_pos, V), dtype=np.float32)
        one_hot[np.arange(n_pos), teacher_labels] = 1.0

        return {
            'active_local': torch.from_numpy(active_local),
            'phi_padded': torch.from_numpy(phi_padded),
            'active_mask': torch.from_numpy(active_mask),
            'one_hot': torch.from_numpy(one_hot),
            'S': S, 'V': V, 'D_phi': D_phi, 'n_pos': n_pos,
        }

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, corpus: List[str], *,
            epochs: int = 320,
            batch_size: int = 128,
            lr: float = 0.01,
            weight_decay: float = 1e-4,
            log_interval: int = 20,
            ) -> Dict[str, float]:
        t0 = time.time()

        for s in corpus:
            for w in WakeDataset._tokenize(s):
                self.model._ensure_word(w)

        positions = self._build_positions(corpus)
        data = self._build_tensors(positions)
        n_pos = data['n_pos']
        S, V, D_phi = data['S'], data['V'], data['D_phi']
        d_rank = int(self.config.CP_RANK)

        print(f"  positions={n_pos}, slots={S}, vocab={V}, "
              f"d_rank={d_rank}, D={D_phi}, device={self.device}")

        # 数据 → device
        active_local = data['active_local'].to(self.device)
        phi_padded = data['phi_padded'].to(self.device)
        active_mask = data['active_mask'].to(self.device)
        one_hot = data['one_hot'].to(self.device)

        word_proj_t = torch.from_numpy(self._word_proj.T.astype(np.float32)).to(self.device)

        # 模型
        net = _SleepModule(S, d_rank, D_phi, word_proj_t).to(self.device)

        # 尝试 torch.compile（CUDA 有效，MPS/CPU 可能 fallback）
        compiled_forward = net
        if self.device.type == 'cuda':
            try:
                compiled_forward = torch.compile(net)
            except Exception:
                pass

        # 优化器
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        n_batches = max(1, (n_pos + batch_size - 1) // batch_size)
        total_steps = epochs * n_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=0)

        best_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            perm = torch.randperm(n_pos, device=self.device)
            epoch_loss = 0.0

            for bi in range(n_batches):
                s_idx = bi * batch_size
                e_idx = min(s_idx + batch_size, n_pos)
                idx = perm[s_idx:e_idx]

                loss = compiled_forward(
                    active_local[idx], phi_padded[idx],
                    active_mask[idx], one_hot[idx])

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            epoch_loss /= n_batches

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()
                              if k in ('E_src', 'E_tgt')}

            if epoch % log_interval == 0 or epoch == epochs - 1:
                print(f"      epoch {epoch:4d}: loss={epoch_loss:.6f}")

        # 写回
        self._write_back(best_state['E_src'].numpy(), best_state['E_tgt'].numpy())

        elapsed = time.time() - t0
        print(f"  训练完成: best_loss={best_loss:.6f}, elapsed={elapsed:.1f}s")

        return {'final_loss': epoch_loss, 'best_loss': best_loss, 'elapsed_s': elapsed}

    def _write_back(self, E_src_np, E_tgt_np):
        """E_src_np/E_tgt_np: (S, d*D) flat 或 (S, d, D) 3D。"""
        slot_universe = self._slot_universe
        d_rank = int(self.config.CP_RANK)
        D = int(self.config.d)
        if E_src_np.ndim == 2:
            E_src_np = E_src_np.reshape(len(slot_universe), d_rank, D)
            E_tgt_np = E_tgt_np.reshape(len(slot_universe), d_rank, D)
        E_src_dict = {int(s): E_src_np[i] for i, s in enumerate(slot_universe.tolist())}
        E_tgt_dict = {int(s): E_tgt_np[i] for i, s in enumerate(slot_universe.tolist())}
        self.model.a.update_from_sleep(E_src_dict, E_tgt_dict)

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(self, corpus: List[str]) -> Dict[str, float]:
        decoder = SparseMatrixDecoder(
            self.model.vocab, self.model.word_to_slots, self.config.K)
        retina = self.model.retina
        correct = total = 0

        for sentence in corpus:
            words = WakeDataset._tokenize(sentence)
            if len(words) < 2:
                continue
            encoded = [retina.encode(w) for w in words]
            self.model.q.reset()
            for i in range(len(words) - 1):
                self.model.q.update(encoded[i])
                active_slots, Q_vals = self.model.q.get_active()
                if len(active_slots) == 0:
                    continue
                phi = self.model.phi_encoder.encode(Q_vals)
                phi_norms = np.linalg.norm(phi, axis=1, keepdims=True)
                phi_hat = (phi / (phi_norms + 1e-8)).astype(np.float32)
                scores = self.model.a.predict(active_slots, phi_hat)
                pred = decoder.decode_top_k(scores, 1)[0][0]
                total += 1
                if pred == words[i + 1]:
                    correct += 1

        acc = correct / total * 100 if total > 0 else 0.0
        return {'accuracy': acc, 'correct': correct, 'total': total}

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        if self._slot_universe is None:
            raise RuntimeError("请先 fit() 再保存")

        slot_universe = self._slot_universe
        S = len(slot_universe)
        d_rank = int(self.config.CP_RANK)
        D = int(self.config.d)

        E_src_np = np.zeros((S, d_rank, D), dtype=np.float32)
        E_tgt_np = np.zeros((S, d_rank, D), dtype=np.float32)
        for i, s in enumerate(slot_universe.tolist()):
            s = int(s)
            if s in self.model.a.E_src:
                E_src_np[i] = self.model.a.E_src[s]
            if s in self.model.a.E_tgt:
                E_tgt_np[i] = self.model.a.E_tgt[s]

        checkpoint = {
            'E_src': E_src_np,
            'E_tgt': E_tgt_np,
            'slot_universe': slot_universe,
            'vocab': sorted(self.model.vocab),
            'word_to_slots': {
                w: slots.tolist()
                for w, slots in self.model.word_to_slots.items()
            },
            'config': {
                'K': self.config.K,
                'D_PHI': self.config.D_PHI,
                'CP_RANK': self.config.CP_RANK,
                'N': self.config.N,
                'RHO': self.config.RHO,
                'PHI_MODE': self.config.PHI_MODE,
                'CHAOS_N_FOLDS': self.config.CHAOS_N_FOLDS,
                'CHAOS_ALPHA': self.config.CHAOS_ALPHA,
                'PHI_SIGMA': self.config.PHI_SIGMA,
                'NGRAM_SIZE': self.config.NGRAM_SIZE,
                'NGRAM_SCALES': self.config.NGRAM_SCALES,
            },
        }

        torch.save(checkpoint, path)
        print(f"  模型已保存到 {path}")

    def load(self, path: Union[str, Path]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        E_src_np = checkpoint['E_src']
        E_tgt_np = checkpoint['E_tgt']
        slot_universe = checkpoint['slot_universe']

        vocab = checkpoint['vocab']
        word_to_slots = checkpoint['word_to_slots']
        self.model.vocab = set(vocab)
        self.model.word_to_slots = {
            w: np.array(slots, dtype=np.int32)
            for w, slots in word_to_slots.items()
        }
        self.model._decoder_dirty = True

        self._slot_universe = slot_universe
        self._slot_to_compact = {
            int(s): i for i, s in enumerate(slot_universe.tolist())
        }

        self._write_back(E_src_np, E_tgt_np)

        print(f"  模型已从 {path} 加载 (slots={len(slot_universe)}, vocab={len(vocab)})")
