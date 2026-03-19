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
import math
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
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


class _ChaosEncoder(nn.Module):
    """PyTorch CosineChaos encoder with optional learnable P.

    P = Q_ortho(A) @ diag(exp(log_s))
    Q_ortho via Cayley transform: Q = (I - A)(I + A)^{-1}, A is skew-symmetric

    Modes:
      - 'chaos': full CosineChaos with P (fixed or learnable)
      - 'raw': identity (pass through q_scaled, normalize only)
    """

    def __init__(self, N, D, n_folds=3, alpha=2.0, phi_mode='chaos',
                 learnable_P=False, P_init=None):
        super().__init__()
        self.N = N
        self.D = D
        self.n_folds = D if n_folds < 0 else n_folds  # -1 → L=D
        self.alpha = alpha
        self.phi_mode = phi_mode
        self.learnable_P = learnable_P

        if phi_mode in ('raw', 'precomputed'):
            # Raw or precomputed: no P, no chaos. D determined by input.
            self.D = D  # will be overridden by actual input dim
            return

        if learnable_P:
            # Cayley parametrization: skew-symmetric A → orthogonal Q
            self.A_param = nn.Parameter(torch.zeros(D, D) * 0.01)
            # Learnable log singular values
            if P_init is not None:
                _, s_init, _ = torch.linalg.svd(torch.from_numpy(P_init).float())
                self.log_s = nn.Parameter(torch.log(s_init[:D].clamp(min=0.01)))
            else:
                self.log_s = nn.Parameter(torch.zeros(min(D, N)))
        else:
            # Fixed P from numpy
            if P_init is not None:
                self.register_buffer('P', torch.from_numpy(P_init).float())
            else:
                rng = np.random.RandomState(42)
                P_np = rng.randn(D, N).astype(np.float32) * 0.5
                self.register_buffer('P', torch.from_numpy(P_np))

    def _get_P(self):
        """Get P matrix (fixed or from Cayley parametrization)."""
        if not self.learnable_P:
            return self.P

        # Cayley: Q = (I - A)(I + A)^{-1} where A is skew-symmetric
        A = self.A_param - self.A_param.t()  # enforce skew-symmetric
        I = torch.eye(self.D, device=A.device)
        Q = torch.linalg.solve(I + A, I - A)  # (D, D) orthogonal

        # Singular values
        s = torch.exp(self.log_s)  # (min(D,N),)

        # P = Q[:, :N] @ diag(s)  if D >= N
        # P = Q @ diag(s)[:D, :N] if D < N
        if self.D >= self.N:
            P = Q[:, :self.N] * s.unsqueeze(0)  # (D, N)
        else:
            P = Q * s[:self.D].unsqueeze(0)  # (D, D) then need to pad
            # This case shouldn't happen in practice (D >= N for chaos)
        return P

    def forward(self, q_scaled):
        """q_scaled: (B, M, N) → phi: (B, M, D), normalized."""
        if self.phi_mode in ('raw', 'precomputed'):
            # Raw or precomputed: just normalize (precomputed is already normalized,
            # but re-normalizing is a no-op for unit vectors)
            norms = torch.norm(q_scaled, dim=-1, keepdim=True).clamp(min=1e-8)
            return q_scaled / norms

        P = self._get_P()  # (D, N)

        # E = q_scaled @ P.T → (B, M, D)
        E = torch.matmul(q_scaled, P.t())

        # Chaos folding: M_t = cos(α * (shift(M_{t-1}) + E))
        B, M, D = E.shape
        state = torch.zeros_like(E)
        for _ in range(self.n_folds):
            shifted = torch.roll(state, 1, dims=-1)
            state = torch.cos(self.alpha * (shifted + E))

        # Normalize
        norms = torch.norm(state, dim=-1, keepdim=True).clamp(min=1e-8)
        return state / norms


class _SleepModule(nn.Module):
    """Bilinear low-rank predictor with integrated CosineChaos.

    q_scaled → ChaosEncoder → φ → E_src ⊙ φ → ctx → E_tgt → slot_scores → word_proj → CE

    E_src uses nn.Embedding for efficient sparse backward (avoids scatter_add).
    """

    def __init__(self, S, d_rank, D, N, word_proj_t, phi_mode='chaos',
                 learnable_P=False, P_init=None, n_folds=3, alpha=2.0):
        super().__init__()
        self.d_rank = d_rank

        self.chaos = _ChaosEncoder(
            N, D, n_folds=n_folds, alpha=alpha,
            phi_mode=phi_mode, learnable_P=learnable_P, P_init=P_init,
        )
        D_eff = self.chaos.D

        self.D = D_eff
        self.S = S
        self.E_src = nn.Embedding(S, d_rank * D_eff)
        nn.init.normal_(self.E_src.weight, std=0.01)
        # E_tgt stored flat: (S, d*D) — no reshape needed in forward
        self.E_tgt = nn.Parameter(torch.randn(S, d_rank * D_eff) * 0.01)
        self.register_buffer('word_proj_t', word_proj_t)

    def forward(self, b_active, b_q, b_mask, b_target):
        phi = self.chaos(b_q)  # (B, M, D)

        B, M, D = phi.shape
        d = self.d_rank

        # E_src lookup: (B, M, d*D)
        src_flat = self.E_src(b_active)

        # phi * mask: (B, M, D)
        phi_masked = phi * b_mask.unsqueeze(-1)

        # 手动 bmm 替代 einsum，避免 reshape 触发 clone:
        # 目标: ctx[b,r,d] = Σ_m src[b,m,r,d] * phi_masked[b,m,d]
        #
        # 重排为 bmm:
        #   src_flat: (B, M, d*D) → (B*D, M, d) via permute
        #   phi_masked: (B, M, D) → (B*D, M, 1)
        #   bmm → (B*D, d, 1) → (B, D, d) → (B, d*D)
        #
        # 但这需要 permute 和 contiguous，同样有 copy 开销。
        #
        # 更好的方案: 把 einsum 拆成 element-wise mul + sum
        # src_4d: (B, M, d, D) * phi_masked: (B, M, 1, D) → (B, M, d, D) → sum(dim=1) → (B, d, D)
        src_4d = src_flat.view(B, M, d, D)
        # 用 mul + sum 替代 einsum（PyTorch 对这个模式有更好的 fusion）
        weighted = (src_4d * phi_masked.unsqueeze(2)).sum(dim=1)  # (B, d, D)

        ctx_flat = weighted.view(B, d * D)  # view 不 copy（weighted 已连续）

        # Scoring: ctx → slot_scores → word_logits
        slot_scores = ctx_flat @ self.E_tgt.t()  # (B, S)

        active_count = b_mask.sum(1, keepdim=True).clamp(min=1.0)
        slot_scores = slot_scores / active_count

        word_logits = slot_scores @ self.word_proj_t  # (B, V), word_proj_t 已是 fp32
        target_idx = b_target.argmax(dim=1)
        loss = F.cross_entropy(word_logits, target_idx)
        return loss


class _SleepModuleTransformer(nn.Module):
    """Transformer-over-slots predictor with integrated CosineChaos.

    q_scaled → ChaosEncoder → φ → slot_emb + phi_proj(φ) → attention → slot_head → word_proj → CE
    """

    def __init__(self, S, N, D, V, word_proj_t, phi_mode='chaos',
                 learnable_P=False, P_init=None, n_folds=3, alpha=2.0,
                 d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.S = S

        self.chaos = _ChaosEncoder(
            N, D, n_folds=n_folds, alpha=alpha,
            phi_mode=phi_mode, learnable_P=learnable_P, P_init=P_init,
        )
        D_eff = self.chaos.D

        self.slot_emb = nn.Embedding(S, d_model)
        self.phi_proj = nn.Linear(D_eff, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        self.slot_head = nn.Linear(d_model, S)
        self.register_buffer('word_proj_t', word_proj_t)

    def forward(self, b_active, b_q, b_mask, b_target):
        phi = self.chaos(b_q)  # (B, M, D_eff)

        tok = self.slot_emb(b_active) + self.phi_proj(phi)  # (B, M, d_model)
        pad_mask = (b_mask == 0)

        out = self.transformer(tok, src_key_padding_mask=pad_mask)

        slot_logits = self.slot_head(out)  # (B, M, S)
        mask_expanded = b_mask.unsqueeze(-1)
        slot_scores = (slot_logits * mask_expanded).sum(dim=1)
        active_count = b_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        slot_scores = slot_scores / active_count

        word_logits = slot_scores @ self.word_proj_t
        target_idx = b_target.argmax(dim=1)
        loss = F.cross_entropy(word_logits, target_idx)
        return loss


class MathBrainTrainer:
    """MathBrain A 模块训练器。

    直接用 gold teacher 做 dream sleep，不需要 B wake。
    """

    def __init__(self, model, device: str = 'auto', predictor: str = 'linear',
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 phi_mode: str = 'chaos', learnable_P: bool = False):
        self.model = model
        self.config = model.config
        self.device = _resolve_device(device)
        self.predictor = predictor
        self.phi_mode = phi_mode
        self.learnable_P = learnable_P
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

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
        # 固定 P 时预计算 phi，不进 autograd 图 → backward 快 ~2x
        precompute = not self.learnable_P

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

                alpha_scale = self.model.phi_encoder.alpha_scale
                q_scaled = (Q_vals * alpha_scale).astype(np.float32)

                if precompute and self.phi_mode == 'chaos':
                    phi = self.model.phi_encoder.encode(Q_vals)
                    norms = np.linalg.norm(phi, axis=1, keepdims=True)
                    feature = (phi / (norms + 1e-8)).astype(np.float32)
                elif precompute and self.phi_mode == 'raw':
                    norms = np.linalg.norm(q_scaled, axis=1, keepdims=True)
                    feature = (q_scaled / (norms + 1e-8)).astype(np.float32)
                else:
                    feature = q_scaled  # learnable P: raw Q

                gold_word = words[i + 1]
                gold_slots = np.fromiter(encoded[i + 1].keys(), dtype=np.int32)
                positions.append((active_slots.copy(), feature.copy(),
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
        feat_dim = positions[0][1].shape[1]  # D (chaos precomputed) or N (raw/learnable)
        max_active = max(len(p[0]) for p in positions)

        active_local = np.zeros((n_pos, max_active), dtype=np.int64)
        q_padded = np.zeros((n_pos, max_active, feat_dim), dtype=np.float32)
        active_mask = np.zeros((n_pos, max_active), dtype=np.float32)
        teacher_labels = np.zeros(n_pos, dtype=np.int64)

        for i, (act, q_sc, _, gw) in enumerate(positions):
            n = len(act)
            local_ids = [self._slot_to_compact[int(s)] for s in act]
            active_local[i, :n] = local_ids
            q_padded[i, :n] = q_sc
            active_mask[i, :n] = 1.0
            teacher_labels[i] = word_to_idx.get(gw, 0)

        one_hot = np.zeros((n_pos, V), dtype=np.float32)
        one_hot[np.arange(n_pos), teacher_labels] = 1.0

        return {
            'active_local': torch.from_numpy(active_local),
            'q_padded': torch.from_numpy(q_padded),
            'active_mask': torch.from_numpy(active_mask),
            'one_hot': torch.from_numpy(one_hot),
            'S': S, 'V': V, 'feat_dim': feat_dim, 'n_pos': n_pos,
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
        S, V, feat_dim = data['S'], data['V'], data['feat_dim']
        d_rank = int(self.config.CP_RANK)
        D = int(self.config.D_PHI)
        N_ema = int(self.config.N)
        precomputed = not self.learnable_P

        # 当预计算 phi 时，forward 里跳过 ChaosEncoder
        # feat_dim = D (chaos) 或 N (raw) — 已经是最终特征
        # 当 learnable P 时，feat_dim = N — forward 里需要 ChaosEncoder
        phi_mode_eff = 'precomputed' if precomputed else self.phi_mode

        print(f"  positions={n_pos}, slots={S}, vocab={V}, "
              f"d_rank={d_rank}, feat_dim={feat_dim}, device={self.device}, "
              f"predictor={self.predictor}, phi_mode={self.phi_mode}, "
              f"learnable_P={self.learnable_P}")

        # 数据 → device
        active_local = data['active_local'].to(self.device)
        q_padded = data['q_padded'].to(self.device)
        active_mask = data['active_mask'].to(self.device)
        one_hot = data['one_hot'].to(self.device)

        word_proj_t = torch.from_numpy(self._word_proj.T.astype(np.float32)).to(self.device)

        P_init = self.model.phi_encoder.P  # (D, N) numpy or None

        # 模型
        # D_eff: 特征维度。预计算时 = feat_dim，learnable P 时由 ChaosEncoder 决定
        D_for_encoder = feat_dim if precomputed else D

        if self.predictor == 'transformer':
            net = _SleepModuleTransformer(
                S, N_ema, D_for_encoder, V, word_proj_t,
                phi_mode=phi_mode_eff,
                learnable_P=self.learnable_P,
                P_init=P_init,
                n_folds=self.config.CHAOS_N_FOLDS,
                alpha=self.config.CHAOS_ALPHA,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
            ).to(self.device)
        else:
            net = _SleepModule(
                S, d_rank, D_for_encoder, N_ema, word_proj_t,
                phi_mode=phi_mode_eff,
                learnable_P=self.learnable_P,
                P_init=P_init,
                n_folds=self.config.CHAOS_N_FOLDS,
                alpha=self.config.CHAOS_ALPHA,
            ).to(self.device)

        # TF32 (CUDA only, 无精度风险)
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')
        compiled_forward = net
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')  # TF32 for matmul
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
            epoch_t0 = time.time()
            perm = torch.randperm(n_pos, device=self.device)
            epoch_loss = 0.0

            for bi in range(n_batches):
                s_idx = bi * batch_size
                e_idx = min(s_idx + batch_size, n_pos)
                idx = perm[s_idx:e_idx]

                optimizer.zero_grad(set_to_none=True)
                loss = compiled_forward(
                    active_local[idx], q_padded[idx],
                    active_mask[idx], one_hot[idx])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            epoch_loss /= n_batches

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}

            if epoch % log_interval == 0 or epoch == epochs - 1:
                epoch_ms = (time.time() - epoch_t0) * 1000
                print(f"      epoch {epoch:4d}: loss={epoch_loss:.6f}  "
                      f"({epoch_ms:.0f}ms, {n_batches} batches)")

        # 写回 — 保留完整 net 用于 evaluate
        net.load_state_dict(best_state)
        self._trained_net = net
        self._trained_net.eval()

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

    def _compute_phi(self, Q_vals):
        """Compute phi from Q_vals, respecting phi_mode setting."""
        if self.phi_mode == 'raw':
            alpha_scale = self.model.phi_encoder.alpha_scale
            phi = Q_vals * alpha_scale  # (n, N)
        else:
            phi = self.model.phi_encoder.encode(Q_vals)  # (n, D)
        phi_norms = np.linalg.norm(phi, axis=1, keepdims=True)
        return (phi / (phi_norms + 1e-8)).astype(np.float32)

    @torch.no_grad()
    def evaluate(self, corpus: List[str]) -> Dict[str, float]:
        net = self._trained_net
        net.eval()
        retina = self.model.retina
        s2c = self._slot_to_compact
        vocab_list = self._vocab_list
        alpha_scale = self.model.phi_encoder.alpha_scale
        precomputed = not self.learnable_P
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

                # 计算特征（和 _build_positions 一致）
                q_scaled = (Q_vals * alpha_scale).astype(np.float32)
                if precomputed and self.phi_mode == 'chaos':
                    phi_np = self.model.phi_encoder.encode(Q_vals)
                    norms = np.linalg.norm(phi_np, axis=1, keepdims=True)
                    feature = (phi_np / (norms + 1e-8)).astype(np.float32)
                elif precomputed and self.phi_mode == 'raw':
                    norms = np.linalg.norm(q_scaled, axis=1, keepdims=True)
                    feature = (q_scaled / (norms + 1e-8)).astype(np.float32)
                else:
                    feature = q_scaled

                local_ids = [s2c[int(s)] for s in active_slots if int(s) in s2c]
                n_valid = len(local_ids)
                if n_valid == 0:
                    continue

                b_active = torch.tensor([local_ids], dtype=torch.long, device=self.device)
                b_feat = torch.tensor(feature[:n_valid][None], dtype=torch.float32,
                                      device=self.device)
                b_mask = torch.ones(1, n_valid, dtype=torch.float32, device=self.device)

                phi = net.chaos(b_feat)  # passthrough for precomputed

                if self.predictor == 'transformer':
                    tok = net.slot_emb(b_active) + net.phi_proj(phi)
                    pad_mask = torch.zeros(1, n_valid, dtype=torch.bool, device=self.device)
                    out = net.transformer(tok, src_key_padding_mask=pad_mask)
                    slot_logits = net.slot_head(out)
                    slot_scores = slot_logits.sum(dim=1) / n_valid
                else:
                    src_flat = net.E_src(b_active)  # (1, n_valid, d_rank*D_eff)
                    src_emb = src_flat.view(1, n_valid, net.d_rank, net.D)
                    phi_masked = phi * b_mask.unsqueeze(-1)
                    weighted = torch.einsum('bmrd,bmd->brd', src_emb, phi_masked)
                    ctx_flat = weighted.reshape(1, -1)
                    slot_scores = ctx_flat @ net.E_tgt.t()
                    slot_scores = slot_scores / n_valid

                word_logits = slot_scores @ net.word_proj_t
                pred_idx = word_logits.argmax(dim=1).item()

                pred_word = vocab_list[pred_idx] if pred_idx < len(vocab_list) else ''
                total += 1
                if pred_word == words[i + 1]:
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
