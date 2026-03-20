"""MathBrain A Trainer — PyTorch 实现

高效的 gold-teacher dream sleep 训练器，支持 CUDA/MPS/CPU。
使用 DataLoader + pin_memory 实现 CPU→GPU 异步流水线，
数据留在 CPU，仅 batch 粒度传 GPU。

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .config import MathBrainConfig
from .wake_dataset import WakeDataset
from .inference_fast import SparseMatrixDecoder
from .data_cache import preprocess_corpus, load_cache


def _resolve_device(device: str) -> torch.device:
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device)


# =====================================================================
# CosineChaos encoder (PyTorch, for learnable_P path only)
# =====================================================================

class _ChaosEncoder(nn.Module):
    """PyTorch CosineChaos encoder with optional learnable P.

    Modes:
      - 'precomputed': identity passthrough (phi already computed in data_cache)
      - 'chaos': full CosineChaos with fixed or learnable P
      - 'raw': normalize-only passthrough
    """

    def __init__(self, N, D, n_folds=-1, alpha=2.1, phi_mode='precomputed',
                 learnable_P=False, P_init=None):
        super().__init__()
        self.N = N
        self.D = D
        self.n_folds = D if n_folds < 0 else n_folds
        self.alpha = alpha
        self.phi_mode = phi_mode
        self.learnable_P = learnable_P

        if phi_mode in ('raw', 'precomputed'):
            return

        # Chaos mode with P matrix
        if learnable_P:
            self.A_param = nn.Parameter(torch.zeros(D, D) * 0.01)
            if P_init is not None:
                _, s_init, _ = torch.linalg.svd(
                    torch.from_numpy(P_init).float())
                self.log_s = nn.Parameter(
                    torch.log(s_init[:min(D, N)].clamp(min=0.01)))
            else:
                self.log_s = nn.Parameter(torch.zeros(min(D, N)))
        else:
            if P_init is not None:
                self.register_buffer('P', torch.from_numpy(P_init).float())
            else:
                rng = np.random.RandomState(42)
                P_np = rng.randn(D, N).astype(np.float32) * 0.5
                self.register_buffer('P', torch.from_numpy(P_np))

    def _get_P(self):
        if not self.learnable_P:
            return self.P
        A = self.A_param - self.A_param.t()
        I = torch.eye(self.D, device=A.device)
        Q = torch.linalg.solve(I + A, I - A)
        s = torch.exp(self.log_s)
        k = min(self.D, self.N)
        return Q[:, :k] * s.unsqueeze(0)

    def forward(self, x):
        """x: (B, M, feat_dim) → phi: (B, M, D), normalized."""
        if self.phi_mode in ('raw', 'precomputed'):
            norms = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
            return x / norms

        P = self._get_P()
        E = torch.matmul(x, P.t())
        state = torch.zeros_like(E)
        for _ in range(self.n_folds):
            shifted = torch.roll(state, 1, dims=-1)
            state = torch.cos(self.alpha * (shifted + E))
        norms = torch.norm(state, dim=-1, keepdim=True).clamp(min=1e-8)
        return state / norms


# =====================================================================
# Predictor modules
# =====================================================================

class _SleepModule(nn.Module):
    """Bilinear low-rank predictor with integrated CosineChaos.

    Forward:
      q_scaled → ChaosEncoder → φ → E_src ⊙ φ → ctx → E_tgt → slot_scores
      → word_proj → CE loss
    """

    def __init__(self, S, d_rank, D, N, word_proj_t, phi_mode='precomputed',
                 learnable_P=False, P_init=None, n_folds=-1, alpha=2.1):
        super().__init__()
        self.d_rank = d_rank
        self.chaos = _ChaosEncoder(
            N, D, n_folds=n_folds, alpha=alpha,
            phi_mode=phi_mode, learnable_P=learnable_P, P_init=P_init)
        D_eff = self.chaos.D
        self.D = D_eff
        self.S = S
        self.E_src = nn.Embedding(S, d_rank * D_eff)
        nn.init.normal_(self.E_src.weight, std=0.01)
        self.E_tgt = nn.Parameter(torch.randn(S, d_rank * D_eff) * 0.01)
        self.register_buffer('word_proj_t', word_proj_t)

    def forward(self, b_active, b_q, b_mask, b_target_idx):
        phi = self.chaos(b_q)
        B, M, D = phi.shape
        d = self.d_rank

        src_flat = self.E_src(b_active)
        phi_masked = phi * b_mask.unsqueeze(-1)
        src_4d = src_flat.view(B, M, d, D)
        weighted = (src_4d * phi_masked.unsqueeze(2)).sum(dim=1)
        ctx_flat = weighted.view(B, d * D)

        slot_scores = ctx_flat @ self.E_tgt.t()
        active_count = b_mask.sum(1, keepdim=True).clamp(min=1.0)
        slot_scores = slot_scores / active_count

        word_logits = slot_scores @ self.word_proj_t
        loss = F.cross_entropy(word_logits, b_target_idx)
        return loss


class _SleepModuleTransformer(nn.Module):
    """Transformer-over-slots predictor with integrated CosineChaos."""

    def __init__(self, S, N, D, V, word_proj_t, phi_mode='precomputed',
                 learnable_P=False, P_init=None, n_folds=-1, alpha=2.1,
                 d_model=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.S = S
        self.chaos = _ChaosEncoder(
            N, D, n_folds=n_folds, alpha=alpha,
            phi_mode=phi_mode, learnable_P=learnable_P, P_init=P_init)
        D_eff = self.chaos.D
        self.slot_emb = nn.Embedding(S, d_model)
        self.phi_proj = nn.Linear(D_eff, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu')
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        self.slot_head = nn.Linear(d_model, S)
        self.register_buffer('word_proj_t', word_proj_t)

    def forward(self, b_active, b_q, b_mask, b_target_idx):
        phi = self.chaos(b_q)
        tok = self.slot_emb(b_active) + self.phi_proj(phi)
        pad_mask = (b_mask == 0)
        out = self.transformer(tok, src_key_padding_mask=pad_mask)
        slot_logits = self.slot_head(out)
        mask_expanded = b_mask.unsqueeze(-1)
        slot_scores = (slot_logits * mask_expanded).sum(dim=1)
        active_count = b_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        slot_scores = slot_scores / active_count
        word_logits = slot_scores @ self.word_proj_t
        loss = F.cross_entropy(word_logits, b_target_idx)
        return loss


# =====================================================================
# Trainer
# =====================================================================

class MathBrainTrainer:
    """MathBrain A 模块训练器 (gold teacher dream sleep)."""

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
        self._vocab_list = None
        self._trained_net = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, corpus: List[str], *,
            epochs: int = 320,
            batch_size: int = 128,
            lr: float = 0.01,
            weight_decay: float = 1e-4,
            log_interval: int = 20,
            cache_dir: str = 'cache/',
            corpus_path: str = None,
            num_workers: int = 2,
            storage_dtype: str = 'fp32',
            ) -> Dict[str, float]:
        t0 = time.time()

        # Phase 1: 预处理（或加载缓存）
        cache_path = preprocess_corpus(
            corpus, self.model,
            cache_dir=cache_dir,
            corpus_path=corpus_path,
            phi_mode=self.phi_mode,
            learnable_P=self.learnable_P,
            storage_dtype=storage_dtype)
        data = load_cache(cache_path, self.device)
        meta = data['meta']

        n_pos = meta['n_pos']
        S = meta['S']
        V = meta['V']
        feat_dim = meta['feat_dim']
        d_rank = int(self.config.CP_RANK)
        D = int(self.config.D_PHI)
        N_ema = int(self.config.N)
        precomputed = not self.learnable_P
        phi_mode_eff = 'precomputed' if precomputed else self.phi_mode

        print(f"  positions={n_pos}, slots={S}, vocab={V}, "
              f"d_rank={d_rank}, feat_dim={feat_dim}, device={self.device}, "
              f"predictor={self.predictor}, phi_mode={self.phi_mode}")

        # Rebuild mappings for evaluate()
        self._rebuild_mappings(data)

        word_proj_t = data['word_proj_t']
        P_init = self.model.phi_encoder.P
        D_for_encoder = feat_dim if precomputed else D

        # Build network
        if self.predictor == 'transformer':
            net = _SleepModuleTransformer(
                S, N_ema, D_for_encoder, V, word_proj_t,
                phi_mode=phi_mode_eff,
                learnable_P=self.learnable_P, P_init=P_init,
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
                learnable_P=self.learnable_P, P_init=P_init,
                n_folds=self.config.CHAOS_N_FOLDS,
                alpha=self.config.CHAOS_ALPHA,
            ).to(self.device)

        # torch.compile (CUDA only)
        compiled_forward = net
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')
            try:
                compiled_forward = torch.compile(net)
            except Exception:
                pass

        # Optimizer
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay)
        n_batches = max(1, (n_pos + batch_size - 1) // batch_size)
        total_steps = epochs * n_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=0)

        # ---- Build DataLoader ----
        # Data stays on CPU; pin_memory + non_blocking for async DMA.
        all_active = data['active']
        all_phi = data['phi']
        all_mask = data['mask']
        target_idx = data['target']
        on_gpu = data['on_gpu']

        if on_gpu:
            # Data already on GPU — use simple index-based loop
            print(f"  data_mode=gpu (all data on device)")
            loader = None
        else:
            # Data on CPU (mmap or regular numpy) — use DataLoader pipeline
            # Convert numpy views to contiguous tensors for pin_memory
            t_active = torch.from_numpy(
                np.ascontiguousarray(all_active).astype(np.int64))
            t_phi = torch.from_numpy(
                np.ascontiguousarray(all_phi).astype(np.float32))
            t_mask = torch.from_numpy(
                np.ascontiguousarray(all_mask).astype(np.float32))
            t_target = target_idx.cpu()  # already torch int64

            dataset = TensorDataset(t_active, t_phi, t_mask, t_target)

            # pin_memory only works for CUDA
            use_pin = (self.device.type == 'cuda')
            # num_workers=0 on MPS to avoid fork issues
            actual_workers = num_workers if self.device.type == 'cuda' else 0

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=actual_workers,
                pin_memory=use_pin,
                persistent_workers=(actual_workers > 0),
                drop_last=False)

            print(f"  data_mode=dataloader (pin_memory={use_pin}, "
                  f"workers={actual_workers})")

        best_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            epoch_t0 = time.time()
            epoch_loss = 0.0
            n_batch_actual = 0

            if loader is not None:
                # DataLoader path: CPU data → device
                # non_blocking only safe with CUDA + pin_memory
                nb = use_pin  # True only on CUDA
                for b_active, b_phi, b_mask, b_target in loader:
                    b_active = b_active.to(self.device, non_blocking=nb)
                    b_phi = b_phi.to(self.device, non_blocking=nb)
                    b_mask = b_mask.to(self.device, non_blocking=nb)
                    b_target = b_target.to(self.device, non_blocking=nb)

                    optimizer.zero_grad(set_to_none=True)
                    loss = compiled_forward(b_active, b_phi, b_mask, b_target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                    n_batch_actual += 1
            else:
                # GPU path: all data on device, index directly
                perm = torch.randperm(n_pos, device=self.device)
                for bi in range(n_batches):
                    s_idx = bi * batch_size
                    e_idx = min((bi + 1) * batch_size, n_pos)
                    idx = perm[s_idx:e_idx]

                    optimizer.zero_grad(set_to_none=True)
                    loss = compiled_forward(
                        all_active[idx], all_phi[idx],
                        all_mask[idx], target_idx[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                    n_batch_actual += 1

            epoch_loss /= max(n_batch_actual, 1)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}

            if epoch % log_interval == 0 or epoch == epochs - 1:
                epoch_ms = (time.time() - epoch_t0) * 1000
                print(f"      epoch {epoch:4d}: loss={epoch_loss:.6f}  "
                      f"({epoch_ms:.0f}ms, {n_batch_actual} batches)")

        # Restore best weights
        if best_state is not None:
            net.load_state_dict(best_state)
        self._trained_net = net
        self._trained_net.eval()

        elapsed = time.time() - t0
        print(f"  训练完成: best_loss={best_loss:.6f}, elapsed={elapsed:.1f}s")

        return {'final_loss': epoch_loss, 'best_loss': best_loss,
                'elapsed_s': elapsed}

    def _rebuild_mappings(self, data: dict):
        """Rebuild slot_to_compact and vocab_list from cache sidecar."""
        slot_universe = data['slot_universe']
        vocab_list = data['vocab_list']
        self._slot_universe = slot_universe
        self._slot_to_compact = {
            int(s): i for i, s in enumerate(slot_universe.tolist())}
        self._vocab_list = vocab_list

    # ------------------------------------------------------------------
    # evaluate (batched, fast)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, corpus: List[str], *,
                 batch_size: int = 512,
                 cache_dir: str = 'cache/',
                 corpus_path: str = None) -> Dict[str, float]:
        """Batched evaluate using cache — same data path as fit().

        比 evaluate_online() 快 100-1000x (GPU batch forward)。
        """
        net = self._trained_net
        net.eval()

        cache_path = preprocess_corpus(
            corpus, self.model,
            cache_dir=cache_dir, corpus_path=corpus_path,
            phi_mode=self.phi_mode, learnable_P=self.learnable_P)
        data = load_cache(cache_path, self.device)
        meta = data['meta']
        n_pos = meta['n_pos']
        vocab_list = data['vocab_list']
        on_gpu = data['on_gpu']

        correct = 0
        total = n_pos

        if on_gpu:
            # All data on GPU — index directly
            all_active = data['active']
            all_phi = data['phi']
            all_mask = data['mask']
            target_idx = data['target']

            for s in range(0, n_pos, batch_size):
                e = min(s + batch_size, n_pos)
                ba = all_active[s:e]
                bp = all_phi[s:e]
                bm = all_mask[s:e]
                bt = target_idx[s:e]

                phi = net.chaos(bp)
                if self.predictor == 'transformer':
                    tok = net.slot_emb(ba) + net.phi_proj(phi)
                    pad_mask = (bm == 0)
                    out = net.transformer(tok, src_key_padding_mask=pad_mask)
                    slot_logits = net.slot_head(out)
                    mask_expanded = bm.unsqueeze(-1)
                    slot_scores = (slot_logits * mask_expanded).sum(dim=1)
                    active_count = bm.sum(dim=1, keepdim=True).clamp(min=1.0)
                    slot_scores = slot_scores / active_count
                else:
                    B, M, D = phi.shape
                    d = net.d_rank
                    src_flat = net.E_src(ba)
                    phi_masked = phi * bm.unsqueeze(-1)
                    src_4d = src_flat.view(B, M, d, D)
                    weighted = (src_4d * phi_masked.unsqueeze(2)).sum(dim=1)
                    ctx_flat = weighted.view(B, d * D)
                    slot_scores = ctx_flat @ net.E_tgt.t()
                    active_count = bm.sum(1, keepdim=True).clamp(min=1.0)
                    slot_scores = slot_scores / active_count

                word_logits = slot_scores @ net.word_proj_t
                preds = word_logits.argmax(dim=1)
                correct += (preds == bt).sum().item()
        else:
            # Data on CPU — batch transfer
            t_active = torch.from_numpy(
                np.ascontiguousarray(data['active']).astype(np.int64))
            t_phi = torch.from_numpy(
                np.ascontiguousarray(data['phi']).astype(np.float32))
            t_mask = torch.from_numpy(
                np.ascontiguousarray(data['mask']).astype(np.float32))
            t_target = data['target'].cpu()

            for s in range(0, n_pos, batch_size):
                e = min(s + batch_size, n_pos)
                ba = t_active[s:e].to(self.device)
                bp = t_phi[s:e].to(self.device)
                bm = t_mask[s:e].to(self.device)
                bt = t_target[s:e].to(self.device)

                phi = net.chaos(bp)
                if self.predictor == 'transformer':
                    tok = net.slot_emb(ba) + net.phi_proj(phi)
                    pad_mask = (bm == 0)
                    out = net.transformer(tok, src_key_padding_mask=pad_mask)
                    slot_logits = net.slot_head(out)
                    mask_expanded = bm.unsqueeze(-1)
                    slot_scores = (slot_logits * mask_expanded).sum(dim=1)
                    active_count = bm.sum(dim=1, keepdim=True).clamp(min=1.0)
                    slot_scores = slot_scores / active_count
                else:
                    B, M, D = phi.shape
                    d = net.d_rank
                    src_flat = net.E_src(ba)
                    phi_masked = phi * bm.unsqueeze(-1)
                    src_4d = src_flat.view(B, M, d, D)
                    weighted = (src_4d * phi_masked.unsqueeze(2)).sum(dim=1)
                    ctx_flat = weighted.view(B, d * D)
                    slot_scores = ctx_flat @ net.E_tgt.t()
                    active_count = bm.sum(1, keepdim=True).clamp(min=1.0)
                    slot_scores = slot_scores / active_count

                word_logits = slot_scores @ net.word_proj_t
                preds = word_logits.argmax(dim=1)
                correct += (preds == bt).sum().item()

        acc = correct / total * 100 if total > 0 else 0.0
        return {'accuracy': acc, 'correct': correct, 'total': total}

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """Save trained model (net state_dict + metadata)."""
        if self._trained_net is None:
            raise RuntimeError("请先 fit() 再保存")

        checkpoint = {
            'net_state_dict': self._trained_net.state_dict(),
            'slot_universe': self._slot_universe,
            'vocab': sorted(self.model.vocab),
            'word_to_slots': {
                w: slots.tolist()
                for w, slots in self.model.word_to_slots.items()
            },
            'predictor': self.predictor,
            'phi_mode': self.phi_mode,
            'learnable_P': self.learnable_P,
            'config': {
                'K': self.config.K,
                'D_PHI': self.config.D_PHI,
                'CP_RANK': self.config.CP_RANK,
                'N': self.config.N,
                'RHO': self.config.RHO,
                'PHI_MODE': self.config.PHI_MODE,
                'CHAOS_N_FOLDS': self.config.CHAOS_N_FOLDS,
                'CHAOS_ALPHA': self.config.CHAOS_ALPHA,
                'CHAOS_NO_P': self.config.CHAOS_NO_P,
                'PHI_SIGMA': self.config.PHI_SIGMA,
                'NGRAM_SIZE': self.config.NGRAM_SIZE,
                'NGRAM_SCALES': self.config.NGRAM_SCALES,
            },
        }

        torch.save(checkpoint, path)
        print(f"  模型已保存到 {path}")

    def load(self, path: Union[str, Path]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        vocab = checkpoint['vocab']
        word_to_slots = checkpoint['word_to_slots']
        self.model.vocab = set(vocab)
        self.model.word_to_slots = {
            w: np.array(slots, dtype=np.int32)
            for w, slots in word_to_slots.items()
        }
        self.model._decoder_dirty = True

        slot_universe = checkpoint['slot_universe']
        self._slot_universe = slot_universe
        self._slot_to_compact = {
            int(s): i for i, s in enumerate(slot_universe.tolist())
        }

        # TODO: reconstruct net from checkpoint config and load state_dict
        print(f"  模型已从 {path} 加载 (slots={len(slot_universe)}, "
              f"vocab={len(vocab)})")
