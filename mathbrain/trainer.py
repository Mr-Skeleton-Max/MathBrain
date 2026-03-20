"""MathBrain Trainer — GPU-optimized Pipeline

全 GPU 训练 pipeline:
  1. cpu_preprocess:  HashRetina → per-sentence x_indicators (CPU parallel)
  2. gpu_preprocess:  Upload to GPU, build per-sentence data
  3. iter_batches:    Dynamic EMA + alive filter per batch
  4. train_step:      Phi CUDA → Triton bilinear → E_word matmul → CE
  5. evaluate:        Parallel GPU evaluation (same pipeline)

优化:
  - CUDA warp shuffle phi encoder (全寄存器, 0.6ms)
  - Triton fused bilinear (forward + atomic backward)
  - E_word = wp_dense @ E_tgt (消除 sparse word_proj)
  - Dense matmul backward (消除 atomic_add)
  - 动态 EMA per batch (省 GPU 内存)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MathBrainConfig
from .retina import HashRetina
from .streaming_preprocessor import build_vocab
from .gpu_preprocessor import gpu_preprocess, iter_batches_gpu
from .gpu_phi_encoder import GPUPhiEncoder
from .triton_kernels import triton_bilinear, build_wp_dense


class MathBrainTrainer:
    """GPU-optimized MathBrain trainer with unified train + evaluate API."""

    def __init__(self, config: MathBrainConfig, *, device: str = 'auto'):
        self.config = config
        self.device = self._resolve_device(device)
        self.retina = HashRetina(config)

        # Lazy-initialized on fit()
        self._phi_gpu: Optional[GPUPhiEncoder] = None
        self._vocab = None
        self._data = None
        self._E_src: Optional[nn.Embedding] = None
        self._E_tgt: Optional[nn.Parameter] = None
        self._wp_dense: Optional[torch.Tensor] = None
        self._optimizer = None
        self._scheduler = None

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            return torch.device('cpu')
        return torch.device(device)

    def fit(self, corpus: List[str], *,
            epochs: int = 320, batch_size: int = 4096,
            lr: float = 0.01, min_lr: float = 0.0,
            warmup_pct: float = 0.05, weight_decay: float = 1e-4,
            log_interval: int = 20, eval_corpus: Optional[List[str]] = None,
            eval_every: int = 0) -> Dict:
        """Train on corpus.

        Args:
            corpus: List of sentences
            epochs: Number of training epochs
            batch_size: Max positions per batch
            lr: Max learning rate
            min_lr: Final learning rate (default: lr/25000)
            warmup_pct: Fraction of steps for warmup (default: 0.05 = 5%)
            weight_decay: AdamW weight decay
            log_interval: Print loss every N epochs
            eval_corpus: Optional corpus for periodic evaluation
            eval_every: Evaluate every N epochs (0 = only at end)

        Returns:
            Dict with training results
        """
        device = self.device
        cfg = self.config
        t_total = time.time()

        # ── Phase 1: Build vocab ──
        t0 = time.time()
        vocab = build_vocab(corpus, self.retina)
        self._vocab = vocab
        print(f"[1] build_vocab: S={vocab.S}, V={vocab.V}, "
              f"{time.time()-t0:.2f}s")

        # ── Phase 2: Preprocess (CPU hash + GPU upload) ──
        t0 = time.time()
        data = gpu_preprocess(corpus, vocab, cfg, device)
        self._data = data
        t_prep = time.time() - t0
        print(f"[2] preprocess: {data.total_pos:,} positions, {t_prep:.2f}s")

        # ── Phase 3: Build model ──
        S, V, D = vocab.S, vocab.V, cfg.D_PHI
        dD = cfg.CP_RANK * D

        self._phi_gpu = GPUPhiEncoder(cfg, device=str(device))

        self._E_src = nn.Embedding(S, dD).to(device)
        self._E_tgt = nn.Parameter(torch.randn(S, dD, device=device) * 0.01)

        wp_off = torch.from_numpy(vocab.wp_word_offsets).to(device)
        wp_idx = torch.from_numpy(vocab.wp_slot_indices).to(device)
        wp_wt = torch.from_numpy(vocab.wp_slot_weights).to(device)
        self._wp_dense = build_wp_dense(wp_off, wp_idx, wp_wt, S, V, device)
        self._wp_off = wp_off
        self._wp_idx = wp_idx
        self._wp_wt = wp_wt

        self._optimizer = torch.optim.AdamW(
            [self._E_src.weight, self._E_tgt], lr=lr, weight_decay=weight_decay)

        n_params = self._E_src.weight.numel() + self._E_tgt.numel()
        print(f"[3] model: dD={dD}, params={n_params:,}, device={device}")

        if device.type == 'cuda':
            torch.set_float32_matmul_precision('high')

        # ── Warmup (also counts batches per epoch) ──
        n_batches_per_epoch = 0
        for b in iter_batches_gpu(data, batch_size, device, shuffle=False):
            if n_batches_per_epoch == 0:
                self._train_step(b)  # compile kernels
            n_batches_per_epoch += 1
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # ── LR Scheduler: warmup + cosine decay ──
        total_steps = epochs * n_batches_per_epoch + 1  # +1 safety for shuffle variance
        final_lr = min_lr if min_lr > 0 else lr / 25000
        div_factor = lr / (lr / 25.0)  # initial_lr = max_lr / div_factor
        final_div_factor = (lr / 25.0) / final_lr  # final_lr = initial_lr / final_div_factor
        self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self._optimizer, max_lr=lr, total_steps=total_steps,
            pct_start=warmup_pct, anneal_strategy='cos',
            div_factor=div_factor, final_div_factor=final_div_factor)
        print(f"  scheduler: OneCycleLR, {total_steps} steps, "
              f"lr={lr:.1e}→{final_lr:.1e}, warmup={warmup_pct:.0%}")

        # ── Phase 4: Train ──
        print(f"[4] Training: {data.total_pos:,} pos, "
              f"{len(data.sentences)} sentences")

        best_loss = float('inf')
        history = []
        _sched_step = 0

        for epoch in range(epochs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_epoch = time.time()
            epoch_loss = torch.tensor(0.0, device=device)
            n_b = 0

            for b in iter_batches_gpu(data, batch_size, device):
                loss = self._train_step(b)
                if _sched_step < total_steps:
                    self._scheduler.step()
                    _sched_step += 1
                epoch_loss += loss
                n_b += 1

            if device.type == 'cuda':
                torch.cuda.synchronize()
            ms = (time.time() - t_epoch) * 1000
            avg_loss = (epoch_loss / n_b).item()
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % log_interval == 0 or epoch == epochs - 1:
                cur_lr = self._scheduler.get_last_lr()[0]
                print(f"  epoch {epoch:4d}: loss={avg_loss:.6f} "
                      f"({ms:.0f}ms, {ms/n_b:.1f}ms/batch, lr={cur_lr:.2e})")

            # Periodic evaluation
            if eval_every > 0 and eval_corpus and (epoch + 1) % eval_every == 0:
                acc = self.evaluate(eval_corpus, verbose=False)['accuracy']
                print(f"         eval: {acc:.1f}%")

        total_time = time.time() - t_total
        print(f"\nDone: {total_time:.1f}s, best_loss={best_loss:.6f}")

        return {
            'best_loss': best_loss,
            'epochs': epochs,
            'total_time': total_time,
            'history': history,
        }

    def _train_step(self, batch) -> torch.Tensor:
        """Single training step on a batch."""
        phi_n = self._phi_gpu.encode_normalized(batch.flat_Q)
        Bi = batch.counts.shape[0]

        self._optimizer.zero_grad(set_to_none=True)
        ctx = triton_bilinear(
            self._E_src.weight, batch.flat_slots, phi_n,
            batch.pos_lo, batch.counts, Bi,
            self.config.CP_RANK * self.config.D_PHI, self.config.D_PHI,
            self._vocab.S)

        E_word = self._wp_dense @ self._E_tgt
        logits = ctx @ E_word.t()
        loss = F.cross_entropy(logits, batch.targets)
        loss.backward()
        self._optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def evaluate(self, corpus: List[str], *,
                 batch_size: int = 4096, verbose: bool = True) -> Dict:
        """Parallel GPU evaluation — same optimized pipeline as training.

        Args:
            corpus: List of sentences to evaluate
            batch_size: Max positions per batch
            verbose: Print results

        Returns:
            Dict with accuracy, correct, total
        """
        device = self.device
        cfg = self.config

        if self._E_src is None:
            raise RuntimeError("Call fit() first")

        # Use the same vocab (corpus should be subset of training vocab)
        vocab = self._vocab
        S, V, D = vocab.S, vocab.V, cfg.D_PHI
        dD = cfg.CP_RANK * D

        # Preprocess eval corpus (reuse same vocab)
        t0 = time.time()
        eval_data = gpu_preprocess(corpus, vocab, cfg, device, verbose=False)
        t_prep = time.time() - t0

        correct = 0
        total = 0
        total_loss = 0.0
        n_b = 0

        for b in iter_batches_gpu(eval_data, batch_size, device, shuffle=False):
            phi_n = self._phi_gpu.encode_normalized(b.flat_Q)
            Bi = b.counts.shape[0]

            ctx = triton_bilinear(
                self._E_src.weight, b.flat_slots, phi_n,
                b.pos_lo, b.counts, Bi, dD, D, S)

            E_word = self._wp_dense @ self._E_tgt
            logits = ctx @ E_word.t()

            # Accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == b.targets).sum().item()
            total += Bi

            # Loss
            loss = F.cross_entropy(logits, b.targets)
            total_loss += loss.item()
            n_b += 1

        acc = correct / max(total, 1) * 100
        avg_loss = total_loss / max(n_b, 1)

        if verbose:
            print(f"  Eval: {correct}/{total} = {acc:.1f}% "
                  f"(loss={avg_loss:.4f}, prep={t_prep:.2f}s)")

        return {
            'accuracy': acc,
            'correct': correct,
            'total': total,
            'loss': avg_loss,
        }

    def predict_topk(self, context_words: List[str], k: int = 5) -> List[tuple]:
        """Predict next word given context.

        Args:
            context_words: List of context words
            k: Number of top predictions

        Returns:
            List of (word, score) tuples
        """
        if self._E_src is None:
            raise RuntimeError("Call fit() first")

        device = self.device
        vocab = self._vocab
        cfg = self.config
        S, V, D = vocab.S, vocab.V, cfg.D_PHI
        dD = cfg.CP_RANK * D

        # Encode context through retina + EMA
        from .phi_encoder import CosineChaosEncoder
        phi_cpu = CosineChaosEncoder(cfg)

        # Build EMA state from context
        rho = np.array(cfg.RHO, dtype=np.float32)
        N = cfg.N
        q_state = np.zeros((S, N), dtype=np.float32)

        for word in context_words:
            slots = self.retina.encode(word)
            x = np.zeros(S, dtype=np.float32)
            for s, w in slots.items():
                if s < S:
                    x[s] = w
            q_state = rho * q_state + (1 - rho) * x[:, None]

        # Get active slots
        eps = 1e-6
        active_mask = np.any(np.abs(q_state) > eps, axis=1)
        active_slots = np.where(active_mask)[0]
        if len(active_slots) == 0:
            return []

        Q_vals = q_state[active_slots]  # (n_active, N)

        # Phi encode
        Q_t = torch.from_numpy(Q_vals.astype(np.float32)).to(device)
        phi_n = self._phi_gpu.encode_normalized(Q_t)

        # Bilinear context
        flat_slots = torch.from_numpy(active_slots.astype(np.int64)).to(device)
        pos_lo = torch.zeros(1, dtype=torch.long, device=device)
        counts = torch.tensor([len(active_slots)], dtype=torch.long, device=device)

        ctx = triton_bilinear(
            self._E_src.weight, flat_slots, phi_n,
            pos_lo, counts, 1, dD, D, S)

        E_word = self._wp_dense @ self._E_tgt
        logits = (ctx @ E_word.t()).squeeze(0)  # (V,)

        # Top-k predictions
        topk_vals, topk_idx = logits.topk(k)
        results = []
        idx_to_word = vocab.idx_to_word
        for val, idx in zip(topk_vals.cpu().numpy(), topk_idx.cpu().numpy()):
            word = idx_to_word.get(int(idx), f"<unk_{idx}>")
            results.append((word, float(val)))

        return results

    def save(self, path: str):
        """Save trained model."""
        if self._E_src is None:
            raise RuntimeError("No trained model")
        torch.save({
            'config': self.config,
            'E_src': self._E_src.state_dict(),
            'E_tgt': self._E_tgt.detach().cpu(),
            'vocab_S': self._vocab.S,
            'vocab_V': self._vocab.V,
        }, path)

    def load(self, path: str):
        """Load trained model."""
        ckpt = torch.load(path, map_location=self.device)
        self.config = ckpt['config']
        # Model must be re-initialized via fit() or manual setup
        print(f"Loaded config from {path}")
