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

import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MathBrainConfig
from .retina import HashRetina, IdentityRetina
from .streaming_preprocessor import build_vocab
from .gpu_preprocessor import gpu_preprocess, iter_batches_gpu
from .gpu_phi_encoder import GPUPhiEncoder
from .mlp_phi import MLPPhiEncoder, FourierPhiEncoder
from .triton_kernels import triton_bilinear, build_wp_dense
from .slot_transformer import SlotTransformer, flat_to_padded


class MathBrainTrainer:
    """GPU-optimized MathBrain trainer with unified train + evaluate API."""

    def __init__(self, config: MathBrainConfig, *, device: str = 'auto'):
        self.config = config
        self.device = self._resolve_device(device)
        if config.RETINA_MODE == 'identity':
            self.retina = IdentityRetina(config)
        else:
            self.retina = HashRetina(config)

        # Lazy-initialized on fit()
        self._phi_gpu: Optional[GPUPhiEncoder] = None
        self._mlp_phi: Optional[MLPPhiEncoder] = None
        self._ctx_mlp: Optional[nn.Sequential] = None
        self._slot_xfr: Optional[SlotTransformer] = None
        self._vocab = None
        self._data = None
        self._E_src: Optional[nn.Embedding] = None
        self._E_tgt: Optional[nn.Parameter] = None
        self._wp_dense: Optional[torch.Tensor] = None
        self._optimizer = None
        self._scheduler = None
        self._noise_sigma = 0.0

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
            scheduler: str = 'constant',
            noise_sigma: float = 0.0,
            val_split: float = 0.0,
            val_corpus_ext: Optional[List[str]] = None,
            phi_residual: bool = False,
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
            scheduler: LR schedule — 'constant', 'cosine', 'step', 'exp'
            noise_sigma: Gaussian noise σ on ctx (0=off, >0=sleep-during-wake)
            val_split: Fraction of corpus for validation (0=no split)
            log_interval: Print loss every N epochs
            eval_corpus: Optional corpus for periodic evaluation
            eval_every: Evaluate every N epochs (0 = only at end)

        Returns:
            Dict with training results
        """
        device = self.device
        cfg = self.config
        self._noise_sigma = noise_sigma
        self._phi_residual = phi_residual
        t_total = time.time()

        # ── Auto train/val split ──
        import random
        if val_corpus_ext is not None:
            # External val corpus provided
            train_corpus = corpus
            val_corpus = val_corpus_ext
            # Build vocab from both train + val
            corpus = train_corpus + val_corpus
            print(f"[0] external val: {len(train_corpus)} train, "
                  f"{len(val_corpus)} val")
        elif val_split > 0 and len(corpus) > 10:
            rng = random.Random(42)
            indices = list(range(len(corpus)))
            rng.shuffle(indices)
            n_val = max(1, int(len(corpus) * val_split))
            val_indices = set(indices[:n_val])
            train_corpus = [corpus[i] for i in range(len(corpus)) if i not in val_indices]
            val_corpus = [corpus[i] for i in val_indices]
            print(f"[0] split: {len(train_corpus)} train, {len(val_corpus)} val "
                  f"({val_split:.0%})")
        else:
            train_corpus = corpus
            val_corpus = None

        # ── Phase 1: Build vocab (from ALL corpus for shared vocab) ──
        t0 = time.time()
        vocab = build_vocab(corpus, self.retina)
        self._vocab = vocab
        print(f"[1] build_vocab: S={vocab.S}, V={vocab.V}, "
              f"{time.time()-t0:.2f}s")

        # ── Phase 2: Preprocess (train only) ──
        t0 = time.time()
        data = gpu_preprocess(train_corpus, vocab, cfg, device)
        self._data = data
        t_prep = time.time() - t0
        print(f"[2] preprocess: {data.total_pos:,} positions, {t_prep:.2f}s")

        # ── Phase 3: Build model ──
        S, V, D = vocab.S, vocab.V, cfg.D_PHI
        dD = cfg.CP_RANK * D

        if self._E_src is None and self._slot_xfr is None:
            # Fresh init
            if cfg.TRANSFORMER_D_MODEL > 0:
                # ── Transformer decoder path ──
                self._slot_xfr = SlotTransformer(
                    S=S, V=V, N_ema=cfg.N,
                    d_model=cfg.TRANSFORMER_D_MODEL,
                    nhead=cfg.TRANSFORMER_NHEAD,
                    num_layers=cfg.TRANSFORMER_LAYERS,
                    d_ffn=cfg.TRANSFORMER_FFN,
                    K_freq=8,
                    dropout=cfg.TRANSFORMER_DROPOUT,
                    pe_mode=cfg.TRANSFORMER_PE_MODE,
                    q_transform=cfg.TRANSFORMER_Q_TRANSFORM,
                    tie_weights=cfg.TRANSFORMER_TIE_WEIGHTS,
                ).to(device)
                n_params = sum(p.numel() for p in self._slot_xfr.parameters())
                qt_str = f', q={cfg.TRANSFORMER_Q_TRANSFORM}' if cfg.TRANSFORMER_Q_TRANSFORM != 'none' else ''
                print(f"[3] SlotTransformer: d_model={cfg.TRANSFORMER_D_MODEL}, "
                      f"layers={cfg.TRANSFORMER_LAYERS}, heads={cfg.TRANSFORMER_NHEAD}, "
                      f"pe={cfg.TRANSFORMER_PE_MODE}{qt_str}, "
                      f"params={n_params:,}, device={device}")
            else:
                # ── Original bilinear path ──
                if cfg.FOURIER_PHI_K > 0:
                    self._mlp_phi = FourierPhiEncoder(
                        N_ema=cfg.N, D_out=cfg.D_PHI,
                        K_freq=cfg.FOURIER_PHI_K).to(device)
                    self._phi_gpu = None
                    print(f"    φ encoder: Fourier (K={cfg.FOURIER_PHI_K}, "
                          f"fourier_dim={cfg.N*2*cfg.FOURIER_PHI_K}→{cfg.D_PHI})")
                elif cfg.MLP_PHI_HIDDEN > 0:
                    self._mlp_phi = MLPPhiEncoder(
                        N_in=cfg.N, D_out=cfg.D_PHI,
                        hidden=cfg.MLP_PHI_HIDDEN).to(device)
                    self._phi_gpu = None
                    print(f"    φ encoder: MLP (hidden={cfg.MLP_PHI_HIDDEN})")
                else:
                    self._phi_gpu = GPUPhiEncoder(cfg, device=str(device))
                    self._mlp_phi = None
                    print(f"    φ encoder: CosineChaos (α={cfg.CHAOS_ALPHA})")
                self._E_src = nn.Embedding(S, dD).to(device)
                self._E_tgt = nn.Parameter(torch.randn(S, dD, device=device) * 0.01)

                wp_off = torch.from_numpy(vocab.wp_word_offsets).to(device)
                wp_idx = torch.from_numpy(vocab.wp_slot_indices).to(device)
                wp_wt = torch.from_numpy(vocab.wp_slot_weights).to(device)
                self._wp_dense = build_wp_dense(wp_off, wp_idx, wp_wt, S, V, device)
                self._wp_off = wp_off
                self._wp_idx = wp_idx
                self._wp_wt = wp_wt
                # ctx MLP (Position C: post-aggregation nonlinearity)
                if cfg.MLP_CTX_HIDDEN > 0:
                    h = cfg.MLP_CTX_HIDDEN
                    self._ctx_mlp = nn.Sequential(
                        nn.Linear(dD, h),
                        nn.GELU(),
                        nn.Linear(h, dD),
                    ).to(device)
                    print(f"    ctx MLP: {dD}→{h}→{dD}")
                else:
                    self._ctx_mlp = None

                n_params = self._E_src.weight.numel() + self._E_tgt.numel()
                print(f"[3] model: dD={dD}, params NEW, device={device}")
                print(f"    params={n_params:,}")
        else:
            # Resume: reuse existing weights
            if self._slot_xfr is not None:
                print(f"[3] SlotTransformer: RESUMED, device={device}")
            else:
                print(f"[3] model: dD={dD}, params RESUMED, device={device}")

        if self._slot_xfr is not None:
            optim_params = list(self._slot_xfr.parameters())
        else:
            optim_params = [self._E_src.weight, self._E_tgt]
            if self._mlp_phi is not None:
                optim_params.extend(self._mlp_phi.parameters())
            if self._ctx_mlp is not None:
                optim_params.extend(self._ctx_mlp.parameters())
        self._optimizer = torch.optim.AdamW(
            optim_params, lr=lr, weight_decay=weight_decay)

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

        # ── LR Scheduler ──
        total_steps = epochs * n_batches_per_epoch + 1
        final_lr = min_lr if min_lr > 0 else lr / 25000

        if scheduler == 'cosine':
            div_factor = 25.0
            final_div_factor = (lr / div_factor) / final_lr
            self._scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self._optimizer, max_lr=lr, total_steps=total_steps,
                pct_start=warmup_pct, anneal_strategy='cos',
                div_factor=div_factor, final_div_factor=final_div_factor)
            sched_desc = f"cosine, lr={lr:.1e}→{final_lr:.1e}, warmup={warmup_pct:.0%}"
        elif scheduler == 'step':
            # Drop lr by 3× at 80% of training
            milestone = int(epochs * 0.8)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self._optimizer, milestones=[milestone], gamma=1/3)
            sched_desc = f"step, lr={lr:.1e}, drop 3× at epoch {milestone}"
        elif scheduler == 'exp':
            # Exponential decay: lr * 0.999^step
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, gamma=0.999)
            sched_desc = f"exp, lr={lr:.1e}, γ=0.999/step"
        else:  # 'constant'
            self._scheduler = None
            sched_desc = f"constant, lr={lr:.1e}"

        print(f"  scheduler: {sched_desc}")

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
                if self._scheduler and scheduler in ('cosine',):
                    if _sched_step < total_steps:
                        self._scheduler.step()
                        _sched_step += 1
                epoch_loss += loss
                n_b += 1

            # Per-epoch schedulers
            if self._scheduler and scheduler in ('step', 'exp'):
                self._scheduler.step()

            if device.type == 'cuda':
                torch.cuda.synchronize()
            ms = (time.time() - t_epoch) * 1000
            avg_loss = (epoch_loss / n_b).item()
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % log_interval == 0 or epoch == epochs - 1:
                cur_lr = (self._scheduler.get_last_lr()[0]
                          if self._scheduler else lr)
                msg = (f"  epoch {epoch:4d}: loss={avg_loss:.6f} "
                       f"({ms:.0f}ms, {ms/n_b:.1f}ms/batch, lr={cur_lr:.2e})")
                print(msg)

            # Periodic evaluation (val or custom eval_corpus)
            do_eval = (eval_every > 0 and (epoch + 1) % eval_every == 0)
            if do_eval:
                if val_corpus:
                    res = self.evaluate(val_corpus, verbose=False)
                    print(f"         val: acc={res['accuracy']:.1f}%, "
                          f"ppl={res['ppl']:.1f}, loss={res['loss']:.4f}")
                elif eval_corpus:
                    res = self.evaluate(eval_corpus, verbose=False)
                    print(f"         eval: acc={res['accuracy']:.1f}%, "
                          f"ppl={res['ppl']:.1f}")

        total_time = time.time() - t_total
        train_ppl = math.exp(min(best_loss, 20))  # clamp to avoid overflow
        print(f"\nDone: {total_time:.1f}s, best_loss={best_loss:.6f}, "
              f"train_ppl={train_ppl:.1f}")

        # Final val eval
        if val_corpus:
            res = self.evaluate(val_corpus)
            print(f"  Val PPL: {res['ppl']:.1f}")

        result = {
            'best_loss': best_loss,
            'train_ppl': train_ppl,
            'epochs': epochs,
            'total_time': total_time,
            'history': history,
        }
        if val_corpus:
            result['val_loss'] = res['loss']
            result['val_ppl'] = res['ppl']
            result['val_accuracy'] = res['accuracy']
        return result

    def _phi_encode(self, Q: torch.Tensor) -> torch.Tensor:
        """Encode Q → φ using either MLP or chaos."""
        if self._mlp_phi is not None:
            return self._mlp_phi(Q)
        phi_n = self._phi_gpu.encode_normalized(Q)
        if getattr(self, '_phi_residual', False):
            phi_n = phi_n + 1.0
        return phi_n

    def _train_step(self, batch) -> torch.Tensor:
        """Single training step on a batch."""
        self._optimizer.zero_grad(set_to_none=True)

        vicreg_w = self.config.VICREG_WEIGHT

        if self._slot_xfr is not None:
            # ── Transformer path ──
            padded_slots, padded_Q, mask, is_latest = flat_to_padded(
                batch.flat_slots, batch.flat_Q,
                batch.pos_lo, batch.counts, batch.flat_slots.device,
                flat_is_latest=batch.flat_is_latest)

            if vicreg_w > 0:
                logits, ctx = self._slot_xfr(
                    padded_slots, padded_Q, mask,
                    is_latest=is_latest, return_ctx=True)
            else:
                logits = self._slot_xfr(
                    padded_slots, padded_Q, mask, is_latest=is_latest)
        else:
            # ── Original bilinear path ──
            phi_n = self._phi_encode(batch.flat_Q)
            Bi = batch.counts.shape[0]
            ctx = triton_bilinear(
                self._E_src.weight, batch.flat_slots, phi_n,
                batch.pos_lo, batch.counts, Bi,
                self.config.CP_RANK * self.config.D_PHI, self.config.D_PHI,
                self._vocab.S,
                needs_phi_grad=(self._mlp_phi is not None))
            if self._ctx_mlp is not None:
                ctx = self._ctx_mlp(ctx)
            if self._noise_sigma > 0:
                ctx = ctx + torch.randn_like(ctx) * self._noise_sigma
            E_word = self._wp_dense @ self._E_tgt
            logits = ctx @ E_word.t()

        # ── Loss ──
        pred_mode = self.config.TRANSFORMER_PRED_MODE if self._slot_xfr is not None else 'ce'
        if pred_mode == 'innovation':
            # Pure MSE against one-hot target
            target_onehot = torch.zeros_like(logits)
            target_onehot.scatter_(1, batch.targets.unsqueeze(1), 1.0)
            loss = F.mse_loss(logits, target_onehot)
        elif pred_mode == 'hybrid':
            # CE + λ · MSE (CE for classification, MSE as geometric regularizer)
            ce_loss = F.cross_entropy(logits, batch.targets)
            target_onehot = torch.zeros_like(logits)
            target_onehot.scatter_(1, batch.targets.unsqueeze(1), 1.0)
            # Normalize logits to [0,1] via sigmoid for MSE comparison
            mse_loss = F.mse_loss(logits.sigmoid(), target_onehot)
            loss = ce_loss + self.config.INNOVATION_WEIGHT * mse_loss
        else:
            loss = F.cross_entropy(logits, batch.targets)

        # ── VICReg anti-collapse regularization ──
        if vicreg_w > 0 and self._slot_xfr is not None:
            B, D = ctx.shape
            if B > 1:
                ctx_centered = ctx - ctx.mean(dim=0, keepdim=True)
                std = ctx_centered.std(dim=0)  # (D,)

                # Variance: hinge loss — force std > 1
                var_loss = F.relu(1.0 - std).mean()

                # Covariance: off-diagonal elements → 0
                cov = (ctx_centered.T @ ctx_centered) / (B - 1)  # (D, D)
                # zero out diagonal
                off_diag = cov - torch.diag(cov.diag())
                cov_loss = (off_diag ** 2).sum() / D

                reg_loss = vicreg_w * (var_loss + cov_loss)
                loss = loss + reg_loss

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

        if self._E_src is None and self._slot_xfr is None:
            raise RuntimeError("Call fit() first")

        # Eval mode (disable dropout)
        was_training = False
        if self._slot_xfr is not None:
            was_training = self._slot_xfr.training
            self._slot_xfr.eval()

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
            Bi = b.counts.shape[0]

            if self._slot_xfr is not None:
                # ── Transformer path ──
                padded_slots, padded_Q, pad_mask, is_latest = flat_to_padded(
                    b.flat_slots, b.flat_Q, b.pos_lo, b.counts, device,
                    flat_is_latest=b.flat_is_latest)
                logits = self._slot_xfr(
                    padded_slots, padded_Q, pad_mask, is_latest=is_latest)
            else:
                # ── Original bilinear path ──
                phi_n = self._phi_encode(b.flat_Q)
                ctx = triton_bilinear(
                    self._E_src.weight, b.flat_slots, phi_n,
                    b.pos_lo, b.counts, Bi, dD, D, S)
                if self._ctx_mlp is not None:
                    ctx = self._ctx_mlp(ctx)
                E_word = self._wp_dense @ self._E_tgt
                logits = ctx @ E_word.t()

            # Accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == b.targets).sum().item()
            total += Bi

            # Loss
            pred_mode = self.config.TRANSFORMER_PRED_MODE if self._slot_xfr is not None else 'ce'
            if pred_mode == 'innovation':
                target_onehot = torch.zeros_like(logits)
                target_onehot.scatter_(1, b.targets.unsqueeze(1), 1.0)
                loss = F.mse_loss(logits, target_onehot)
            elif pred_mode == 'hybrid':
                ce_loss = F.cross_entropy(logits, b.targets)
                target_onehot = torch.zeros_like(logits)
                target_onehot.scatter_(1, b.targets.unsqueeze(1), 1.0)
                mse_loss = F.mse_loss(logits.sigmoid(), target_onehot)
                loss = ce_loss + self.config.INNOVATION_WEIGHT * mse_loss
            else:
                loss = F.cross_entropy(logits, b.targets)
            total_loss += loss.item()
            n_b += 1

        acc = correct / max(total, 1) * 100
        avg_loss = total_loss / max(n_b, 1)
        ppl = math.exp(min(avg_loss, 20))  # clamp to avoid overflow

        if verbose:
            print(f"  Eval: {correct}/{total} = {acc:.1f}% "
                  f"(loss={avg_loss:.4f}, ppl={ppl:.1f}, prep={t_prep:.2f}s)")

        # Restore training mode
        if was_training and self._slot_xfr is not None:
            self._slot_xfr.train()

        return {
            'accuracy': acc,
            'correct': correct,
            'total': total,
            'loss': avg_loss,
            'ppl': ppl,
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
        phi_n = self._phi_encode(Q_t)

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
        """Save trained model (config + weights + vocab)."""
        if self._E_src is None and self._slot_xfr is None:
            raise RuntimeError("No trained model")
        v = self._vocab
        save_dict = {
            'config': self.config,
            'vocab': {
                'word_to_idx': v.word_to_idx,
                'vocab_list': v.vocab_list,
                'slot_universe': v.slot_universe,
                'slot_to_compact': v.slot_to_compact,
                'S': v.S, 'V': v.V,
                'wp_word_offsets': v.wp_word_offsets,
                'wp_slot_indices': v.wp_slot_indices,
                'wp_slot_weights': v.wp_slot_weights,
            },
        }
        if self._slot_xfr is not None:
            save_dict['slot_xfr'] = self._slot_xfr.state_dict()
        else:
            save_dict['E_src'] = self._E_src.weight.detach().cpu()
            save_dict['E_tgt'] = self._E_tgt.detach().cpu()
            save_dict['wp_dense'] = self._wp_dense.detach().cpu()
            if self._mlp_phi is not None:
                save_dict['mlp_phi'] = self._mlp_phi.state_dict()
            if self._ctx_mlp is not None:
                save_dict['ctx_mlp'] = self._ctx_mlp.state_dict()
        torch.save(save_dict, path)
        print(f"Saved model to {path} (S={v.S}, V={v.V})")

    def load(self, path: str, corpus=None):
        """Load trained model. Restores weights + vocab.

        Args:
            path: Checkpoint path
            corpus: Optional corpus for preprocessing (needed for sleep/eval)
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.config = ckpt['config']
        cfg = self.config
        device = self.device

        # Restore vocab
        vd = ckpt['vocab']
        from .streaming_preprocessor import VocabInfo
        self._vocab = VocabInfo(**vd)

        S = vd['S']
        dD = cfg.CP_RANK * cfg.D_PHI

        # Restore model
        if 'slot_xfr' in ckpt:
            # ── Transformer path ──
            V = vd['V']
            self._slot_xfr = SlotTransformer(
                S=S, V=V, N_ema=cfg.N,
                d_model=cfg.TRANSFORMER_D_MODEL,
                nhead=cfg.TRANSFORMER_NHEAD,
                num_layers=cfg.TRANSFORMER_LAYERS,
                d_ffn=cfg.TRANSFORMER_FFN,
                K_freq=8,
                dropout=cfg.TRANSFORMER_DROPOUT,
                pe_mode=cfg.TRANSFORMER_PE_MODE,
                q_transform=cfg.TRANSFORMER_Q_TRANSFORM,
            ).to(device)
            self._slot_xfr.load_state_dict(ckpt['slot_xfr'])
            self._slot_xfr.eval()

            optim_params = list(self._slot_xfr.parameters())
            n_params = sum(p.numel() for p in optim_params)
            print(f"Loaded SlotTransformer from {path}: S={S}, V={V}, "
                  f"params={n_params:,}")
        else:
            # ── Original bilinear path ──
            self._E_src = nn.Embedding(S, dD).to(device)
            self._E_src.weight.data = ckpt['E_src'].to(device)
            self._E_tgt = nn.Parameter(ckpt['E_tgt'].to(device))
            self._wp_dense = ckpt['wp_dense'].to(device)

            # Restore phi encoder
            if cfg.FOURIER_PHI_K > 0 and 'mlp_phi' in ckpt:
                self._mlp_phi = FourierPhiEncoder(
                    N_ema=cfg.N, D_out=cfg.D_PHI,
                    K_freq=cfg.FOURIER_PHI_K).to(device)
                self._mlp_phi.load_state_dict(ckpt['mlp_phi'])
                self._mlp_phi.eval()
                self._phi_gpu = None
            elif cfg.MLP_PHI_HIDDEN > 0 and 'mlp_phi' in ckpt:
                self._mlp_phi = MLPPhiEncoder(
                    N_in=cfg.N, D_out=cfg.D_PHI,
                    hidden=cfg.MLP_PHI_HIDDEN).to(device)
                self._mlp_phi.load_state_dict(ckpt['mlp_phi'])
                self._mlp_phi.eval()
                self._phi_gpu = None
            else:
                self._phi_gpu = GPUPhiEncoder(cfg, device=str(device))
                self._mlp_phi = None

            # Restore ctx MLP
            if cfg.MLP_CTX_HIDDEN > 0 and 'ctx_mlp' in ckpt:
                h = cfg.MLP_CTX_HIDDEN
                self._ctx_mlp = nn.Sequential(
                    nn.Linear(dD, h),
                    nn.GELU(),
                    nn.Linear(h, dD),
                ).to(device)
                self._ctx_mlp.load_state_dict(ckpt['ctx_mlp'])
                self._ctx_mlp.eval()
            else:
                self._ctx_mlp = None

            # Restore wp sparse indices (for compatibility)
            wp_off = torch.from_numpy(vd['wp_word_offsets']).to(device)
            wp_idx = torch.from_numpy(vd['wp_slot_indices']).to(device)
            wp_wt = torch.from_numpy(vd['wp_slot_weights']).to(device)
            self._wp_off = wp_off
            self._wp_idx = wp_idx
            self._wp_wt = wp_wt

            optim_params = [self._E_src.weight, self._E_tgt]
            if self._mlp_phi is not None:
                optim_params.extend(self._mlp_phi.parameters())
            if self._ctx_mlp is not None:
                optim_params.extend(self._ctx_mlp.parameters())

            n_params = self._E_src.weight.numel() + self._E_tgt.numel()
            print(f"Loaded model from {path}: S={S}, V={vd['V']}, "
                  f"dD={dD}, params={n_params:,}")

        self._optimizer = torch.optim.AdamW(optim_params, lr=0.01)

        # Preprocess corpus if provided
        if corpus is not None:
            from .gpu_preprocessor import gpu_preprocess
            self._data = gpu_preprocess(corpus, self._vocab, cfg, device)

        return self

