import time
import math
import torch
import torch.nn.functional as F
from typing import List, Tuple
from .config import MathBrainConfig
from .decoder import SlotTransformer
from .encoder import compute_ema, compute_ema_v5, recompute_slots

def extract_active_slots(
    Q: torch.Tensor,
    Q_max: torch.Tensor,
    eps_q: float,
    max_active: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy: extract active slots from full Q tensor."""
    BT, V, N = Q.shape
    k = int((Q_max >= eps_q).sum(dim=1).max().clamp(min=1, max=max_active).item())
    top_vals, top_indices = torch.topk(Q_max, k, dim=1, sorted=False)
    pad_mask = top_vals < eps_q
    expanded = top_indices.unsqueeze(-1).expand(-1, -1, N)
    Q_active = torch.gather(Q, 1, expanded)
    return Q_active, top_indices, pad_mask


def extract_topk_indices(
    Q_max: torch.Tensor,
    eps_q: float,
    max_active: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract topk slot indices from Q_max.
    Always returns FIXED k=max_active to avoid padding and enable FlashAttention.
    """
    BT, V = Q_max.shape
    k = min(max_active, V)
    top_vals, top_indices = torch.topk(Q_max, k, dim=1, sorted=False)
    pad_mask = top_vals < eps_q
    return top_indices, pad_mask


class MathBrainTrainer:
    def __init__(self, config: MathBrainConfig, device: str = None, lr: float = 1e-3,
                 ema_chunk: int = 32, compile_decoder: bool = False):
        self.config = config
        if device is None:
            if torch.cuda.is_available(): device = 'cuda'
            elif torch.backends.mps.is_available(): device = 'mps'
            else: device = 'cpu'
        self.device = torch.device(device)
        self.decoder = SlotTransformer(config).to(self.device)
        if compile_decoder and self.device.type == 'cuda':
            self.decoder = torch.compile(self.decoder, mode='max-autotune')
        self.rho = torch.tensor(config.rho_array, device=self.device)
        self.eps_q = config.eps_q
        self.ema_chunk = ema_chunk
        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(), lr=lr, weight_decay=1e-4
        )
        self.carry = None

    def _init_carry(self, B: int):
        if self.carry is None or self.carry.shape[0] != B:
            self.carry = torch.zeros(B, self.config.vocab_size, self.config.N, device=self.device, dtype=torch.float32)

    def _prepare_features(self, inputs: torch.Tensor):
        return self._prepare_features_v5(inputs, ema_chunk=self.ema_chunk)

    def _prepare_features_v5(self, inputs: torch.Tensor, ema_chunk: int = 32):
        """
        V5b fused pipeline. No one_hot. Q_max fused into kernel.

        Pipeline per sub-chunk:
          Phase 1: compute_ema_v5(tokens, rho, S) → carry + Q_max. ~0.25ms.
          Phase 2: topk on Q_max(B,S) → slot_indices(B, k). ~0.04ms.
          Phase 3: recompute_slots(tokens, rho, S, merged) → Q_active + q_query. ~0.08ms.
        """
        B, T = inputs.shape
        S, N = self.config.vocab_size, self.config.N

        all_Q_active   = []
        all_slot_idx   = []
        all_pad_mask   = []
        all_q_query    = []
        all_idx_query  = []

        self._init_carry(B)

        with torch.no_grad():
            for t_start in range(0, T, ema_chunk):
                t_end = min(t_start + ema_chunk, T)
                sub_inputs = inputs[:, t_start:t_end].contiguous()
                tc = t_end - t_start
                Btc = B * tc

                # Phase 1: fused kernel — reads tokens directly, writes carry + Q_max
                prev_carry = self.carry
                self.carry, Qm_chunk = compute_ema_v5(
                    sub_inputs, self.rho, S, init_state=prev_carry)
                self.carry = self.carry.detach()

                # Phase 2: topk on (B, S)
                si_b, pm_b = extract_topk_indices(Qm_chunk, self.config.eps_q,
                                                  self.config.max_active_slots)
                k = si_b.shape[1]
                si = si_b.unsqueeze(1).expand(B, tc, k).reshape(Btc, k)
                pm = pm_b.unsqueeze(1).expand(B, tc, k).reshape(Btc, k)

                # Phase 3: merged recompute (no one_hot)
                iq = sub_inputs.reshape(Btc)
                merged_slots = torch.cat([si, iq.unsqueeze(1)], dim=1)
                merged = recompute_slots(sub_inputs, self.rho, S, merged_slots,
                                         init_state=prev_carry)
                Qa = merged[:, :k, :]
                qq = merged[:, k:, :]

                all_Q_active.append(Qa)
                all_slot_idx.append(si)
                all_pad_mask.append(pm)
                all_q_query.append(qq)
                all_idx_query.append(iq)

                del prev_carry

        # Pad across sub-chunks to a common max_active, then concat
        global_max = max(a.shape[1] for a in all_Q_active)

        def pad2d(t, mx, pad_val=0):
            """Pad dim-1 of (M, k, ...) to (M, mx, ...)"""
            d = mx - t.shape[1]
            if d == 0:
                return t
            pad_shape = list(t.shape)
            pad_shape[1] = d
            return torch.cat([t, t.new_full(pad_shape, pad_val)], dim=1)

        def pad_bool(t, mx):
            d = mx - t.shape[1]
            if d == 0:
                return t
            return torch.cat([t, t.new_ones(t.shape[0], d, dtype=torch.bool)], dim=1)

        Q_active    = torch.cat([pad2d(a, global_max, 0)  for a in all_Q_active], dim=0)
        slot_indices= torch.cat([pad2d(s, global_max, 0)  for s in all_slot_idx], dim=0)
        pad_mask    = torch.cat([pad_bool(p, global_max)  for p in all_pad_mask], dim=0)
        q_query     = torch.cat(all_q_query, dim=0)
        idx_query   = torch.cat(all_idx_query, dim=0)
        BT = B * T

        return Q_active, slot_indices, pad_mask, q_query, idx_query, BT

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.decoder.train()
        
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        Q_active, slot_indices, pad_mask, q_query, idx_query, BT = self._prepare_features(inputs)
        y = targets.reshape(BT)

        logits = self.decoder(Q_active, slot_indices, pad_mask, q_query, idx_query.unsqueeze(1))
        
        loss = F.cross_entropy(logits, y)
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        self.decoder.eval()
        total_loss = 0.0
        total_tokens = 0

        # Reset carry for evaluation to test from clean slate
        self.carry = None

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            Q_active, slot_indices, pad_mask, q_query, idx_query, BT = self._prepare_features(inputs)
            y = targets.reshape(BT)

            logits = self.decoder(Q_active, slot_indices, pad_mask, q_query, idx_query.unsqueeze(1))
            
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
            
        return total_loss / max(total_tokens, 1)

    def fit(self, train_loader, val_loader=None, epochs: int = 5):
        print(f"Params: {sum(p.numel() for p in self.decoder.parameters()):,}")
        
        # Scheduler
        # Ensure we don't divide by zero if dataloader length is not directly known for generators, 
        # so we will use a fixed high number or a stepped scheduler if len is available.
        try:
            total_steps = epochs * len(train_loader)
        except TypeError:
            total_steps = epochs * 10000 # Fallback for generator

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-5
        )
        
        for epoch in range(epochs):
            t0 = time.time()
            train_loss = 0.0
            steps = 0
            
            self.carry = None # Reset carry at the beginning of each epoch
            
            for step, (inputs, targets) in enumerate(train_loader):
                loss = self.train_step(inputs, targets)
                train_loss += loss
                scheduler.step()
                steps += 1
                
                if step % 100 == 0:
                    dt = (time.time() - t0) * 1000 / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch} | Step {step} | Loss: {loss:.4f} | LR: {lr:.2e} | {dt:.1f}ms / batch")
                    
            train_loss /= max(steps, 1)
            train_ppl = math.exp(min(train_loss, 20.0))
            
            val_str = ""
            if val_loader:
                val_loss = self.evaluate(val_loader)
                val_ppl = math.exp(min(val_loss, 20.0))
                val_str = f"| Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
                
            print(f"=== Epoch {epoch} | Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} {val_str} ===")
