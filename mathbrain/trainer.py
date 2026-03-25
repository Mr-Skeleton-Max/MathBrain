import time
import math
import torch
import torch.nn.functional as F
from typing import List, Tuple
from .config import MathBrainConfig
from .decoder import SlotTransformer
from .encoder import compute_ema



class MathBrainTrainer:
    def __init__(self, config: MathBrainConfig, device: str = None):
        self.config = config
        if device is None:
            if torch.cuda.is_available(): device = 'cuda'
            elif torch.backends.mps.is_available(): device = 'mps'
            else: device = 'cpu'
        self.device = torch.device(device)
        self.decoder = SlotTransformer(config).to(self.device)
        self.rho = torch.tensor(config.rho_array, device=self.device)
        self.eps_q = config.eps_q
        self.optimizer = torch.optim.AdamW(
            self.decoder.parameters(), lr=1e-3, weight_decay=1e-4
        )

    def train_step(self, batch: dict) -> float:
        self.decoder.train()
        
        # B and T are already merged in the dataset or kept separate.
        # batch yields (B, T, ...) so we just flatten B*T
        B = batch['q_active'].shape[0]
        T = batch['q_active'].shape[1]
        BT = B * T
        
        Q_active = batch['q_active'].view(BT, -1, self.config.N).to(self.device)
        slot_indices = batch['slot_indices'].view(BT, -1).to(self.device)
        pad_mask = batch['pad_mask'].view(BT, -1).to(self.device)
        q_query = batch['q_query'].view(BT, 1, self.config.N).to(self.device)
        idx_query = batch['idx_query'].view(BT, 1).to(self.device)
        y = batch['targets'].view(BT).to(self.device)

        logits = self.decoder(Q_active, slot_indices, pad_mask, q_query, idx_query)
        
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

        for batch in dataloader:
            B = batch['q_active'].shape[0]
            T = batch['q_active'].shape[1]
            BT = B * T
            
            Q_active = batch['q_active'].view(BT, -1, self.config.N).to(self.device)
            slot_indices = batch['slot_indices'].view(BT, -1).to(self.device)
            pad_mask = batch['pad_mask'].view(BT, -1).to(self.device)
            q_query = batch['q_query'].view(BT, 1, self.config.N).to(self.device)
            idx_query = batch['idx_query'].view(BT, 1).to(self.device)
            y = batch['targets'].view(BT).to(self.device)

            logits = self.decoder(Q_active, slot_indices, pad_mask, q_query, idx_query)
            
            loss = F.cross_entropy(logits, y, reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
            
        return total_loss / max(total_tokens, 1)

    def fit(self, train_loader, val_loader=None, epochs: int = 5):
        print(f"Params: {sum(p.numel() for p in self.decoder.parameters()):,}")
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * len(train_loader), eta_min=1e-5
        )
        
        for epoch in range(epochs):
            t0 = time.time()
            train_loss = 0.0
            
            for step, batch in enumerate(train_loader):
                loss = self.train_step(batch)
                train_loss += loss
                scheduler.step()
                
                if step % 100 == 0:
                    dt = (time.time() - t0) * 1000 / (step + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch} | Step {step} | Loss: {loss:.4f} | LR: {lr:.2e} | {dt:.1f}ms / batch")
                    
            train_loss /= len(train_loader)
            train_ppl = math.exp(min(train_loss, 20.0))
            
            val_str = ""
            if val_loader:
                val_loss = self.evaluate(val_loader)
                val_ppl = math.exp(min(val_loss, 20.0))
                val_str = f"| Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
                
            print(f"=== Epoch {epoch} | Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f} {val_str} ===")
