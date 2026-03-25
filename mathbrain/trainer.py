import time
import math
import torch
import torch.nn.functional as F
from typing import List, Tuple
from .config import MathBrainConfig
from .decoder import SlotTransformer
from .encoder import compute_ema

def extract_active_slots(Q: torch.Tensor, eps_q: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized extraction of active slots on GPU using sorting padding.
    Replaces 400 lines of complex CPU preprocessing.
    """
    # Flatten Q: (B*T, V, N)
    BT, V, N = Q.shape
    Q_max = Q.abs().amax(dim=-1) # (BT, V)
    active_mask = (Q_max >= eps_q).int() # (BT, V)

    # Sort descending so 1s (active) come before 0s (inactive)
    sorted_mask, sorted_indices = torch.sort(active_mask, dim=1, descending=True)
    max_active = int(sorted_mask.sum(dim=1).max().item())
    max_active = max(max_active, 1)

    # Truncate to the maximum active slot count
    sorted_indices = sorted_indices[:, :max_active] # (BT, max_active)
    pad_mask = sorted_mask[:, :max_active] == 0     # (BT, max_active) True where padded

    # Gather Q_active
    expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, N)
    Q_active = torch.gather(Q, 1, expanded_indices) # (BT, max_active, N)

    return Q_active, sorted_indices, pad_mask

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

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        x: (B, T)
        y: (B, T)
        """
        self.decoder.train()
        B, T = x.shape
        V = self.config.vocab_size
        
        # 1. Active Slot Triggers: x(t) = 1 if slot activated, 0 otherwise
        # Instead of allocating (B, T, V) directly which is fine but takes memory, 
        # F.one_hot handles it cleanly.
        x_onehot = F.one_hot(x, num_classes=V).to(self.device, dtype=torch.float32)
        
        # 2. Parallel EMA Scan
        # Q will be (B, T, V, N)
        Q = compute_ema(x_onehot, self.rho)
        
        # 3. Flatten and Extract Active Slots
        Q_flat = Q.view(B * T, V, self.config.N)
        Q_active, slot_indices, pad_mask = extract_active_slots(Q_flat, self.eps_q)
        
        # Construct Query from latest tokens
        idx_query = x.to(self.device).view(B * T)
        # Gather Q values for the latest tokens
        q_query = Q_flat[torch.arange(B * T, device=self.device), idx_query]
        
        # Reshape for decoder
        idx_query = idx_query.unsqueeze(1) # (BT, 1)
        q_query = q_query.unsqueeze(1)     # (BT, 1, N)
        
        # 4. Decoder Prediction
        logits = self.decoder(Q_active, slot_indices, pad_mask, q_query, idx_query)
        
        # 5. Loss
        y_flat = y.to(self.device).view(-1)
        # Apply label smoothing or pure CE
        loss = F.cross_entropy(logits, y_flat)
        
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
        V = self.config.vocab_size

        for x, y in dataloader:
            B, T = x.shape
            x_onehot = F.one_hot(x, num_classes=V).to(self.device, dtype=torch.float32)
            Q = compute_ema(x_onehot, self.rho)
            Q_flat = Q.view(B * T, V, self.config.N)
            Q_active, slot_indices, pad_mask = extract_active_slots(Q_flat, self.eps_q)
            
            idx_query = x.to(self.device).view(B * T)
            q_query = Q_flat[torch.arange(B * T, device=self.device), idx_query]
            idx_query = idx_query.unsqueeze(1)
            q_query = q_query.unsqueeze(1)
            
            logits = self.decoder(Q_active, slot_indices, pad_mask, q_query, idx_query)
            
            y_flat = y.to(self.device).view(-1)
            loss = F.cross_entropy(logits, y_flat, reduction='sum')
            
            total_loss += loss.item()
            total_tokens += y_flat.numel()
            
        return total_loss / max(total_tokens, 1)

    def fit(self, train_loader, val_loader=None, epochs: int = 5):
        print(f"Params: {sum(p.numel() for p in self.decoder.parameters()):,}")
        
        for epoch in range(epochs):
            t0 = time.time()
            train_loss = 0.0
            
            for step, (x, y) in enumerate(train_loader):
                loss = self.train_step(x, y)
                train_loss += loss
                
                if step % 20 == 0:
                    dt = (time.time() - t0) * 1000 / (step + 1)
                    print(f"Epoch {epoch} | Step {step} | Loss: {loss:.4f} | {dt:.1f}ms / batch")
                    
            train_loss /= len(train_loader)
            val_str = ""
            if val_loader:
                val_loss = self.evaluate(val_loader)
                val_ppl = math.exp(min(val_loss, 20.0))
                val_str = f"| Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
                
            print(f"=== Epoch {epoch} | Train Loss: {train_loss:.4f} {val_str} ===")
