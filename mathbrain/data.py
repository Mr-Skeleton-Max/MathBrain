import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

class LMDataset(Dataset):
    """
    Language Modeling Dataset for MathBrain.
    Chunks the input token IDs into fixed-length sequences.
    """
    def __init__(self, token_ids: List[int], seq_len: int = 256):
        self.seq_len = seq_len
        # Drop the remainder that doesn't fit exactly into seq_len
        n_seqs = (len(token_ids) - 1) // seq_len
        self.n_seqs = n_seqs
        
        # Avoid zero length for tiny datasets
        if n_seqs == 0 and len(token_ids) > 1:
            self.n_seqs = 1
            self.seq_len = len(token_ids) - 1
            self.token_ids = token_ids
        else:
            self.token_ids = token_ids[:n_seqs * seq_len + 1]
            
    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        x = torch.tensor(self.token_ids[start:end], dtype=torch.long)
        y = torch.tensor(self.token_ids[start+1:end+1], dtype=torch.long)
        return x, y

def get_dataloader(token_ids: List[int], batch_size: int, seq_len: int = 256, shuffle: bool = True):
    dataset = LMDataset(token_ids, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
