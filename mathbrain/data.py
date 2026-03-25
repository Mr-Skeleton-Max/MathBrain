import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

class PrecomputedEMADataset(Dataset):
    """
    Loads precomputed EMA states and target tokens.
    It yields sequences of length `seq_len` to parallelize Decoder training,
    but crucially, since Q_active is already precomputed across the whole corpus,
    these chunks carry INFINITE historical context and can be safely SHUFFLED.
    """
    def __init__(self, pt_path: str, seq_len: int = 256):
        self.seq_len = seq_len
        print(f"Loading precomputed dataset from {pt_path} ...")
        data = torch.load(pt_path, map_location='cpu')
        
        # Load tensors
        self.q_active = data['q_active']            # (Total, max_active, N)
        self.slot_indices = data['slot_indices']    # (Total, max_active)
        self.pad_mask = data['pad_mask']            # (Total, max_active)
        self.q_query = data['q_query']              # (Total, 1, N)
        self.idx_query = data['idx_query']          # (Total, 1)
        self.targets = data['targets']              # (Total,)
        
        self.total_tokens = self.targets.shape[0]
        self.n_seqs = self.total_tokens // seq_len
        
        # In rare cases of tiny datasets
        if self.n_seqs == 0 and self.total_tokens > 0:
            self.n_seqs = 1
            self.seq_len = self.total_tokens
        
        print(f"Loaded {self.total_tokens} tokens. Using seq_len={self.seq_len} -> {self.n_seqs} sequences.")

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # We return the precomputed features for this chunk.
        # Since they were precomputed WITHOUT chunk slicing, they retain full history!
        return {
            'q_active': self.q_active[start:end].to(torch.float32),
            'slot_indices': self.slot_indices[start:end].to(torch.long),
            'pad_mask': self.pad_mask[start:end],
            'q_query': self.q_query[start:end].to(torch.float32),
            'idx_query': self.idx_query[start:end].to(torch.long),
            'targets': self.targets[start:end].to(torch.long)
        }

def get_precomputed_dataloader(pt_path: str, batch_size: int, seq_len: int = 256, shuffle: bool = True):
    dataset = PrecomputedEMADataset(pt_path, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
