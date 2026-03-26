import torch
import numpy as np
from typing import Iterator, Tuple
from .retina import BPERetina, IdentityRetina
from .config import MathBrainConfig

class StreamingEMADataset:
    """
    An iterable dataset that re-reads and yields streaming chunks.
    This allows `for batch in dataloader:` to work multiple times across epochs.
    """
    def __init__(self, corpus_path: str, config: MathBrainConfig, batch_size: int, seq_len: int, device: torch.device, bpe_model_path: str = None):
        self.corpus_path = corpus_path
        self.config = config
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.bpe_model_path = bpe_model_path
        
        # We can tokenize once and keep in RAM (it's small enough, a few 100MB max)
        print(f"Loading corpus from {corpus_path} ...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if config.retina_mode == 'bpe':
            assert bpe_model_path is not None, "BPE model path required for bpe retina mode"
            retina = BPERetina(bpe_model_path)
        else:
            retina = IdentityRetina()
            
        print("Tokenizing corpus...")
        tokens = retina.encode(text)
        
        if len(tokens) == 0:
            print("Warning: Empty corpus!")
            self.tokens_tensor = None
            self.stream_len = 0
            return

        self.tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        total_len = self.tokens_tensor.shape[0]
        print(f"Loaded {total_len} tokens. Formatting into {batch_size} continuous streams.")
        
        self.stream_len = (total_len - 1) // batch_size
        if self.stream_len < seq_len:
            print(f"Corpus too small for batch_size={batch_size} and seq_len={seq_len}. Yielding partial fallback.")
            
        # Trim multiple
        self.tokens_tensor = self.tokens_tensor[:batch_size * self.stream_len + 1]
        
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.tokens_tensor is None:
            return

        if self.stream_len < self.seq_len:
            inputs = self.tokens_tensor[:-1].unsqueeze(0)
            targets = self.tokens_tensor[1:].unsqueeze(0)
            yield inputs, targets
            return
            
        inputs = self.tokens_tensor[:-1].view(self.batch_size, self.stream_len)
        targets = self.tokens_tensor[1:].view(self.batch_size, self.stream_len)
        
        # Yield temporal chunks
        for start_idx in range(0, self.stream_len - self.seq_len + 1, self.seq_len):
            end_idx = start_idx + self.seq_len
            yield inputs[:, start_idx:end_idx], targets[:, start_idx:end_idx]
            
    def __len__(self):
        if self.stream_len < self.seq_len:
            return 1
        return (self.stream_len - self.seq_len + 1) // self.seq_len + (1 if (self.stream_len - self.seq_len + 1) % self.seq_len != 0 else 0)

def get_streaming_batches(
    corpus_path: str, 
    config: MathBrainConfig, 
    batch_size: int, 
    seq_len: int, 
    device: torch.device, 
    bpe_model_path: str = None
):
    return StreamingEMADataset(corpus_path, config, batch_size, seq_len, device, bpe_model_path)
