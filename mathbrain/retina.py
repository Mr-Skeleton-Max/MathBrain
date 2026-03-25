import numpy as np
from typing import Dict, List
import sentencepiece as spm
from .config import MathBrainConfig

class Retina:
    """Base class for Retina mappings."""
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError


class BPERetina(Retina):
    """Maps BPE tokens 1:1 to Slots."""
    def __init__(self, model_path: str):
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
    
    def encode(self, text: str) -> List[int]:
        return self._sp.EncodeAsIds(text.lower())
        
    @property
    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()


class IdentityRetina(Retina):
    """Word-level identity mapping. 1 exact word = 1 slot."""
    def __init__(self):
        self._word_to_id: Dict[str, int] = {}
        self._id_to_word: Dict[int, str] = {}
        # Start at 1 (0 can be reserved for padding/unk if needed, but we don't pad EMA)
        self._next_id = 0

    def _tokenize(self, text: str) -> List[str]:
        # Minimal robust tokenization
        text = text.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,')
        return [w.strip() for w in text.split() if w.strip()]

    def encode(self, text: str) -> List[int]:
        words = self._tokenize(text)
        ids = []
        for w in words:
            if w not in self._word_to_id:
                self._word_to_id[w] = self._next_id
                self._id_to_word[self._next_id] = w
                self._next_id += 1
            ids.append(self._word_to_id[w])
        return ids

    @property
    def vocab_size(self) -> int:
        # Return current maximum known vocab size.
        return len(self._word_to_id)
