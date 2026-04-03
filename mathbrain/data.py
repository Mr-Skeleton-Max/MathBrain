"""
Document-aware training data pipeline for Per-Slot EMA Transformer.

Key principles:
  1. Each sample = one document's contiguous slice (single-document guarantee)
  2. c_base carries full document history (infinite context)
  3. NO fixed seq_len padding — dynamic length per sample
  4. Token-budget batching: each batch fills a fixed token budget, not a fixed sample count
  5. Collation pads to batch-max length only (minimal waste)

For EMA: seq_len is meaningless. block_size only controls max chunk size for splitting
long documents. Short documents are never padded beyond batch-level alignment.

For RoPE: seq_len = context window. Use block_size to control it.
"""
import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, Sampler
import time
import math


class WikiTextMMapDataset(Dataset):
    def __init__(self, cache_dir, block_size, rhos, in_memory=True):
        self.block_size = block_size
        self.N = rhos.shape[0]
        self.rhos = rhos

        tokens_path = os.path.join(cache_dir, "tokens.bin")
        states_path = os.path.join(cache_dir, "states.bin")
        index_path = os.path.join(cache_dir, "pos_dict.pkl")
        doc_ids_path = os.path.join(cache_dir, "doc_ids.bin")

        self.tokens = np.memmap(tokens_path, dtype=np.int32, mode='r')
        self.doc_ids = np.memmap(doc_ids_path, dtype=np.int32, mode='r')

        if in_memory:
            self.states = np.fromfile(states_path, dtype=np.float32).reshape(len(self.tokens), self.N)
        else:
            self.states = np.memmap(states_path, dtype=np.float32, mode='r', shape=(len(self.tokens), self.N))

        with open(index_path, 'rb') as f:
            self.pos_dict = pickle.load(f)

        self.chunks = []  # [(flat_start, flat_end, doc_id)]
        self._build_index()

    def _build_index(self):
        total = len(self.tokens)
        if total == 0:
            return

        doc_ranges = []
        doc_start = 0
        current_doc = int(self.doc_ids[0])
        for i in range(1, total):
            d = int(self.doc_ids[i])
            if d != current_doc:
                doc_ranges.append((doc_start, i, current_doc))
                doc_start = i
                current_doc = d
        doc_ranges.append((doc_start, total, current_doc))

        n_docs = 0
        total_useful = 0
        for doc_start, doc_end, doc_id in doc_ranges:
            doc_len = doc_end - doc_start
            if doc_len < 2:
                continue
            n_docs += 1

            n_chunks = max(1, (doc_len - 1 + self.block_size - 1) // self.block_size)
            for c in range(n_chunks):
                cs = doc_start + c * self.block_size
                ce = min(cs + self.block_size + 1, doc_end)
                if ce - cs >= 2:
                    self.chunks.append((cs, ce, doc_id))
                    total_useful += ce - cs

        print(f"Dataset: {n_docs} docs, {len(self.chunks)} chunks, "
              f"block_size={self.block_size}, {total_useful} useful tokens")

    def __len__(self):
        return len(self.chunks)

    def chunk_len(self, idx):
        """Return actual token length of chunk (for token-budget batching)."""
        cs, ce, _ = self.chunks[idx]
        return ce - cs

    def __getitem__(self, idx):
        start, end, doc_id = self.chunks[idx]

        # Actual-length chunk — NO fixed padding
        chunk_np = np.array(self.tokens[start:end])
        doc_ids_np = np.array(self.doc_ids[start:end])

        chunk = torch.from_numpy(chunk_np).long()
        chunk_doc_ids = torch.from_numpy(doc_ids_np).long()

        unique_slots, inverse_indices = torch.unique(chunk, return_inverse=True)
        k = len(unique_slots)

        c_base = torch.zeros(k, self.N, dtype=torch.float32)
        for i, v_tensor in enumerate(unique_slots):
            v = v_tensor.item()
            if v not in self.pos_dict:
                continue
            history_positions = self.pos_dict[v]
            idx_end = np.searchsorted(history_positions, start)
            if idx_end > 0:
                last_pos = int(history_positions[idx_end - 1])
                if int(self.doc_ids[last_pos]) == doc_id:
                    last_state = torch.from_numpy(self.states[last_pos].copy())
                    d = start - last_pos
                    c_base[i] = last_state * (self.rhos ** d)

        return {
            'chunk': chunk,
            'doc_ids': chunk_doc_ids,
            'unique_slots': unique_slots,
            'inverse_indices': inverse_indices,
            'c_base': c_base,
        }


class TokenBudgetSampler(Sampler):
    """
    Groups samples into batches by total token budget instead of fixed batch size.
    Each batch has at most `token_budget` tokens (sum of actual chunk lengths).
    Optionally sorts by length to minimize padding within each batch.
    """
    def __init__(self, dataset, token_budget, shuffle=True, sort_within_batch=True):
        self.dataset = dataset
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.sort_within_batch = sort_within_batch

        # Precompute chunk lengths
        self.lengths = [dataset.chunk_len(i) for i in range(len(dataset))]
        self._build_batches()

    def _build_batches(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # Shuffle then greedily pack into batches
            import random
            random.shuffle(indices)

        self.batches = []
        current_batch = []
        current_tokens = 0
        max_len_in_batch = 0

        for idx in indices:
            chunk_len = self.lengths[idx]
            # Would adding this sample exceed the budget?
            new_max = max(max_len_in_batch, chunk_len)
            new_total = new_max * (len(current_batch) + 1)  # padded total

            if current_batch and new_total > self.token_budget:
                # Flush current batch
                if self.sort_within_batch:
                    current_batch.sort(key=lambda i: self.lengths[i])
                self.batches.append(current_batch)
                current_batch = [idx]
                current_tokens = chunk_len
                max_len_in_batch = chunk_len
            else:
                current_batch.append(idx)
                max_len_in_batch = new_max
                current_tokens = max_len_in_batch * len(current_batch)

        if current_batch:
            if self.sort_within_batch:
                current_batch.sort(key=lambda i: self.lengths[i])
            self.batches.append(current_batch)

    def __iter__(self):
        if self.shuffle:
            self._build_batches()  # re-shuffle each epoch
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def dynamic_collate_fn(batch):
    """Collate variable-length samples, padding to batch-max length."""
    B = len(batch)
    L = max(item['chunk'].shape[0] for item in batch)  # batch-max, not fixed block_size
    N = batch[0]['c_base'].shape[1]
    K_max = max(item['unique_slots'].shape[0] for item in batch)

    chunks = torch.zeros(B, L, dtype=torch.long)
    doc_ids = torch.full((B, L), -1, dtype=torch.long)  # -1 = padding
    unique_slots = torch.zeros(B, K_max, dtype=torch.long)
    pad_mask = torch.zeros(B, K_max, dtype=torch.bool)
    c_bases = torch.zeros(B, K_max, N, dtype=torch.float32)
    inverse_indices = torch.zeros(B, L, dtype=torch.long)

    for i, item in enumerate(batch):
        l = item['chunk'].shape[0]
        chunks[i, :l] = item['chunk']
        doc_ids[i, :l] = item['doc_ids']
        k = item['unique_slots'].shape[0]
        unique_slots[i, :k] = item['unique_slots']
        pad_mask[i, :k] = True
        c_bases[i, :k, :] = item['c_base']
        inverse_indices[i, :l] = item['inverse_indices']

    return chunks, unique_slots, pad_mask, inverse_indices, c_bases, K_max, doc_ids


# Keep old collate_fn name for backward compatibility
inverted_collate_fn = dynamic_collate_fn


# ==========================================
# PREPROCESSING (offline, one-time)
# ==========================================
def preprocess_corpus(documents: list, data_dir: str, N: int = 64, min_hl: float = 1.0, max_hl: float = 2048.0):
    """
    Tokenize and compute EMA states for all documents.
    Each document is processed independently with hard state reset at boundaries.
    """
    os.makedirs(data_dir, exist_ok=True)

    tokens_path = os.path.join(data_dir, "tokens.bin")
    doc_ids_path = os.path.join(data_dir, "doc_ids.bin")
    states_path = os.path.join(data_dir, "states.bin")
    index_path = os.path.join(data_dir, "pos_dict.pkl")

    tokens_list = []
    doc_ids_list = []
    for doc_idx, doc in enumerate(documents):
        tokens_list.extend(doc)
        doc_ids_list.extend([doc_idx] * len(doc))

    L_doc = len(tokens_list)
    tokens_array = np.array(tokens_list, dtype=np.int32)
    doc_ids_array = np.array(doc_ids_list, dtype=np.int32)

    tokens_array.tofile(tokens_path)
    doc_ids_array.tofile(doc_ids_path)

    log_base = np.log(2.0)
    scales = np.logspace(np.log10(min_hl), np.log10(max_hl), N)
    rhos = np.exp(-log_base / scales).astype(np.float32)

    states_out = np.memmap(states_path, dtype=np.float32, mode='w+', shape=(L_doc, N))

    V_max = np.max(tokens_array) + 1
    current_state = np.zeros((V_max, N), dtype=np.float32)
    last_pos = np.full((V_max,), -1, dtype=np.int32)

    import collections
    pos_dict = collections.defaultdict(list)

    print(f"Preprocessing {L_doc} tokens across {len(documents)} documents...")
    t0 = time.time()

    current_doc_id = -1
    for t in range(L_doc):
        d_id = doc_ids_array[t]

        if d_id != current_doc_id:
            current_state.fill(0.0)
            last_pos.fill(-1)
            current_doc_id = d_id

        v = tokens_array[t]
        pos_dict[v].append(t)

        prev_pos = last_pos[v]
        if prev_pos == -1:
            current_state[v] = 1.0
        else:
            d = t - prev_pos
            current_state[v] = current_state[v] * (rhos ** d) + 1.0

        last_pos[v] = t
        states_out[t] = current_state[v]

        if (t + 1) % 1000000 == 0:
            print(f"  {t+1} / {L_doc} tokens...")

    states_out.flush()

    for v in pos_dict.keys():
        pos_dict[v] = np.array(pos_dict[v], dtype=np.int32)

    with open(index_path, 'wb') as f:
        pickle.dump(dict(pos_dict), f)

    t1 = time.time()
    print(f"Preprocessing completed in {t1 - t0:.2f}s")
    print(f"  tokens: {tokens_path} ({os.path.getsize(tokens_path) / 1024**2:.2f} MB)")
    print(f"  states: {states_path} ({os.path.getsize(states_path) / 1024**3:.2f} GB)")
    print(f"  index:  {index_path} ({os.path.getsize(index_path) / 1024**2:.2f} MB)")
