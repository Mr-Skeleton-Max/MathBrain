"""
Gold-standard numerical verification: single-pass full-document EMA vs chunked pipeline.

If these match at every position, the ENTIRE chain is proven correct:
  preprocessing → c_base → chunk splitting → history expansion → boundary precomputation

No model parameters involved — pure EMA state verification.
"""
import torch
import numpy as np
import tempfile
import os
import pickle
import collections
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mathbrain.data import WikiTextMMapDataset
from mathbrain.triton_ema_query import compute_query_ema_history_pt

PASS = 0
FAIL = 0
def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def build_corpus(tmpdir, tokens_list, doc_ids_list, N, min_hl, max_hl):
    tokens_array = np.array(tokens_list, dtype=np.int32)
    doc_ids_array = np.array(doc_ids_list, dtype=np.int32)
    tokens_array.tofile(os.path.join(tmpdir, 'tokens.bin'))
    doc_ids_array.tofile(os.path.join(tmpdir, 'doc_ids.bin'))

    scales = np.logspace(np.log10(min_hl), np.log10(max_hl), N)
    rhos_np = np.exp(-np.log(2.0) / scales).astype(np.float32)

    V_max = int(np.max(tokens_array)) + 1
    current_state = np.zeros((V_max, N), dtype=np.float32)
    last_pos = np.full(V_max, -1, dtype=np.int32)
    states = np.zeros((len(tokens_array), N), dtype=np.float32)
    pos_dict = collections.defaultdict(list)
    cur_doc = -1
    for t in range(len(tokens_array)):
        if doc_ids_array[t] != cur_doc:
            current_state.fill(0)
            last_pos.fill(-1)
            cur_doc = doc_ids_array[t]
        v = tokens_array[t]
        pos_dict[v].append(t)
        prev = last_pos[v]
        if prev == -1:
            current_state[v] = 1.0
        else:
            current_state[v] = current_state[v] * (rhos_np ** (t - prev)) + 1.0
        last_pos[v] = t
        states[t] = current_state[v]

    states.astype(np.float32).tofile(os.path.join(tmpdir, 'states.bin'))
    for v in pos_dict:
        pos_dict[v] = np.array(pos_dict[v], dtype=np.int32)
    with open(os.path.join(tmpdir, 'pos_dict.pkl'), 'wb') as f:
        pickle.dump(dict(pos_dict), f)

    return torch.from_numpy(rhos_np)


# ============================================================
# Build a realistic test document
# ============================================================
N = 16
MIN_HL, MAX_HL = 1.0, 512.0
VOCAB = 50

np.random.seed(42)
# Simulate burstiness: some tokens repeat frequently, some rarely
common_tokens = list(range(10, 20))  # 10 common tokens
rare_tokens = list(range(30, 50))    # 20 rare tokens
doc_tokens = []
for _ in range(200):
    if np.random.random() < 0.7:
        doc_tokens.append(np.random.choice(common_tokens))
    else:
        doc_tokens.append(np.random.choice(rare_tokens))

DOC_LEN = len(doc_tokens)
doc_ids = [0] * DOC_LEN

tmpdir = tempfile.mkdtemp()
rhos = build_corpus(tmpdir, doc_tokens, doc_ids, N, MIN_HL, MAX_HL)
vocab_size = VOCAB + 1

print(f"Test document: {DOC_LEN} tokens, {len(set(doc_tokens))} unique, N={N}, max_hl={MAX_HL}")


# ============================================================
# GROUND TRUTH: single-pass full-document C_seq
# ============================================================
print("\n=== Ground Truth: single-pass full-document EMA ===")

x_full = torch.tensor(doc_tokens, dtype=torch.long).unsqueeze(0)  # [1, DOC_LEN]
C_seq_truth = compute_query_ema_history_pt(x_full, rhos, vocab_size)  # [1, DOC_LEN, N]
print(f"  C_seq_truth shape: {C_seq_truth.shape}")


# ============================================================
# Test A: Query-side C_seq matches across ALL chunk sizes
# ============================================================
for block_size in [8, 16, 32, 64, 128]:
    print(f"\n=== Test A: Query C_seq with block_size={block_size} ===")

    tmpdir_bs = tempfile.mkdtemp()
    rhos_bs = build_corpus(tmpdir_bs, doc_tokens, doc_ids, N, MIN_HL, MAX_HL)
    ds = WikiTextMMapDataset(tmpdir_bs, block_size=block_size, rhos=rhos_bs, in_memory=True)

    # Reconstruct C_seq from chunks
    C_seq_chunked = torch.zeros(1, DOC_LEN, N)
    positions_covered = set()

    for idx in range(len(ds)):
        item = ds[idx]
        start, end, _ = ds.chunks[idx]
        chunk = item['chunk'].unsqueeze(0)  # [1, chunk_len]
        us = item['unique_slots'].unsqueeze(0)
        cb = item['c_base'].unsqueeze(0)

        C_seq_chunk = compute_query_ema_history_pt(chunk, rhos, vocab_size, c_init=cb, c_init_keys=us)

        # Map back to full document positions
        chunk_len = end - start
        for t in range(chunk_len):
            global_pos = start + t
            if global_pos < DOC_LEN:
                C_seq_chunked[0, global_pos, :] = C_seq_chunk[0, t, :]
                positions_covered.add(global_pos)

    check(f"bs={block_size}: all positions covered",
          len(positions_covered) == DOC_LEN,
          f"covered {len(positions_covered)}/{DOC_LEN}")

    # Compare
    max_diff = (C_seq_truth - C_seq_chunked).abs().max().item()
    mean_diff = (C_seq_truth - C_seq_chunked).abs().mean().item()
    check(f"bs={block_size}: max diff < 1e-5",
          max_diff < 1e-5,
          f"max_diff={max_diff:.2e}")
    check(f"bs={block_size}: mean diff < 1e-7",
          mean_diff < 1e-7,
          f"mean_diff={mean_diff:.2e}")

    # Show worst position
    if max_diff > 1e-5:
        worst_pos = (C_seq_truth - C_seq_chunked).abs().max(dim=2)[0].argmax().item()
        print(f"    Worst position: {worst_pos}")
        print(f"    Truth:   {C_seq_truth[0, worst_pos, :4].numpy()}")
        print(f"    Chunked: {C_seq_chunked[0, worst_pos, :4].numpy()}")


# ============================================================
# Test B: Memory-side EMA state for expanded vocabulary
# ============================================================
print(f"\n=== Test B: Memory-side EMA with history expansion (block_size=32) ===")

ds_b = WikiTextMMapDataset(tmpdir, block_size=32, rhos=rhos, in_memory=True)

# Ground truth: full-document EMA state at every position, for every vocab token
# C_mem_truth[t, k, :] = EMA state of token k at time t (BEFORE update at t)
C_mem_truth = torch.zeros(DOC_LEN, vocab_size, N)
C_running = torch.zeros(vocab_size, N)
rhos_vec = rhos.unsqueeze(0)  # [1, N]

for t in range(DOC_LEN):
    k = doc_tokens[t]
    # Variant C: decay ALL slots first
    C_running = C_running * rhos_vec
    # Store state BEFORE update (this is what Memory sees)
    C_mem_truth[t] = C_running.clone()
    # Update after attention
    C_running[k] += 1.0

# For each chunk, verify that expanded unique_slots + c_base produce correct Memory states
for idx in range(len(ds_b)):
    item = ds_b[idx]
    start, end, _ = ds_b.chunks[idx]
    chunk_len = end - start

    unique_slots = item['unique_slots']
    c_base = item['c_base']

    n_slots = len(unique_slots)
    chunk_tokens = item['chunk'].tolist()
    # Key: c_base already includes the decay AT chunk_start.
    # So the correct order is: compare → update → decay (for next step).
    all_match = True
    max_chunk_diff = 0.0
    C_verify = torch.zeros(n_slots, N)
    for i in range(n_slots):
        C_verify[i] = c_base[i]

    for t_local in range(chunk_len):
        t_global = start + t_local
        if t_global >= DOC_LEN:
            break
        # 1. Compare: C_verify is what Memory sees at t_global (after decay, before update)
        for i, v in enumerate(unique_slots.tolist()):
            d = (C_mem_truth[t_global, v] - C_verify[i]).abs().max().item()
            max_chunk_diff = max(max_chunk_diff, d)
            if d > 1e-4:
                all_match = False
        # 2. Update: current token gets +1 (Variant C: after attention)
        cur_tok = chunk_tokens[t_local]
        for i, v in enumerate(unique_slots.tolist()):
            if v == cur_tok:
                C_verify[i] += 1.0
                break
        # 3. Decay for next step
        C_verify = C_verify * rhos_vec

    check(f"Chunk {idx} (pos {start}-{end}): Memory EMA max_diff={max_chunk_diff:.2e}",
          all_match)


# ============================================================
# Test C: History-only tokens have correct Memory states
# ============================================================
print(f"\n=== Test C: History-only tokens have correct Memory EMA ===")

for idx in range(len(ds_b)):
    item = ds_b[idx]
    start, end, _ = ds_b.chunks[idx]
    chunk_set = set(item['chunk'].tolist())
    unique_set = set(item['unique_slots'].tolist())
    history_only = unique_set - chunk_set

    if not history_only:
        continue

    c_base = item['c_base']
    unique_list = item['unique_slots'].tolist()

    # c_base already includes the decay AT chunk_start, so compare directly
    all_ok = True
    worst_diff = 0.0
    for v in history_only:
        i = unique_list.index(v)
        cb = c_base[i]
        truth_at_start = C_mem_truth[start, v]
        diff = (cb - truth_at_start).abs().max().item()
        worst_diff = max(worst_diff, diff)
        if diff > 1e-4:
            all_ok = False

    if history_only:
        check(f"Chunk {idx}: {len(history_only)} history-only tokens match truth (worst={worst_diff:.2e})",
              all_ok)


# ============================================================
# Test D: Block-size invariance (C_seq identical across block sizes)
# ============================================================
print(f"\n=== Test D: Block-size invariance ===")

# Already computed C_seq for block_size=8,16,32,64,128 in Test A
# Here verify they're all identical to each other
ref_bs = 8
tmpdir_ref = tempfile.mkdtemp()
rhos_ref = build_corpus(tmpdir_ref, doc_tokens, doc_ids, N, MIN_HL, MAX_HL)
ds_ref = WikiTextMMapDataset(tmpdir_ref, block_size=ref_bs, rhos=rhos_ref, in_memory=True)

C_ref = torch.zeros(1, DOC_LEN, N)
for idx in range(len(ds_ref)):
    item = ds_ref[idx]
    start, end, _ = ds_ref.chunks[idx]
    chunk = item['chunk'].unsqueeze(0)
    us = item['unique_slots'].unsqueeze(0)
    cb = item['c_base'].unsqueeze(0)
    C_chunk = compute_query_ema_history_pt(chunk, rhos, vocab_size, c_init=cb, c_init_keys=us)
    for t in range(end - start):
        if start + t < DOC_LEN:
            C_ref[0, start + t] = C_chunk[0, t]

for other_bs in [32, 128]:
    tmpdir_o = tempfile.mkdtemp()
    rhos_o = build_corpus(tmpdir_o, doc_tokens, doc_ids, N, MIN_HL, MAX_HL)
    ds_o = WikiTextMMapDataset(tmpdir_o, block_size=other_bs, rhos=rhos_o, in_memory=True)

    C_other = torch.zeros(1, DOC_LEN, N)
    for idx in range(len(ds_o)):
        item = ds_o[idx]
        start, end, _ = ds_o.chunks[idx]
        chunk = item['chunk'].unsqueeze(0)
        us = item['unique_slots'].unsqueeze(0)
        cb = item['c_base'].unsqueeze(0)
        C_chunk = compute_query_ema_history_pt(chunk, rhos, vocab_size, c_init=cb, c_init_keys=us)
        for t in range(end - start):
            if start + t < DOC_LEN:
                C_other[0, start + t] = C_chunk[0, t]

    diff = (C_ref - C_other).abs().max().item()
    check(f"bs={ref_bs} vs bs={other_bs}: max diff={diff:.2e}",
          diff < 1e-5)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
if FAIL == 0:
    print("ALL GOLD-STANDARD CHECKS PASSED")
    print("\nThis proves:")
    print("  1. Query-side C_seq: chunked pipeline = single-pass ground truth")
    print("  2. Memory-side EMA: expanded unique_slots + c_base = full-document truth")
    print("  3. History-only tokens: c_base decay matches full-document computation")
    print("  4. Block-size invariance: seq_len is truly a pure compute parameter")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
