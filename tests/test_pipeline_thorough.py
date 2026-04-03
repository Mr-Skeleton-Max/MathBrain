"""
Thorough data pipeline verification.
Tests every invariant that must hold for correct EMA training.

Usage:
    python test_pipeline_thorough.py          # CPU
    python test_pipeline_thorough.py --cuda   # GPU (Triton path)
"""
import torch
import numpy as np
import os, sys, tempfile, pickle, argparse, math
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mathbrain.data import WikiTextMMapDataset, preprocess_corpus, dynamic_collate_fn, TokenBudgetSampler
from mathbrain.triton_ema_query import compute_query_ema_history_pt, compute_query_ema_history
from torch.utils.data import DataLoader

PASS = 0
FAIL = 0

def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        print(f"    ✗ FAIL: {msg}")


def make_corpus(docs, tmpdir, N=4, min_hl=1.0, max_hl=64.0):
    preprocess_corpus(docs, tmpdir, N=N, min_hl=min_hl, max_hl=max_hl)
    log_base = torch.log(torch.tensor(2.0))
    scales = torch.logspace(math.log10(min_hl), math.log10(max_hl), N)
    rhos = torch.exp(-log_base / scales)
    return rhos


# ═══════════════════════════════════════════════════
# SECTION 1: PREPROCESSING
# ═══════════════════════════════════════════════════
def test_preprocessing():
    print("\n[1] PREPROCESSING")

    docs = [
        [10, 20, 30, 10, 20],  # doc 0: token 10 at pos 0,3; token 20 at pos 1,4
        [10, 40, 50],           # doc 1: token 10 at pos 5 (different doc!)
    ]
    tmpdir = tempfile.mkdtemp()
    N = 2
    rhos = make_corpus(docs, tmpdir, N=N, min_hl=1.0, max_hl=4.0)
    rhos_np = rhos.numpy()

    tokens = np.fromfile(os.path.join(tmpdir, "tokens.bin"), dtype=np.int32)
    doc_ids = np.fromfile(os.path.join(tmpdir, "doc_ids.bin"), dtype=np.int32)
    states = np.fromfile(os.path.join(tmpdir, "states.bin"), dtype=np.float32).reshape(-1, N)

    # 1a. Flat arrays correct
    check(list(tokens) == [10,20,30,10,20, 10,40,50], "tokens flat array")
    check(list(doc_ids) == [0,0,0,0,0, 1,1,1], "doc_ids flat array")

    # 1b. States reset at doc boundary
    # Doc 0, pos 0: token 10 first seen → state = 1.0
    check(np.allclose(states[0], [1.0, 1.0]), f"states[0] = {states[0]} (expected [1,1])")

    # Doc 0, pos 3: token 10 again, d=3 → state = 1.0 * rhos^3 + 1.0
    expected_3 = 1.0 * rhos_np**3 + 1.0
    check(np.allclose(states[3], expected_3, atol=1e-5), f"states[3] = {states[3]} (expected {expected_3})")

    # Doc 1, pos 5: token 10 first in doc 1 → state = 1.0 (NOT carrying doc 0 history)
    check(np.allclose(states[5], [1.0, 1.0]), f"states[5] = {states[5]} (expected [1,1], doc boundary reset)")

    # 1c. pos_dict spans all docs (global positions)
    with open(os.path.join(tmpdir, "pos_dict.pkl"), 'rb') as f:
        pos_dict = pickle.load(f)
    check(list(pos_dict[10]) == [0, 3, 5], f"pos_dict[10] = {list(pos_dict[10])} (expected [0,3,5])")

    print(f"  {PASS} checks passed")


# ═══════════════════════════════════════════════════
# SECTION 2: CHUNK INDEX
# ═══════════════════════════════════════════════════
def test_chunk_index():
    print("\n[2] CHUNK INDEX")

    docs = [
        list(range(100)),   # doc 0: 100 tokens
        list(range(50)),    # doc 1: 50 tokens
        list(range(200)),   # doc 2: 200 tokens
        [1],                # doc 3: 1 token (should be skipped, < 2)
    ]
    tmpdir = tempfile.mkdtemp()
    rhos = make_corpus(docs, tmpdir)
    ds = WikiTextMMapDataset(tmpdir, block_size=64, rhos=rhos, in_memory=True)

    # 2a. Doc 3 (1 token) should be skipped
    all_doc_ids = set()
    for cs, ce, doc_id in ds.chunks:
        all_doc_ids.add(doc_id)
    check(3 not in all_doc_ids, f"doc 3 (1 token) should be skipped, got doc_ids={all_doc_ids}")

    # 2b. Every chunk is single-document
    for i, (cs, ce, doc_id) in enumerate(ds.chunks):
        chunk_doc_ids = ds.doc_ids[cs:ce]
        unique_docs = set(int(d) for d in chunk_doc_ids)
        check(len(unique_docs) == 1, f"chunk {i} spans docs {unique_docs}")
        check(doc_id in unique_docs, f"chunk {i} doc_id={doc_id} not in {unique_docs}")

    # 2c. All tokens covered (no gaps within documents)
    doc_coverage = defaultdict(set)
    for cs, ce, doc_id in ds.chunks:
        for pos in range(cs, ce):
            doc_coverage[doc_id].add(pos)

    # Doc 0: 100 tokens starting at flat pos 0
    check(len(doc_coverage[0]) == 100, f"doc 0 coverage: {len(doc_coverage[0])} (expected 100)")

    # 2d. Chunks don't exceed block_size + 1
    for i, (cs, ce, doc_id) in enumerate(ds.chunks):
        check(ce - cs <= 65, f"chunk {i} length {ce-cs} > block_size+1=65")

    print(f"  {PASS} checks passed")


# ═══════════════════════════════════════════════════
# SECTION 3: c_base LOADING
# ═══════════════════════════════════════════════════
def test_c_base():
    print("\n[3] c_base LOADING")

    # Two docs, both containing token 1
    docs = [
        [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],  # doc 0: 10 tokens, token 1 at pos 0,3,6,9
        [1, 4, 5, 1, 4, 5],               # doc 1: 6 tokens, token 1 at pos 10,13
    ]
    tmpdir = tempfile.mkdtemp()
    N = 2
    rhos = make_corpus(docs, tmpdir, N=N, min_hl=1.0, max_hl=4.0)
    rhos_np = rhos.numpy()

    ds = WikiTextMMapDataset(tmpdir, block_size=4, rhos=rhos, in_memory=True)

    # 3a. First chunk of doc 0: c_base should be all zeros
    item0 = ds[0]
    cs0, ce0, did0 = ds.chunks[0]
    check(did0 == 0, f"chunk 0 doc_id={did0}")
    check(cs0 == 0, f"chunk 0 starts at {cs0} (expected 0)")
    check(item0['c_base'].abs().sum() == 0, f"chunk 0 c_base sum={item0['c_base'].abs().sum()} (expected 0)")

    # 3b. First chunk of doc 1: c_base should be all zeros (doc boundary reset!)
    doc1_chunks = [(i, cs, ce, d) for i, (cs, ce, d) in enumerate(ds.chunks) if d == 1]
    check(len(doc1_chunks) > 0, "doc 1 should have chunks")
    if doc1_chunks:
        idx1, cs1, ce1, did1 = doc1_chunks[0]
        item1 = ds[idx1]
        check(item1['c_base'].abs().sum() == 0,
              f"doc 1 first chunk c_base sum={item1['c_base'].abs().sum()} (expected 0, no cross-doc history)")

    # 3c. Continuation chunk of doc 0: c_base should be non-zero and correct
    doc0_chunks = [(i, cs, ce, d) for i, (cs, ce, d) in enumerate(ds.chunks) if d == 0]
    if len(doc0_chunks) >= 2:
        idx_cont, cs_cont, ce_cont, _ = doc0_chunks[1]
        item_cont = ds[idx_cont]

        # Find token 1 in unique_slots
        tok1_mask = (item_cont['unique_slots'] == 1)
        if tok1_mask.any():
            tok1_idx = tok1_mask.nonzero(as_tuple=True)[0][0]
            c_base_tok1 = item_cont['c_base'][tok1_idx].numpy()

            # Token 1 last occurred before cs_cont in doc 0
            # Find the exact position
            pos_dict_tok1 = [0, 3, 6, 9]  # doc 0 positions of token 1
            last_before = max(p for p in pos_dict_tok1 if p < cs_cont)

            # Expected: states[last_before] * rhos^(cs_cont - last_before)
            states = ds.states
            expected = states[last_before] * (rhos_np ** (cs_cont - last_before))
            check(np.allclose(c_base_tok1, expected, atol=1e-4),
                  f"c_base mismatch: got {c_base_tok1}, expected {expected}")
        else:
            check(False, "token 1 not found in continuation chunk")

    print(f"  {PASS} checks passed")


# ═══════════════════════════════════════════════════
# SECTION 4: EMA QUERY SCAN
# ═══════════════════════════════════════════════════
def test_ema_scan(device='cpu'):
    print(f"\n[4] EMA QUERY SCAN (device={device})")

    docs = [[1, 2, 1, 3, 1, 2, 1]]  # one doc
    tmpdir = tempfile.mkdtemp()
    N = 4
    rhos = make_corpus(docs, tmpdir, N=N, min_hl=1.0, max_hl=16.0)

    ds = WikiTextMMapDataset(tmpdir, block_size=4, rhos=rhos, in_memory=True)

    # Manually compute expected EMA for full document
    rhos_np = rhos.numpy()
    V = 4
    manual_C = np.zeros((7, N), dtype=np.float32)  # C_seq for each position
    state = np.zeros((V, N), dtype=np.float32)
    last_t = np.full(V, -1000000, dtype=np.int32)

    tokens = [1, 2, 1, 3, 1, 2, 1]
    for t in range(7):
        v = tokens[t]
        if last_t[v] >= 0:
            d = t - last_t[v]
            state[v] = state[v] * (rhos_np ** d)
        else:
            state[v] = np.zeros(N)
        manual_C[t] = state[v].copy()
        state[v] += 1.0
        last_t[v] = t

    # Now compute via chunks with c_base
    all_C = []
    for i in range(len(ds)):
        item = ds[i]
        cs, ce, doc_id = ds.chunks[i]
        actual_len = ce - cs

        x = item['chunk'][:actual_len].unsqueeze(0).to(device)
        c_init = item['c_base'].unsqueeze(0).to(device)
        c_init_keys = item['unique_slots'].unsqueeze(0).to(device)

        if device == 'cpu':
            C_seq = compute_query_ema_history_pt(x, rhos.to(device), V, c_init=c_init, c_init_keys=c_init_keys)
        else:
            C_seq = compute_query_ema_history(x, rhos.to(device), V, c_init=c_init, c_init_keys=c_init_keys)

        if i > 0:
            all_C.append(C_seq[0, 1:].cpu())  # skip overlap
        else:
            all_C.append(C_seq[0].cpu())

    reconstructed = torch.cat(all_C, dim=0).numpy()

    # Compare
    min_len = min(len(manual_C), len(reconstructed))
    max_diff = np.abs(manual_C[:min_len] - reconstructed[:min_len]).max()
    check(max_diff < 1e-4, f"EMA scan max diff = {max_diff} (expected < 1e-4)")

    # 4b. Different block_sizes should give same result
    for bs in [2, 3, 7]:
        ds2 = WikiTextMMapDataset(tmpdir, block_size=bs, rhos=rhos, in_memory=True)
        all_C2 = []
        for i in range(len(ds2)):
            item = ds2[i]
            cs, ce, doc_id = ds2.chunks[i]
            actual_len = ce - cs
            x = item['chunk'][:actual_len].unsqueeze(0).to(device)
            c_init = item['c_base'].unsqueeze(0).to(device)
            c_init_keys = item['unique_slots'].unsqueeze(0).to(device)
            if device == 'cpu':
                C_seq = compute_query_ema_history_pt(x, rhos.to(device), V, c_init=c_init, c_init_keys=c_init_keys)
            else:
                C_seq = compute_query_ema_history(x, rhos.to(device), V, c_init=c_init, c_init_keys=c_init_keys)
            if i > 0:
                all_C2.append(C_seq[0, 1:].cpu())
            else:
                all_C2.append(C_seq[0].cpu())
        reconstructed2 = torch.cat(all_C2, dim=0).numpy()
        diff = np.abs(manual_C[:min(len(manual_C), len(reconstructed2))] - reconstructed2[:min(len(manual_C), len(reconstructed2))]).max()
        check(diff < 1e-4, f"block_size={bs} max diff = {diff}")

    print(f"  {PASS} checks passed")


# ═══════════════════════════════════════════════════
# SECTION 5: COLLATION
# ═══════════════════════════════════════════════════
def test_collation():
    print("\n[5] COLLATION")

    docs = [
        list(range(30)),   # doc 0: 30 tokens
        list(range(10)),   # doc 1: 10 tokens
        list(range(50)),   # doc 2: 50 tokens
    ]
    tmpdir = tempfile.mkdtemp()
    rhos = make_corpus(docs, tmpdir)
    ds = WikiTextMMapDataset(tmpdir, block_size=20, rhos=rhos, in_memory=True)

    batch = [ds[i] for i in range(min(3, len(ds)))]
    collated = dynamic_collate_fn(batch)
    chunks, unique_slots, pad_mask, inverse_indices, c_bases, K_max, doc_ids = collated

    B = len(batch)
    L = chunks.shape[1]

    # 5a. Shapes
    check(chunks.shape[0] == B, f"batch dim {chunks.shape[0]} != {B}")
    check(doc_ids.shape == chunks.shape, f"doc_ids shape {doc_ids.shape} != chunks shape {chunks.shape}")

    # 5b. Padding positions have doc_id = -1
    for i in range(B):
        actual_len = batch[i]['chunk'].shape[0]
        if actual_len < L:
            check((doc_ids[i, actual_len:] == -1).all(),
                  f"batch {i}: padding doc_ids should be -1")

    # 5c. Real positions have correct doc_ids
    for i in range(B):
        actual_len = batch[i]['chunk'].shape[0]
        real_docs = doc_ids[i, :actual_len]
        unique_docs = real_docs[real_docs >= 0].unique()
        check(len(unique_docs) <= 1, f"batch {i}: multiple docs in real region: {unique_docs}")

    print(f"  {PASS} checks passed")


# ═══════════════════════════════════════════════════
# SECTION 6: LOSS MASKING
# ═══════════════════════════════════════════════════
def test_loss_masking():
    print("\n[6] LOSS MASKING")

    docs = [[10, 20, 30]]
    tmpdir = tempfile.mkdtemp()
    rhos = make_corpus(docs, tmpdir)
    ds = WikiTextMMapDataset(tmpdir, block_size=8, rhos=rhos, in_memory=True)
    item = ds[0]

    chunk = item['chunk']
    doc_ids = item['doc_ids']

    # Simulate train.py logic
    x = chunk[:-1]
    y = chunk[1:]
    doc_ids_x = doc_ids[:-1]
    doc_ids_y = doc_ids[1:]

    y_target = y.clone()
    y_target[doc_ids_y == -1] = -1
    y_target[doc_ids_x != doc_ids_y] = -1

    # 6a. Real predictions are NOT masked
    n_real = (doc_ids[1:] >= 0).sum().item()  # real target positions
    # Subtract cross-doc boundaries
    n_cross = ((doc_ids_x >= 0) & (doc_ids_y >= 0) & (doc_ids_x != doc_ids_y)).sum().item()
    n_valid = (y_target != -1).sum().item()
    check(n_valid == n_real - n_cross, f"valid={n_valid}, expected={n_real - n_cross}")

    # 6b. Padding predictions are masked
    pad_positions = (doc_ids_y == -1)
    check((y_target[pad_positions] == -1).all(), "padding targets should be -1")

    # 6c. For single-doc chunk, no cross-doc masking
    check(n_cross == 0, f"single-doc chunk should have 0 cross-doc, got {n_cross}")

    print(f"  {PASS} checks passed")


# ═══════════════════════════════════════════════════
# SECTION 7: TOKEN BUDGET SAMPLER
# ═══════════════════════════════════════════════════
def test_token_budget():
    print("\n[7] TOKEN BUDGET SAMPLER")

    docs = [list(range(i*10, i*10 + 50 + i*20)) for i in range(10)]
    tmpdir = tempfile.mkdtemp()
    rhos = make_corpus(docs, tmpdir)
    ds = WikiTextMMapDataset(tmpdir, block_size=128, rhos=rhos, in_memory=True)

    sampler = TokenBudgetSampler(ds, token_budget=512, shuffle=False)

    # 7a. All indices covered exactly once
    all_indices = []
    for batch in sampler:
        all_indices.extend(batch)
    check(sorted(all_indices) == list(range(len(ds))),
          f"sampler covers {len(all_indices)} indices, expected {len(ds)}")

    # 7b. No batch exceeds budget (padded estimate)
    for batch_indices in sampler:
        if not batch_indices:
            continue
        lengths = [ds.chunk_len(i) for i in batch_indices]
        padded = max(lengths) * len(lengths)
        check(padded <= 512 or len(batch_indices) == 1,
              f"batch padded={padded} > budget=512, B={len(batch_indices)}")

    print(f"  {PASS} checks passed")


# ═══════════════════════════════════════════════════
# SECTION 8: MULTI-DOC ISOLATION END-TO-END
# ═══════════════════════════════════════════════════
def test_multi_doc_e2e(device='cpu'):
    print(f"\n[8] MULTI-DOC END-TO-END ISOLATION (device={device})")

    # Two docs with same tokens but different order
    docs = [
        [1, 2, 3, 1, 2],  # doc 0
        [2, 1, 3, 2, 1],  # doc 1 (same tokens, different order)
    ]
    tmpdir = tempfile.mkdtemp()
    N = 4
    rhos = make_corpus(docs, tmpdir, N=N, min_hl=1.0, max_hl=16.0)

    ds = WikiTextMMapDataset(tmpdir, block_size=16, rhos=rhos, in_memory=True)

    # Get EMA states for both docs
    V = 4
    results = {}
    for i in range(len(ds)):
        item = ds[i]
        cs, ce, doc_id = ds.chunks[i]
        actual_len = ce - cs
        x = item['chunk'][:actual_len].unsqueeze(0).to(device)
        c_init = item['c_base'].unsqueeze(0).to(device)
        c_init_keys = item['unique_slots'].unsqueeze(0).to(device)

        if device == 'cpu':
            C_seq = compute_query_ema_history_pt(x, rhos.to(device), V, c_init=c_init, c_init_keys=c_init_keys)
        else:
            C_seq = compute_query_ema_history(x, rhos.to(device), V, c_init=c_init, c_init_keys=c_init_keys)
        results[doc_id] = C_seq[0].cpu()

    # 8a. Doc 0 pos 3: token=1, seen before at pos 0 → C > 0
    #     Doc 1 pos 3: token=2, seen before at pos 0 → C > 0
    #     Both have non-zero C at pos 3, but for different tokens → independent histories
    # Verify: doc 0 pos 3 (token 1, second occurrence) has same C as
    #         doc 1 pos 3 (token 2, second occurrence) — both are "seen 3 steps ago"
    # This confirms EMA tracks per-token timing, not global position
    check(torch.allclose(results[0][3], results[1][3], atol=1e-4),
          "same recency pattern should produce same EMA value regardless of which token")
    # But pos 1: doc0 token=2 (first seen), doc1 token=1 (first seen) → both zero
    check(torch.allclose(results[0][1], torch.zeros_like(results[0][1]), atol=1e-4),
          "doc 0 pos 1 (token 2 first seen) should have C=0")
    check(torch.allclose(results[1][1], torch.zeros_like(results[1][1]), atol=1e-4),
          "doc 1 pos 1 (token 1 first seen) should have C=0")

    # 8b. Compute doc 0 in two chunks (block_size=3) and verify same result
    ds_small = WikiTextMMapDataset(tmpdir, block_size=3, rhos=rhos, in_memory=True)
    doc0_chunks = [(i, cs, ce, d) for i, (cs, ce, d) in enumerate(ds_small.chunks) if d == 0]

    all_C = []
    for idx, cs, ce, doc_id in doc0_chunks:
        item = ds_small[idx]
        actual_len = ce - cs
        x = item['chunk'][:actual_len].unsqueeze(0).to(device)
        c_init = item['c_base'].unsqueeze(0).to(device)
        c_init_keys = item['unique_slots'].unsqueeze(0).to(device)
        if device == 'cpu':
            C_seq = compute_query_ema_history_pt(x, rhos.to(device), V, c_init=c_init, c_init_keys=c_init_keys)
        else:
            C_seq = compute_query_ema_history(x, rhos.to(device), V, c_init=c_init, c_init_keys=c_init_keys)
        if all_C:
            all_C.append(C_seq[0, 1:].cpu())
        else:
            all_C.append(C_seq[0].cpu())

    reconstructed = torch.cat(all_C, dim=0)
    diff = (results[0][:len(reconstructed)] - reconstructed).abs().max().item()
    check(diff < 1e-4, f"multi-chunk reconstruction diff={diff}")

    print(f"  {PASS} checks passed")


# ═══════════════════════════════════════════════════
# SECTION 9: DATALOADER INTEGRATION
# ═══════════════════════════════════════════════════
def test_dataloader_integration():
    print("\n[9] DATALOADER INTEGRATION")

    docs = [list(range(i, i + 80)) for i in range(0, 500, 80)]  # 6 docs, 80 tokens each
    tmpdir = tempfile.mkdtemp()
    rhos = make_corpus(docs, tmpdir)
    ds = WikiTextMMapDataset(tmpdir, block_size=32, rhos=rhos, in_memory=True)

    # Test with fixed batch_size
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=dynamic_collate_fn)

    total_valid = 0
    for batch in loader:
        chunks, unique_slots, pad_mask, inverse_indices, c_bases, K_max, doc_ids = batch
        B, L = chunks.shape

        # Every real position should have same doc_id within each batch element
        for b in range(B):
            real = doc_ids[b][doc_ids[b] >= 0]
            if len(real) > 0:
                check(len(real.unique()) == 1, f"batch element {b} has multiple docs: {real.unique()}")

        # Count valid targets
        x = chunks[:, :-1]
        y = chunks[:, 1:]
        dx = doc_ids[:, :-1]
        dy = doc_ids[:, 1:]
        y_t = y.clone()
        y_t[dy == -1] = -1
        y_t[dx != dy] = -1
        total_valid += (y_t != -1).sum().item()

    # Should cover all tokens minus doc-start positions
    total_tokens = sum(len(d) for d in docs)
    # Each doc has (len-1) prediction targets
    expected_targets = sum(len(d) - 1 for d in docs)
    check(total_valid == expected_targets,
          f"total valid targets={total_valid}, expected={expected_targets}")

    print(f"  {PASS} checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*60}")
    print(f"  THOROUGH PIPELINE VERIFICATION (device={device})")
    print(f"{'='*60}")

    test_preprocessing()
    test_chunk_index()
    test_c_base()
    test_ema_scan(device)
    test_collation()
    test_loss_masking()
    test_token_budget()
    test_multi_doc_e2e(device)
    test_dataloader_integration()

    print(f"\n{'='*60}")
    if FAIL == 0:
        print(f"  ALL {PASS} CHECKS PASSED ✓")
    else:
        print(f"  {FAIL} CHECKS FAILED, {PASS} PASSED")
    print(f"{'='*60}\n")
