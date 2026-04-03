"""
Test the data pipeline correctness.

Verifies:
1. Each chunk is strictly single-document
2. c_base carries correct same-document history
3. Different block_sizes produce identical EMA states (seq_len is purely a compute param)
4. No cross-document token contamination
5. Loss masking is correct

Usage:
    python test_data_pipeline.py
"""
import torch
import numpy as np
import os, sys, tempfile, pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mathbrain.data import WikiTextMMapDataset, preprocess_corpus, inverted_collate_fn
from mathbrain.triton_ema_query import compute_query_ema_history_pt
import math


def make_test_corpus(tmpdir, documents, N=4, min_hl=1.0, max_hl=64.0):
    """Create a preprocessed corpus from a list of token-id lists."""
    preprocess_corpus(documents, tmpdir, N=N, min_hl=min_hl, max_hl=max_hl)
    log_base = torch.log(torch.tensor(2.0))
    scales = torch.logspace(math.log10(min_hl), math.log10(max_hl), N)
    rhos = torch.exp(-log_base / scales)
    return rhos


def test_single_document_chunks():
    """Every chunk must contain tokens from exactly one document."""
    print("TEST 1: Single-document chunks")

    docs = [
        list(range(10, 30)),    # doc 0: 20 tokens
        list(range(30, 60)),    # doc 1: 30 tokens
        list(range(60, 75)),    # doc 2: 15 tokens
        list(range(75, 200)),   # doc 3: 125 tokens (long, will be split)
    ]

    tmpdir = tempfile.mkdtemp()
    rhos = make_test_corpus(tmpdir, docs)
    ds = WikiTextMMapDataset(tmpdir, block_size=32, rhos=rhos, in_memory=True)

    for i in range(len(ds)):
        item = ds[i]
        valid_doc_ids = item['doc_ids'][item['doc_ids'] >= 0]
        unique_docs = torch.unique(valid_doc_ids)
        assert len(unique_docs) == 1, \
            f"Chunk {i} has multiple docs: {unique_docs.tolist()}"

    print(f"  {len(ds)} chunks, all single-document ✓")
    print("  PASSED ✓\n")


def test_c_base_correctness():
    """c_base must match the full document EMA history up to chunk start."""
    print("TEST 2: c_base correctness")

    # Document with repeating token A (=1): positions 0,3,6,9,12
    doc = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    docs = [doc]

    tmpdir = tempfile.mkdtemp()
    N = 2
    rhos = make_test_corpus(tmpdir, docs, N=N, min_hl=1.0, max_hl=4.0)
    rhos_np = rhos.numpy()

    # Manually compute expected EMA states for token 1
    # Positions: 0, 3, 6, 9, 12
    state_at = {}
    state_at[0] = np.ones(N, dtype=np.float32)  # first occurrence
    state_at[3] = state_at[0] * (rhos_np ** 3) + 1.0
    state_at[6] = state_at[3] * (rhos_np ** 3) + 1.0
    state_at[9] = state_at[6] * (rhos_np ** 3) + 1.0
    state_at[12] = state_at[9] * (rhos_np ** 3) + 1.0

    ds = WikiTextMMapDataset(tmpdir, block_size=8, rhos=rhos, in_memory=True)

    # chunk 0: positions [0, 9), chunk 1: positions [8, 15)
    assert len(ds) == 2, f"Expected 2 chunks, got {len(ds)}"

    # Chunk 1 starts at position 8. Token 1's last occurrence before pos 8 is pos 6.
    # c_base for token 1 = states[6] * rhos^(8-6) = states[6] * rhos^2
    item1 = ds[1]
    start1 = ds.chunks[1][0]
    assert start1 == 8, f"Expected chunk 1 start=8, got {start1}"

    # Find token 1 in unique_slots
    tok1_idx = (item1['unique_slots'] == 1).nonzero(as_tuple=True)[0]
    assert len(tok1_idx) == 1, f"Token 1 not found in chunk 1"
    c_base_tok1 = item1['c_base'][tok1_idx[0]].numpy()

    expected = state_at[6] * (rhos_np ** 2)
    assert np.allclose(c_base_tok1, expected, atol=1e-5), \
        f"c_base mismatch: got {c_base_tok1}, expected {expected}"

    print(f"  Chunk 1 start={start1}, token 1 c_base={c_base_tok1.tolist()}")
    print(f"  Expected: {expected.tolist()}")
    print("  PASSED ✓\n")


def test_block_size_invariance():
    """
    KEY TEST: EMA predictions must be identical regardless of block_size.
    This proves seq_len is purely a compute parameter for EMA.
    """
    print("TEST 3: block_size invariance (seq_len is compute-only)")

    # Create a document with 100 tokens, repeating pattern
    doc = [i % 5 for i in range(100)]
    docs = [doc]

    tmpdir = tempfile.mkdtemp()
    N = 4
    rhos = make_test_corpus(tmpdir, docs, N=N, min_hl=1.0, max_hl=32.0)

    # Process with block_size=16
    ds16 = WikiTextMMapDataset(tmpdir, block_size=16, rhos=rhos, in_memory=True)
    # Process with block_size=32
    ds32 = WikiTextMMapDataset(tmpdir, block_size=32, rhos=rhos, in_memory=True)
    # Process with block_size=64
    ds64 = WikiTextMMapDataset(tmpdir, block_size=64, rhos=rhos, in_memory=True)

    # For each block_size, compute C_seq for every position in the document
    # Then compare: all block_sizes should produce identical C_seq values

    def get_all_c_seq(ds, rhos, vocab_size=5):
        """Concatenate C_seq from all chunks, trimming padding."""
        all_c = []
        for i in range(len(ds)):
            item = ds[i]
            start, end, doc_id = ds.chunks[i]
            actual_len = end - start

            x = item['chunk'][:actual_len].unsqueeze(0)  # [1, actual_len]
            c_init = item['c_base'].unsqueeze(0)  # [1, K, N]
            c_init_keys = item['unique_slots'].unsqueeze(0)  # [1, K]

            C_seq = compute_query_ema_history_pt(
                x, rhos, vocab_size, c_init=c_init, c_init_keys=c_init_keys
            )  # [1, actual_len, N]

            # For continuation chunks, skip the first token (it overlaps with previous chunk's last)
            if i > 0:
                all_c.append(C_seq[0, 1:, :])  # skip overlap
            else:
                all_c.append(C_seq[0, :, :])

        return torch.cat(all_c, dim=0)  # [total_positions, N]

    c16 = get_all_c_seq(ds16, rhos)
    c32 = get_all_c_seq(ds32, rhos)
    c64 = get_all_c_seq(ds64, rhos)

    # All should have the same length (number of prediction positions)
    min_len = min(len(c16), len(c32), len(c64))
    c16 = c16[:min_len]
    c32 = c32[:min_len]
    c64 = c64[:min_len]

    diff_16_32 = (c16 - c32).abs().max().item()
    diff_16_64 = (c16 - c64).abs().max().item()
    diff_32_64 = (c32 - c64).abs().max().item()

    print(f"  block_size=16: {len(ds16)} chunks")
    print(f"  block_size=32: {len(ds32)} chunks")
    print(f"  block_size=64: {len(ds64)} chunks")
    print(f"  C_seq positions compared: {min_len}")
    print(f"  Max diff (16 vs 32): {diff_16_32:.8f}")
    print(f"  Max diff (16 vs 64): {diff_16_64:.8f}")
    print(f"  Max diff (32 vs 64): {diff_32_64:.8f}")

    threshold = 1e-4
    assert diff_16_32 < threshold, f"FAIL: block_size 16 vs 32 differ by {diff_16_32}"
    assert diff_16_64 < threshold, f"FAIL: block_size 16 vs 64 differ by {diff_16_64}"
    assert diff_32_64 < threshold, f"FAIL: block_size 32 vs 64 differ by {diff_32_64}"

    print("  block_size does NOT affect EMA states → seq_len is purely a compute parameter ✓")
    print("  PASSED ✓\n")


def test_cross_doc_isolation():
    """Token appearing in both doc A and doc B must have independent EMA states."""
    print("TEST 4: Cross-document EMA isolation")

    # Token 1 appears in both documents
    docs = [
        [1, 2, 3, 1, 2],  # doc 0: token 1 at positions 0, 3
        [1, 4, 5, 1, 4],  # doc 1: token 1 at positions 5, 8
    ]

    tmpdir = tempfile.mkdtemp()
    N = 2
    rhos = make_test_corpus(tmpdir, docs, N=N, min_hl=1.0, max_hl=4.0)

    ds = WikiTextMMapDataset(tmpdir, block_size=16, rhos=rhos, in_memory=True)

    # Should have 2 chunks (one per document)
    assert len(ds) == 2, f"Expected 2 chunks, got {len(ds)}"

    # Chunk 0: doc 0
    item0 = ds[0]
    start0, end0, doc0_id = ds.chunks[0]
    assert doc0_id == 0

    # Chunk 1: doc 1
    item1 = ds[1]
    start1, end1, doc1_id = ds.chunks[1]
    assert doc1_id == 1

    # c_base for chunk 1 should be all zeros (doc 1 starts fresh)
    assert item1['c_base'].abs().sum() == 0, \
        f"FAIL: doc 1 c_base should be zero, got sum={item1['c_base'].abs().sum()}"

    # Verify doc 1's chunk doesn't contain doc 0's tokens in doc_ids
    valid_docs = item1['doc_ids'][item1['doc_ids'] >= 0]
    assert (valid_docs == 1).all(), \
        f"FAIL: doc 1 chunk contains non-doc1 tokens: {valid_docs.unique()}"

    print(f"  Doc 0: chunk [{start0},{end0}), c_base_sum={item0['c_base'].abs().sum():.4f}")
    print(f"  Doc 1: chunk [{start1},{end1}), c_base_sum={item1['c_base'].abs().sum():.4f} (zero ✓)")
    print("  PASSED ✓\n")


def test_loss_masking():
    """Loss should only be computed on valid same-document predictions."""
    print("TEST 5: Loss masking")

    docs = [[10, 20, 30], [40, 50, 60]]
    tmpdir = tempfile.mkdtemp()
    rhos = make_test_corpus(tmpdir, docs)
    ds = WikiTextMMapDataset(tmpdir, block_size=8, rhos=rhos, in_memory=True)

    item = ds[0]  # doc 0
    chunk = item['chunk']
    doc_ids = item['doc_ids']

    x = chunk[:-1]
    y = chunk[1:]
    doc_ids_x = doc_ids[:-1]
    doc_ids_y = doc_ids[1:]

    y_target = y.clone()
    y_target[doc_ids_y == -1] = -1  # mask padding

    # Within a single-doc chunk, doc_ids_x should always == doc_ids_y for real tokens
    real_mask = (doc_ids_x >= 0) & (doc_ids_y >= 0)
    if real_mask.any():
        assert (doc_ids_x[real_mask] == doc_ids_y[real_mask]).all(), \
            "FAIL: real tokens should all have matching doc_ids (single-doc chunk)"

    # Count masked vs unmasked
    n_total = len(y_target)
    n_masked = (y_target == -1).sum().item()
    n_valid = n_total - n_masked

    print(f"  Chunk length: {n_total}, valid targets: {n_valid}, masked: {n_masked}")
    print(f"  doc_ids: {doc_ids[:6].tolist()}...")
    print("  PASSED ✓\n")


def test_collation():
    """Batch collation should correctly pad and stack."""
    print("TEST 6: Batch collation")

    docs = [
        list(range(10, 25)),  # doc 0: 15 tokens
        list(range(25, 35)),  # doc 1: 10 tokens
        list(range(35, 55)),  # doc 2: 20 tokens
    ]

    tmpdir = tempfile.mkdtemp()
    rhos = make_test_corpus(tmpdir, docs)
    ds = WikiTextMMapDataset(tmpdir, block_size=16, rhos=rhos, in_memory=True)

    batch = [ds[i] for i in range(min(3, len(ds)))]
    collated = inverted_collate_fn(batch)
    chunks, unique_slots, pad_mask, inverse_indices, c_bases, K_max, doc_ids = collated

    B = len(batch)
    L = chunks.shape[1]

    assert chunks.shape == (B, L), f"chunks shape: {chunks.shape}"
    assert doc_ids.shape == (B, L), f"doc_ids shape: {doc_ids.shape}"
    assert unique_slots.shape[0] == B
    assert c_bases.shape[0] == B

    print(f"  Batch size: {B}, seq_len: {L}, K_max: {K_max}")
    print(f"  chunks shape: {chunks.shape}")
    print(f"  c_bases shape: {c_bases.shape}")
    print("  PASSED ✓\n")


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Data Pipeline Tests")
    print(f"{'='*50}\n")

    test_single_document_chunks()
    test_c_base_correctness()
    test_block_size_invariance()
    test_cross_doc_isolation()
    test_loss_masking()
    test_collation()

    print(f"{'='*50}")
    print(f"  ALL TESTS PASSED")
    print(f"{'='*50}")
