# Per-Slot EMA: Compress Time, Not Semantics

**A BPTT-free, O(1)-state sequence model that matches RoPE Transformers on standard benchmarks.**

---

## What is this?

Per-Slot EMA is a new approach to sequence modeling that replaces the KV cache (Transformer) or learned hidden state (RNN/SSM) with a fixed-size matrix of exponential moving average counters — one per vocabulary token, tracking *when* each token appeared, not *what* it means.

```
Transformer KV Cache:              Per-Slot EMA:
  Stores d-dim vectors per pos       Stores N-dim EMA per vocab slot
  Size: O(T × d), grows with T      Size: O(V × N), constant
  Needs BPTT for training            No BPTT — pure analytic update
  
  [K₁,V₁][K₂,V₂]...[K_T,V_T]      slot "the":  C = [0.82, 0.35, ...]
                                     slot "cat":  C = [0.91, 0.67, ...]
                                     slot "sat":  C = [0.00, 0.00, ...]
```

**Core idea**: Each time step writes a 1-bit signal ("which token appeared") instead of a d-dimensional semantic vector. Semantics are generated *on demand* at prediction time (Late Semantic Binding) by a Transformer decoder that combines the static word embeddings with EMA temporal statistics.

## Key Results (WikiText-103, 46M params)

| Model | seq_len | Val PPL | Train PPL |
|-------|---------|---------|-----------|
| **Per-Slot EMA** | 512 | **23.4** | **18.6** |
| RoPE Transformer | 2048 | 24.2 | 22.6 |
| RoPE Transformer | 4096 | 22.0 | 19.9 |

- EMA@512 **outperforms** RoPE@2048 — because c_base provides full document context regardless of seq_len
- EMA's train PPL is the **lowest across all configurations** — its orthogonal slot design eliminates inter-token information competition
- RoPE needs 8x compute (seq_len=4096) to surpass EMA

## Key Properties

| Property | Mechanism |
|----------|-----------|
| **O(1) inference state** | State size = V × N, independent of sequence length |
| **No BPTT** | EMA update is analytic: C ← ρC + I(x=k), no learnable params in state |
| **Intrinsic position encoding** | Exponential decay ρ^(T-t) naturally encodes relative distance |
| **Infinite context training** | c_base carries full document history across arbitrary chunk boundaries |
| **Hyperparameter robust** | N=32~128, half-life=1024~4096 all give <2% PPL variation |

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Prepare Data
```bash
# WikiText-103
python datasets/prepare_wikitext103.py
```

### Train
```bash
# Per-Slot EMA (seq_len is just a compute parameter — c_base provides full context)
python train.py --model ema \
    --train_file datasets/wikitext103_train.txt \
    --val_file datasets/wikitext103_val.txt \
    --tokenizer gpt2 \
    --d_model 512 --n_layers 6 --n_heads 8 --N 64 \
    --seq_len 512 --batch_size 56 \
    --lr 3e-4 --epochs 20

# RoPE Transformer baseline (seq_len = context window)
python train.py --model rope \
    --train_file datasets/wikitext103_train.txt \
    --val_file datasets/wikitext103_val.txt \
    --tokenizer gpt2 \
    --d_model 512 --n_layers 6 --n_heads 8 \
    --seq_len 2048 --batch_size 16 \
    --lr 3e-4 --epochs 20
```

### Run Tests
```bash
python tests/test_pipeline_thorough.py         # CPU
python tests/test_pipeline_thorough.py --cuda  # GPU (Triton)
```

## Architecture Overview

```
Input: x₁, x₂, ..., x_T  (token sequence)
          │
          ▼
┌─────────────────────┐
│  Per-Slot EMA State  │  C[k] = ρ · C[k] + I(x_t = k)
│  V × N matrix        │  Pure analytic, no gradients
│  (one row per token)  │  c_base: carries full doc history
└─────────────────────┘
          │
          ▼  SiLU(C · W_pe) → multiplicative gating
┌─────────────────────┐
│  Static Embeddings   │  E[k] ∈ R^d (shared across layers)
│  × Time Gate         │  K = E ⊙ gate,  V = E ⊙ gate
└─────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Transformer Decoder │  Only Query iterates across layers
│  (Cross-Attention)   │  Late Semantic Binding
│  + FFN (SwiGLU)      │
└─────────────────────┘
          │
          ▼
      logits → next token prediction
```

## File Structure

```
mathbrain/
  model.py                 # SlotTransformerLM (Per-Slot EMA model)
  baseline.py              # RoPETransformerLM (standard Transformer)
  data.py                  # Document-aware data pipeline with c_base
  ema.py                   # EMA decay rate computation
  flash_ema_attention.py   # PyTorch reference implementation
  triton_flash_ema.py      # Triton GPU kernels (forward + backward)
  triton_ema_query.py      # Triton kernel for EMA sequential scan
  triton_fused_gating.py   # Triton kernel for fused gating

train.py                   # Training script (EMA + RoPE)
tests/                     # Pipeline correctness tests
datasets/                  # Data preparation scripts
```

## Important: Data Pipeline Requirements

Per-Slot EMA is sensitive to **causal purity** — each training chunk must come from exactly one document. Cross-document concatenation introduces spurious temporal co-occurrences that EMA records as real signals (PPL degrades from 23.3 to 30.6). The data pipeline enforces this by:

1. Per-document chunking (no cross-doc concatenation)
2. c_base loaded only from same-document history
3. Loss masked at padding positions

This is not a bug — it's a fundamental property of temporal statistics-based models.

## Citation

Paper forthcoming.

## License

Apache 2.0
