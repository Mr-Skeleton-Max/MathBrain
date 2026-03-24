# EMA Slot Encoding

**A Parameter-Free Sequence Context Encoder with O(1) Inference and No Backpropagation Through Time**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](paper/mathbrain_arxiv_draft.pdf)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

---

## What is this?

A new way to encode sequence history for language modeling. Instead of storing a growing list of past tokens (Transformer KV cache) or compressing everything into a learned hidden state (RNN/SSM), we maintain a **fixed-size matrix** where each row is a vocabulary slot and each column is an EMA timescale.

```
Standard approach:                    EMA Slot approach:
                                      
  Sequence grows → cost grows           Vocabulary slots (fixed)
  [t=0][t=1][t=2]...[t=T]  O(T)        slot "the":  Q = [0.82, 0.35, 0.12, ...]
                                        slot "cat":  Q = [0.91, 0.67, 0.44, ...]
                                        slot "sat":  Q = [0.00, 0.00, 0.00, ...]  (never seen)
                                        ↑ size = V × N, independent of T
```

**Core insight**: when a token appears, its slot's EMA state gets a `+1` impulse and decays over time. With multiple decay rates (timescales), the resulting vector uniquely fingerprints the token's *entire activation history* — when it appeared, how often, how recently.

## Key Properties

| Property | How | Proven? |
|----------|-----|---------|
| **O(1) inference** w.r.t. sequence length T | Encoder update is constant per step; decoder attends over active slots (bounded by vocab size V, a constant) | ✅ By construction + Zipf saturation |
| **Mathematically lossless** under exact arithmetic | Theorem 1: for transcendental β and binary activations, distinct histories → distinct EMA values | ✅ Number-theoretic proof (3 lines) |
| **No BPTT** in training | Encoder has zero learnable parameters; β values are preset constants, not optimized | ✅ By construction |
| **~23K effective context** in float32 | Multi-timescale channels (h=1 to h=1000) jointly cover different temporal ranges | ✅ Quantitative formula |

### What this does NOT claim

- "Lossless" is a **mathematical** property under exact arithmetic. Finite-precision float32/64 limits resolution (quantified in §4.3 of the paper).
- O(1) is **with respect to T** (sequence length). The decoder cost is O(|active_slots| × d), bounded by O(V × d) — a constant for fixed architecture, but can be large.
- The current experiments are **small-scale** (≤1M params, 2 layers). Large-scale validation is future work.

## Results

### WikiText-2 (BPE 8K vocab, d=64, 2 layers, 960 epochs)

| Model | Params | Val PPL ↓ | Val Acc ↑ |
|-------|--------|-----------|-----------|
| **EMA Slot Transformer** | **1,014K** | **35.7** | **33.9%** |
| Transformer (no weight tying) | 1,124K | 38.4 | 32.2% |
| Transformer (weight tying) | 612K | 39.5 | 31.3% |

→ Zero-parameter encoder + standard decoder = fewer params, better perplexity.

### DailyDialog (80K sentences, ~758K params)

| Model | Train Acc | Val Acc | Val PPL |
|-------|-----------|---------|---------|
| **EMA Slot Transformer** | **53.3%** | 34.7% | 143.3 |
| Baseline Transformer | 39.8% | 34.8% | 45.1 |

→ Validation accuracy matches baseline. 13.5pp higher training accuracy demonstrates higher information density in EMA representation. PPL gap reverses on WikiText-2 — indicating a data-scale phenomenon, not an architectural limit.

### Memorization (Bilinear Voter, TinyStories)

8.1M parameters memorize 0.96M token positions at **99.2% accuracy**.

## How it works

```
Token ──→ Retina ──→ EMA Encoder ──→ Q matrix ──→ [Any Decoder] ──→ Prediction
          (word→slots)  (no params)    (V × N)       (all learning
                                       fixed size     happens here)
```

### 1. Retina (tokenization → slot mapping)
- **Identity**: 1 word = 1 slot (used in experiments)
- **N-gram hash**: character n-grams → shared slot space (compression)

### 2. EMA Encoder (zero learnable parameters)
```
Q_i^(n)(t) = β_n · Q_i^(n)(t-1) + x_i(t)
```
- `β_n = exp(-ln2 / h_n)` with half-lives `h_1=1` to `h_N=1000`
- Each slot independently tracks its activation history across N timescales
- **Theorem 1**: this mapping is injective for transcendental β (which almost all reals are)

### 3. Decoder (all model capacity)
- **Causal Cross-Attention Transformer**: projects Q values into embedding space, uses latest-word as query, historical slots as keys/values
- **Bilinear Voter**: nonlinear feature map φ(Q) → bilinear scoring → weighted slot votes

## Quick Start

```bash
git clone https://github.com/Mr-Skeleton-Max/MathBrain.git
cd MathBrain
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0 (CUDA), Triton ≥ 2.0

### Train EMA Slot Transformer on WikiText-2

```bash
# Step 1: Train BPE tokenizer
python train_bpe.py --corpus datasets/wikitext2_train.txt --vocab-size 8000

# Step 2: Train model
python train_gpu.py --corpus datasets/wikitext2_train.txt \
    --val-corpus datasets/wikitext2_val.txt \
    --epochs 960 --transformer --d-model 64 --n-layers 2 \
    --pe-mode linear --scheduler cosine --N 64 \
    --half-life 1 1000 --retina identity --tokenizer bpe
```

### Train on DailyDialog

```bash
python train_gpu.py --corpus datasets/dailydialog_utt_all_train.txt \
    --val-corpus datasets/dailydialog_utt_all_val.txt \
    --epochs 640 --transformer --d-model 64 --pe-mode linear \
    --eval-every 160 --scheduler cosine --N 64 --half-life 1 1000 \
    --retina identity
```

### Train Bilinear Voter on TinyStories

```bash
python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 100
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--corpus` | required | Training corpus file |
| `--val-corpus` | — | Validation corpus file |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 4096 | Max positions per batch |
| `--lr` | 0.01 | Max learning rate |
| `--scheduler` | constant | LR schedule: constant / cosine / step / exp |
| `--N` | 8 | Number of EMA timescales |
| `--half-life` | — | Half-life range (e.g., `1 1000`) |
| `--retina` | hash | Retina mode: hash / identity |
| `--transformer` | off | Use SlotTransformer decoder |
| `--d-model` | 128 | Transformer hidden dimension |
| `--n-layers` | 2 | Number of decoder layers |
| `--pe-mode` | fourier | Position encoding mode: fourier / linear |
| `--tokenizer` | word | Tokenizer: word / bpe |

## Project Structure

```
MathBrain/
├── train_gpu.py                    # Main training + evaluation CLI
├── train_bpe.py                    # BPE tokenizer training
├── baseline_transformer.py         # Standard Transformer baseline
├── diagnose.py                     # Generalization diagnostic toolkit
├── mathbrain/                      # Core package
│   ├── config.py                   #   Hyperparameters
│   ├── trainer.py                  #   GPU trainer
│   ├── slot_transformer.py         #   Cross-attention decoder
│   ├── retina.py                   #   Hash / Identity slot mapping
│   ├── gpu_preprocessor.py         #   GPU batch iterator + dynamic EMA
│   └── triton_kernels.py           #   Triton fused kernels
├── datasets/                       # DailyDialog + TinyStories subsets
├── paper/                          # Paper draft + supplementary
│   ├── mathbrain_arxiv_draft.pdf   #   Latest paper (PDF)
│   └── mathbrain_arxiv_draft.tex   #   LaTeX source
└── experiments/                    # Analysis scripts + heatmaps
```

## Current Status

🟡 **Pre-print stage** — arXiv submission pending endorsement.

**Done:**
- [x] Core EMA Slot encoder + two decoder backends
- [x] Uniqueness theorem (Theorem 1) + precision analysis
- [x] WikiText-2 benchmark: outperforms Transformer at matched scale
- [x] DailyDialog benchmark: matches validation accuracy
- [x] BPE tokenizer support

**Planned:**
- [ ] Large-scale experiments (>1M params, long context)
- [ ] Direct comparison with S4 / Mamba at matched scale
- [ ] Empirical |S_active| saturation curve vs T
- [ ] arXiv submission

## Paper

> **[EMA Slot Encoding: A Parameter-Free Sequence Context Encoder with O(1) Inference and No Backpropagation Through Time](paper/mathbrain_arxiv_draft.pdf)**
>
> Yuyue Li, March 2026

## Citation

```bibtex
@article{li2026ema_slot,
  title   = {EMA Slot Encoding: A Parameter-Free Sequence Context Encoder
             with O(1) Inference and No Backpropagation Through Time},
  author  = {Li, Yuyue},
  year    = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
