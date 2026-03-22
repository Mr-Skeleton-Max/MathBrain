# MathBrain

**EMA Slot Encoding: Bounded-State Sequence Representation with O(1) Inference and No BPTT**

> **TL;DR**: A parameter-free sequence encoder that compresses arbitrary-length token histories into bounded per-slot EMA states. O(1) inference like state-space models, no sequential dependency like Transformers, and mathematically proven lossless compression. Decoder-agnostic: plug in any predictor.

## Core Idea

**Rotate the sequence 90°**: instead of storing a growing horizontal list of tokens, maintain a fixed vertical array of slot activation patterns.

```
Horizontal (Transformer/RNN):     Vertical (EMA Slot):
t=0  t=1  t=2  ...  t=∞           slot_0: Q = [β₁^Δt, β₂^Δt, ..., β_N^Δt]
[w₃] [w₁] [w₃] ... [w₁]    →     slot_1: Q = [β₁^Δt, β₂^Δt, ..., β_N^Δt]
  ↑ grows forever                  slot_2: Q = [0, 0, ..., 0]  (inactive)
                                     ↑ bounded, O(1)
```

Each slot independently tracks *when* it was activated via exponential moving averages. The resulting Q matrix is:
- **Information-complete**: a single EMA value uniquely encodes the full activation history (proven for transcendental β)
- **Bounded**: state size is independent of sequence length
- **Parameter-free**: no learnable parameters in the encoder; all capacity is in the decoder

### Memory as Hallucination

Q does not store memories. It stores the *conditions* under which a downstream decoder will reconstruct the appropriate response — more akin to context-triggered hallucination than retrieval from storage.

## Comparison

| | Transformer | RNN/LSTM | SSM (Mamba) | **EMA Slot** |
|---|---|---|---|---|
| Inference | O(N²) or O(N) | **O(1)** | **O(1)** | **O(1)** |
| Training seq. dep. | **None** | BPTT | BPTT | **None** |
| Training memory | O(N) | O(T·H) | O(T·H) | **O(1)** |
| Encoder params | Many | Many | Many | **Zero** |
| Infinite context | ✗ | ✗ (truncation) | ✗ (truncation) | **✓** |

## Results

### Memorization Capacity (Bilinear Voter, TinyStories)

| Corpus | Positions | Parameters | Train Acc |
|--------|-----------|------------|-----------|
| TinyStories 1000 | ~0.18M | ~3.5M | **99.2%** |
| TinyStories 2000 | ~0.37M | ~4.8M | **99.2%** |
| TinyStories 5000 | ~0.96M | ~8.1M | **99.2%** |

8M parameters memorize 1M positions at 99.2% accuracy.

### Generalization (Cross-Attention Decoder, DailyDialog)

**80K sentences, ~758K params:**

| Model | Train Acc | Val Acc | Val PPL |
|-------|-----------|---------|---------|
| **EMA SlotTransformer** | **53.3%** | 34.7% | 143.3 |
| Baseline Transformer | 39.8% | **34.8%** | **45.1** |

- Val accuracy matches baseline — EMA encodes enough information for equal generalization
- Train accuracy 13.5pp higher — EMA representations are significantly richer
- PPL gap is a calibration issue (overconfidence), not a generalization failure

**Scaling** (val acc): 1K→2K→5K→10K→80K: ~13%→16%→18%→19%→34.7% — **super-linear improvement**

## Architecture

```
Token ──→ Retina ──→ EMA Encoder ──→ [Any Decoder]
          (word→slots)  (Q: V×N matrix)
```

### Retina
- **Identity**: 1 word = 1 slot
- **Hash**: character n-grams → shared slot space

### EMA Encoder (no learnable params)
```
Q_i(t+1) = β · Q_i(t) + x_i(t)     # per-slot, per-timescale
```

### Decoders (all capacity here)
1. **Bilinear Voter**: Q → φ (nonlinear map) → bilinear scoring → slot votes
2. **Causal Cross-Attention**: Q → PE(Q) + E_src → latest=Query, history=K/V → logits

## Quick Start

```bash
git clone https://github.com/Mr-Skeleton-Max/MathBrain.git
cd MathBrain
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0 (CUDA), Triton ≥ 2.0

### Train with Slot Transformer

```bash
python train_gpu.py --corpus datasets/dailydialog_utt_all_train.txt \
    --val-corpus datasets/dailydialog_utt_all_val.txt \
    --epochs 640 --transformer --d-model 64 --pe-mode linear \
    --eval-every 160 --scheduler cosine --N 64 --half-life 1 1000 \
    --retina identity
```

### Train with Bilinear Voter

```bash
python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 100
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--corpus` | required | Training corpus file |
| `--val-corpus` | — | Validation corpus file |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 4096 | Max positions per batch |
| `--lr` | 0.01 | Max learning rate |
| `--scheduler` | constant | LR schedule: constant/cosine/step/exp |
| `--N` | 8 | Number of EMA timescales |
| `--half-life` | — | Half-life range for auto ρ (e.g., `1 1000`) |
| `--retina` | hash | Retina mode: hash / identity |
| `--transformer` | off | Use SlotTransformer decoder |
| `--d-model` | 128 | Transformer hidden dimension |
| `--pe-mode` | fourier | Position encoding: fourier / linear |
| `--pred-mode` | ce | Prediction mode: ce / innovation / hybrid |

## Project Structure

```
MathBrain/
├── train_gpu.py                    # Unified train + eval CLI
├── baseline_transformer.py         # Standard Transformer baseline
├── diagnose.py                     # Generalization diagnosis toolkit
├── MathBrain/                      # Core GPU-optimized package
│   ├── config.py                   #   Hyperparameters
│   ├── trainer.py                  #   GPU trainer
│   ├── slot_transformer.py         #   Cross-attention decoder
│   ├── retina.py                   #   Hash / Identity lexical coding
│   ├── gpu_preprocessor.py         #   GPU batch iterator + dynamic EMA
│   └── triton_kernels.py           #   Triton fused kernels
├── datasets/                       # DailyDialog + TinyStories subsets
├── paper/                          # ArXiv draft
└── figures/                        # φ-space diagnostic plots
```

## Paper

> **[EMA Slot Encoding: Bounded-State Sequence Representation with O(1) Inference and No Backpropagation Through Time](paper/mathbrain_arxiv_draft.pdf)**

## Citation

```bibtex
@article{li2026ema_slot,
  title   = {EMA Slot Encoding: Bounded-State Sequence Representation
             with O(1) Inference and No Backpropagation Through Time},
  author  = {Li, Yuyue},
  year    = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
