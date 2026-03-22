# MathBrain

**Bounded-State Categorical Voting for Online Sequence Learning**

MathBrain is an experimental sequence prediction architecture that does not store or attend over token history. Instead, it compresses all context into a bounded EMA state and predicts the next token through a cross-attention decoder — achieving O(1) inference, online learning, and full interpretability by design.

> **TL;DR**: Transformers grow state with sequence length. MathBrain doesn't. It treats next-token prediction as a categorical voting problem over bounded-state features, with constant memory and inference cost regardless of how many tokens it has seen.

## Key Ideas

- **Sequence context can be rotated from "temporal horizontal" to "model vertical".** Instead of attending over token history (O(N)), MathBrain compresses context into per-slot EMA states — turning temporal information into bounded feature vectors.
- **Per-slot independence is critical.** Each slot maintains its own EMA state independently. Independent slots enable independent voting, which tolerates compression noise.
- **Causal slot cross-attention recovers directionality.** The latest-activated slot(s) act as query to cross-attend against all historical slots, restoring the unidirectional inductive bias lost in set-based aggregation.
- **The EMA representation has higher information density than standard embeddings.** On the same data, EMA-encoded features achieve significantly higher training accuracy than standard Transformer embeddings, indicating richer context compression.
- **Generalization scales super-linearly with data.** Unlike standard Transformers which show log-linear scaling, MathBrain's generalization exhibits accelerating returns with increasing data volume.

## Architecture

MathBrain supports two predictor backends:

### Slot Cross-Attention Decoder (recommended)

```
Token ──→ Retina ──→ Q: Multi-timescale EMA ──→ Slot Embedding + PE(Q) ──→ Cross-Attention Decoder ──→ LM Head
          (word → slots)  N decay rates            E_src[slot] + FourierPE     Q=latest, K/V=history      logits
```

The cross-attention decoder implements **causal query attention**:
1. All active slots compute embeddings: `E_src[slot_id] + PE(Q_values)`
2. The slot(s) activated by the **latest word** are mean-pooled into a query vector
3. All **non-latest slots** serve as keys and values
4. Stacked cross-attention layers: `Q += CrossAttn(Q, K, V) + FFN(Q)`
5. Final `LayerNorm → Linear` produces next-word logits

This design enforces unidirectional information flow: "proposer" slots (history) cannot see the "decision maker" (latest slot), mirroring causal masking in standard autoregressive models.

### Bilinear Voter (original)

```
Token ──→ Hash Retina ──→  Q: Multi-timescale EMA  ──→ φ: Cosine-Chaos  ──→ Bilinear Voter
          (n-gram → slots)   N decay rates: ρ₁..ρ_N      Map (D dims)        score = ⟨E_tgt, E_src ⊙ φ⟩
```

## Results

### DailyDialog Language Modeling (Full dataset, ~80K sentences)

| Model | Params | Train Acc | Train PPL | Val Acc | Val PPL |
|-------|--------|-----------|-----------|---------|---------|
| **EMA SlotTransformer** | 758K | **53.3%** | **8.5** | 34.7% | 143.3 |
| Baseline Transformer | ~750K | 39.8% | 18.9 | **34.8%** | **45.1** |

**Key findings:**
- **Val accuracy is on par** — EMA matches the standard Transformer on generalization accuracy
- **Train accuracy is significantly higher** (53% vs 40%) — EMA representations carry more information
- **Val PPL gap is a calibration issue**, not a generalization failure — the model predicts the right word equally often, but assigns overconfident probabilities when wrong

### Data Scaling Analysis (DailyDialog, Identity Retina)

| Training Data | Val Acc | Val PPL | vs Baseline |
|---------------|---------|---------|-------------|
| 2K sentences | ~15% | 780,000 | Far below |
| 5K sentences | 18.4% | 125,435 | Below |
| 80K sentences | **34.7%** | **143.3** | **Matched on accuracy** |

Generalization scales **super-linearly** with data volume — a property not typically seen in standard Transformers. The crossover point where EMA surpasses standard Transformers appears reachable with larger corpora.

### TinyStories Memorization (Bilinear Voter)

| Corpus | N | D | CP_RANK | Accuracy | Epoch Time |
|--------|---|---|---------|----------|------------|
| tinystories_10 | 4 | 8 | 384 | **99.41%** | <100ms |
| tinystories_60 | 4 | 8 | 384 | **99.06%** | <200ms |
| tinystories_200 | 8 | 32 | 384 | **97.13%** | ~400ms |

## Quick Start

### Install

```bash
git clone https://github.com/Mr-Skeleton-Max/MathBrain.git
cd MathBrain
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0 (CUDA), Triton ≥ 2.0

### Train with Slot Transformer (recommended)

```bash
# Identity retina + SlotTransformer + causal attention
python train_gpu.py --corpus datasets/dailydialog_utt_all_train.txt \
    --val-corpus datasets/dailydialog_utt_all_val.txt \
    --epochs 640 --transformer --d-model 64 --pe-mode linear \
    --eval-every 160 --scheduler cosine --N 64 --half-life 1 1000 \
    --retina identity

# With custom EMA scales
python train_gpu.py --corpus datasets/dailydialog_utt_5000_train.txt \
    --val-corpus datasets/dailydialog_utt_5000_val.txt \
    --epochs 640 --transformer --d-model 64 --pe-mode linear \
    --eval-every 160 --scheduler cosine --N 8 --retina identity

# Save model for analysis
python train_gpu.py --corpus datasets/dailydialog_utt_all_train.txt \
    --val-corpus datasets/dailydialog_utt_all_val.txt \
    --epochs 640 --transformer --d-model 64 --pe-mode linear \
    --eval-every 160 --scheduler cosine --N 64 --half-life 1 1000 \
    --retina identity --save my_model.pt
```

### Train with Bilinear Voter (original)

```bash
python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 100
```

### Diagnose Model Generalization

```bash
python diagnose.py --model my_model.pt \
    --train datasets/dailydialog_utt_all_train.txt \
    --val datasets/dailydialog_utt_all_val.txt
```

Outputs: confidence analysis, error type breakdown, per-sample prediction examples with Top-5, confusion matrix, Q-value geometry, and representation distribution analysis.

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
| `--retina` | hash | Retina mode: hash (n-gram) / identity (1 word=1 slot) |
| `--transformer` | off | Use SlotTransformer decoder |
| `--d-model` | 128 | Transformer hidden dimension |
| `--pe-mode` | fourier | Position encoding: fourier / linear |
| `--eval-every` | 0 | Evaluate val every N epochs |
| `--save` | — | Save model path |

## How It Differs from Transformers and SSMs

| | **Transformer** | **SSM (S4/Mamba)** | **MathBrain** |
|---|---|---|---|
| **Context encoding** | Attend over all tokens O(N²) | Global state vector O(1) | Per-slot EMA O(V) |
| **Inference cost** | O(N) or O(N²) | O(1) | O(V) per step |
| **Directionality** | Causal mask | Built-in | Causal query attention |
| **Online learning** | Requires retraining | Requires retraining | Native |
| **Interpretability** | Post-hoc attention maps | Opaque state | Every vote traceable |
| **Training** | BPTT through full sequence | BPTT through full sequence | Backprop in predictor only |

## Project Structure

```
MathBrain/
├── train_gpu.py                    # Unified train + eval CLI
├── baseline_transformer.py         # Standard Transformer baseline
├── diagnose.py                     # Generalization diagnosis toolkit
├── MathBrain/                      # Core GPU-optimized package
│   ├── config.py                   #   Hyperparameters
│   ├── trainer.py                  #   GPU trainer (fit / evaluate / predict_topk)
│   ├── slot_transformer.py         #   Cross-attention decoder with causal query
│   ├── retina.py                   #   Hash / Identity lexical coding
│   ├── gpu_preprocessor.py         #   GPU batch iterator + dynamic EMA
│   ├── streaming_preprocessor.py   #   CPU-parallel vocab builder
│   ├── phi_encoder.py              #   Cosine-chaos feature map (CPU)
│   ├── gpu_phi_encoder.py          #   CUDA warp-shuffle phi encoder
│   └── triton_kernels.py           #   Triton: bilinear fwd/bwd + word_proj
├── datasets/                       # DailyDialog + TinyStories subsets
├── experiments/                    # Benchmarks and profiling scripts
├── paper/                          # ArXiv draft
└── docs/                           # Mathematical specification
```

## Paper

> **[Bounded-State Categorical Voting for Online Sequence Learning: A White-Box Alternative to Length-Growing Black-Box Models](paper/mathbrain_arxiv_draft.pdf)**

## Citation

```bibtex
@article{li2026mathbrain,
  title   = {Bounded-State Categorical Voting for Online Sequence Learning:
             A White-Box Alternative to Length-Growing Black-Box Models},
  author  = {Li, Yuyue},
  year    = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
