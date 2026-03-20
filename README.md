# MathBrain

**Bounded-State Categorical Voting for Online Sequence Learning**

MathBrain is an experimental sequence prediction architecture that does not store or attend over token history. Instead, it compresses all context into a bounded EMA state and predicts the next token through explicit memory voting — achieving O(1) inference, online learning, and full interpretability by design.

> **TL;DR**: Transformers grow state with sequence length. MathBrain doesn't. It treats next-token prediction as a categorical voting problem over bounded-state features, with constant memory and inference cost regardless of how many tokens it has seen.

## Key Ideas

- **Sequence context can be rotated from "temporal horizontal" to "model vertical".** Instead of attending over token history (O(N)), MathBrain compresses context into per-slot EMA states — turning temporal information into bounded feature vectors.
- **Per-slot independence is critical.** Each slot maintains its own EMA state independently. Independent slots enable independent voting, which tolerates compression noise.
- **Nonlinear feature expansion at edge of chaos.** Raw EMA states have very low discriminability. Cosine-chaos folding amplifies micro-differences between contexts into distinguishable features, while remaining bounded.
- **The predictor is a modular component.** The current bilinear low-rank voter (E_src, E_tgt) is one choice. The encoding is the core contribution, not the predictor.
- **Everything is inspectable.** Every prediction traces back to specific memory entries, source slots, and feature vectors.

## Architecture

```
  Token ──→ Hash Retina ──→  Q: Multi-timescale EMA  ──→ φ: Cosine-Chaos  ──→ Predictor
            (n-gram → slots)   N decay rates: ρ₁..ρ_N      Map (D dims)        (bilinear voter)
            O(1), deterministic   O(1)/step, bounded        bounded [-1,1]^D    learned
```

For a sentence like *"the cat sat on"*:

1. **Hash** each word into sparse slot IDs — `"the" → {3712, 8891, ...}`
2. **Compress** history into bounded EMA state — `Q[3712] = [0.95, 0.82, ..., 0.01]`
3. **Extract** positional features φ — CUDA warp-shuffle chaos map, D=32 dims, all in registers
4. **Vote** via bilinear predictor — `score[word] = ⟨E_tgt, E_src ⊙ φ⟩` → next word

## GPU-Optimized Training Pipeline

The training pipeline is fully GPU-accelerated with custom CUDA and Triton kernels:

| Component | Implementation | Time/batch |
|---|---|---|
| EMA filter | Triton parallel scan | ~1.7ms |
| Phi encoder | **CUDA warp shuffle** (全寄存器, zero VRAM) | ~0.6ms |
| Bilinear forward | Triton fused gather+mul+reduce | ~1.2ms |
| E_word build | Dense matmul `wp_dense @ E_tgt` | ~0.5ms |
| Logits + CE | cuBLAS matmul + PyTorch CE | ~0.8ms |
| Backward | Dense matmul + Triton atomic scatter | ~4.6ms |
| **Total** | | **~9.6ms/batch** |

**Epoch time**: ~530ms for 1000 sentences (221K positions, 53 batches).

### Key Optimizations

- **CUDA warp shuffle phi encoder**: 32 threads = 1 warp, phi computed entirely in registers via `__shfl_sync`. Zero global memory access during 32-fold chaos iteration. 15× faster than Triton version.
- **E_word fusion**: Replaces sparse `word_proj` kernel with dense `E_word = wp_dense @ E_tgt`, leveraging cuBLAS for both forward and backward. Eliminates 133MB intermediate tensors.
- **Dense matmul backward**: Replaces atomic_add contention in word_proj backward with `d_scores = d_logits @ wp_dense`. 3× faster.
- **Triton bilinear**: Fused gather + element-wise multiply + reduce. Backward uses efficient L2-cache atomic scatter.
- **Dynamic EMA**: Per-batch GPU computation instead of pre-computing all positions. Saves ~2GB VRAM.
- **CPU-parallel preprocessing**: 207-worker hash retina encoding for vocabulary building.

## Quick Start

### Install

```bash
git clone https://github.com/Mr-Skeleton-Max/MathBrain.git
cd MathBrain
pip install -e .
```

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.0 (CUDA), Triton ≥ 2.0

### Train (GPU)

```bash
# Basic training (1000 stories, 100 epochs)
python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 100

# With periodic evaluation
python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 320 --eval-every 40

# Custom learning rate schedule
python train_gpu.py --corpus datasets/tinystories_2000.txt --epochs 640 \
    --lr 0.01 --min-lr 1e-5 --warmup-pct 0.05

# Scaled config
python train_gpu.py --corpus datasets/tinystories_5000.txt --epochs 320 \
    --N 8 --D 32 --rank 32 --batch-size 4096
```

### Python API

```python
from MathBrain import MathBrainConfig, MathBrainTrainer

cfg = MathBrainConfig(
    N=8,
    RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
    D_PHI=32, CP_RANK=32,
)

trainer = MathBrainTrainer(cfg, device='cuda')

corpus = open('datasets/tinystories_1000.txt').read().strip().split('\n')
trainer.fit(corpus, epochs=320, lr=0.01)
trainer.evaluate(corpus)

# Predict next word
predictions = trainer.predict_topk(["the", "cat", "sat", "on"], k=5)
for word, score in predictions:
    print(f"  {word}: {score:.4f}")
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--corpus` | required | Training corpus file |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 4096 | Max positions per batch |
| `--lr` | 0.01 | Max learning rate |
| `--min-lr` | lr/25000 | Final learning rate |
| `--warmup-pct` | 0.05 | Warmup fraction (5%) |
| `--N` | 8 | Number of EMA timescales |
| `--D` | 32 | Phi encoding dimension |
| `--rank` | 32 | CP rank (d = rank × D) |
| `--eval-every` | 0 | Evaluate every N epochs |
| `--save` | — | Save model path |

## Results

All results are train-set accuracy (memorization) on TinyStories subsets.

### v2: GPU-Accelerated Direct Training

| Corpus | N | D | CP_RANK | Accuracy | Epoch Time |
|--------|---|---|---------|----------|------------|
| tinystories_10 | 4 | 8 | 384 | **99.41%** | <100ms |
| tinystories_60 | 4 | 8 | 384 | **99.06%** | <200ms |
| tinystories_200 | 8 | 32 | 384 | **97.13%** | ~400ms |
| tinystories_1000 | 8 | 32 | 32 | — | ~530ms |
| tinystories_2000 | 8 | 32 | 32 | — | ~910ms |

### φ Encoding Analysis

| P matrix | φ effective rank | Accuracy |
|----------|-----------------|----------|
| Random Gaussian (cond=50) | 25.9 | 97.96% |
| Random Orthogonal (cond=1) | 21.7 | 96.93% |
| No P (chaos only) | ~18 | 87.22% |

**Key finding**: P's non-uniform singular values create multi-resolution features that increase φ effective rank by ~25%.

## How It Differs from Transformers and SSMs

| | **Transformer** | **SSM (S4/Mamba)** | **MathBrain** |
|---|---|---|---|
| **Context encoding** | Attend over all tokens O(N²) | Global state vector O(1) | Per-slot EMA O(V) |
| **Inference cost** | O(N) or O(N²) | O(1) | O(V) per step |
| **Online learning** | Requires retraining | Requires retraining | Native |
| **Interpretability** | Post-hoc attention maps | Opaque state | Every vote traceable |
| **Training** | BPTT through full sequence | BPTT through full sequence | Backprop in predictor only |

## Project Structure

```
MathBrain/
├── train_gpu.py                    # Unified train + eval CLI
├── MathBrain/                      # Core GPU-optimized package
│   ├── config.py                   #   Hyperparameters (N, D, CP_RANK, RHO)
│   ├── trainer.py                  #   GPU trainer (fit / evaluate / predict_topk)
│   ├── retina.py                   #   Hash-based lexical coding
│   ├── phi_encoder.py              #   Cosine-chaos feature map (CPU)
│   ├── gpu_phi_encoder.py          #   CUDA warp-shuffle phi encoder
│   ├── triton_kernels.py           #   Triton: bilinear fwd/bwd + word_proj
│   ├── streaming_preprocessor.py   #   CPU-parallel vocab builder
│   └── gpu_preprocessor.py         #   GPU batch iterator + dynamic EMA
├── datasets/                       # TinyStories subsets (10-5000 stories)
├── experiments/                    # Benchmarks and profiling scripts
├── paper/                          # ArXiv draft
├── docs/                           # Mathematical specification
└── v1/                             # Original CPU-based architecture
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
