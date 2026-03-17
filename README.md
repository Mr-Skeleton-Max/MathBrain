# MathBrain

**Bounded-State Categorical Voting for Online Sequence Learning**

MathBrain is an experimental sequence prediction architecture that does not store or attend over token history. Instead, it compresses all context into a bounded EMA state and predicts the next token through explicit memory voting -- achieving O(1) inference and O(logN) train(experimental,but stable), online learning, and full interpretability by design.

> **TL;DR**: Transformers grow state with sequence length. MathBrain doesn't. It treats next-token prediction as a categorical voting problem over bounded-state features, with constant memory and inference cost regardless of how many tokens it has seen.

## Key Ideas

- **Prediction is voting.** Every next-token prediction is a categorical competition. The architecture is designed around this from the ground up.
- **History should be compressed, not stored.** Multi-timescale EMA states capture "when did I last see this pattern" without keeping the full sequence.
- **Voting tolerates compression.** You don't need exact replay to rank the correct candidate above competitors -- sufficient evidence is enough.
- **Low-rank factorization is implicit abstraction.** The CP_RANK-dimensional intermediate space in A's prediction (`E_src → virtual space → E_tgt`) acts as a hidden layer -- each source slot projects through a learned abstract space before voting on targets.
- **Width over depth.** Extending EMA timescales (N) and feature dimensions (D) is more effective than stacking layers. Single-layer MathBrain with sufficient width reaches 97-99% on test corpora.
- **Everything is inspectable.** Every prediction traces back to specific memory entries, source slots, and feature vectors. No black box.

## How It Works

Consider predicting the next word after *"the cat sat on the"*. A transformer must attend over all previous tokens. MathBrain works differently:

```
"the cat sat on the"

  1. Hash each word into sparse slot IDs           "the" → {3712, 8891, ...}
     (deterministic character n-gram hashing)       "cat" → {1204, 5567, ...}

  2. Compress history into bounded EMA state        Q[3712] = [0.95, 0.82, ..., 0.01]
     (N timescales: fast → slow decay)                        ← recent ──── distant →

  3. Extract positional features φ                  φ[3712] = [0.73, -0.21, ...] ∈ ℝ^D
     (cosine-chaos map at edge of chaos)            bounded, no explosion

  4. Vote from two memory systems:
     B (short-term): exact episodic traces          "sat on the → mat"  score: 2.31
     A (long-term):  generalized structure          "on the → mat"      score: 1.87
                                                    "on the → floor"    score: 0.42
  5. Winner: "mat" ✓
     (every vote is traceable to specific source slots and feature vectors)
```

The key insight: **voting tolerates compression**. To rank the correct response above competitors, the system needs sufficient evidence, not exact replay. This is why bounded state works.

## Architecture

```
                              ┌─────────────────────────────┐
  Token ──→ Hash Retina ──→   │  Q: Multi-timescale EMA     │ ──→ Active slots
            (n-gram → slots)  │  N decay rates: ρ₁..ρ_N    │     + Q values
                              │  bounded state, O(1) memory │
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  φ: Cosine-Chaos Feature Map │
                              │  α=2 (edge of chaos)        │
                              │  L recursive folds → ℝ^D    │
                              └──────────────┬──────────────┘
                                             │
                         ┌───────────────────┼───────────────────┐
                         │                                       │
              ┌──────────▼──────────┐             ┌──────────────▼──────────┐
              │  B: Short-Term      │             │  A: Long-Term           │
              │  Sparse hash table  │             │  Slice-wise low-rank    │
              │  Exact episodic     │             │  Generalized structure  │
              │  traces             │             │  E_src[s], E_tgt[t]     │
              │  (hippocampus)      │             │  ∈ ℝ^{d×D}             │
              │                     │             │  (neocortex)            │
              └──────────┬──────────┘             └──────────────┬──────────┘
                         │    ℓ_B(t) = Σᵢ ⟨b_{sᵢ,t}, φ̂ᵢ⟩      │
                         │                                       │
                         │         ℓ_A(t) = ⟨E_tgt[t], C_A⟩_F   │
                         └───────────────────┬───────────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  Categorical Vote            │
                              │  ℓ(t) = ℓ_B(t) + ℓ_A(t)    │
                              │  Predict: argmax_t ℓ(t)      │
                              └──────────────────────────────┘
```

### A: The Core Prediction Engine

A is the primary prediction module. It uses a **slice-wise low-rank factorization**: each slot `s` has two factor matrices `E_src[s]` and `E_tgt[s]` in `ℝ^{d×D}`, where `d` = CP_RANK (default 384). Prediction works as:

```
score[tgt] = Σ_src ⟨E_tgt[tgt], E_src[src] ⊙ φ(src)⟩_F / n_active
```

The `d`-dimensional rank space acts as an **implicit hidden layer** -- each source doesn't directly vote on targets, but first projects through a learned abstract space. This provides representational abstraction without explicit multi-layer stacking.

A is trained by **dream sleep**: a batch optimization (AdamW + MSE loss) on pre-extracted `(φ, gold_word)` pairs. The EMA states and φ features are precomputed as fixed inputs -- only `E_src` and `E_tgt` are learned. This means backpropagation is used, but only within the prediction layer -- **not through time steps** (no BPTT). The temporal feature extraction pipeline (HashRetina → EMA → CosineChaos) is entirely deterministic and fixed.

### B: Short-Term Episodic Memory (Experimental)

B is a sparse hash table that writes exact co-activation traces online (`Δb = η · gap · φ̂`). It serves as a fast episodic memory (hippocampus analog) that can capture patterns immediately. The original design uses B as a teacher for A during sleep consolidation.

**Current status:** B's online wake learning has signal-to-noise limitations at scale. The v2 trainer bypasses B entirely, training A directly on gold data. Redesigning B as a proper episodic memory that tracks both input and output trajectories is an active research direction.

## How It Differs from Transformers

| | **Transformer** | **MathBrain** |
|---|---|---|
| **Inference cost** | O(L) or O(L²) in sequence length | **O(1)** after EMA saturation |
| **KV cache** | Grows with every token | No cache; bounded EMA state |
| **Online learning** | Requires full retraining | Native: wake writes are instant |
| **Interpretability** | Post-hoc attention maps | Every vote is an explicit memory trace |
| **Memory model** | Implicit in weights | Explicit dual system with inspectable entries |
| **Training** | Backprop through time | Backprop within prediction layer only; no BPTT through time steps |

The tradeoff: MathBrain's prediction cost scales with vocabulary size O(V) rather than sequence length. It excels at continual, interpretable, bounded-state operation -- not at replacing large-scale language models on broad benchmarks.

## Results

### v2: Gold-Teacher Sleep (A-only, direct training)

| Setting | N | D | Accuracy | Note |
|---------|---|---|----------|------|
| tinystories_10 (10 stories) | 4 | 8 | **99.41%** | Near-perfect memorization; ~10 errors are genuine ambiguity |
| tinystories_60 (60 stories) | 4 | 8 | **99.06%** | Capacity holds at 6x corpus |
| tinystories_200 (200 stories) | 4 | 8 | 71.14% | Context window too short (N=4, ~14 steps) |
| tinystories_200 (200 stories) | 16 | 32 | **97.84%** | Extending EMA scales solves the context bottleneck |

**Key finding:** Scaling N (EMA timescales) and D (feature dimension) together is far more effective than stacking layers. N=16 extends the effective context window from ~14 to ~210 steps, which is sufficient for 200 stories.

### v1: Wake-Sleep with B (original architecture)

| Setting | Accuracy | Note |
|---------|----------|------|
| tinystories_10, A+B | 92-93% | Combined dual memory |
| tinystories_10, A-only after sleep | 84-86% | B as teacher, A as student |
| tinystories_11_20, wake-only B | 87.0% | Pure online learning |
| PTB first 200 lines | 85% | Standard benchmark |

These results are on small-scale corpora. The claim is not SOTA on broad benchmarks, but viability of a fundamentally different regime: bounded state, categorical voting, and full interpretability.

## Quick Start

### Install

```bash
git clone https://github.com/Mr-Skeleton-Max/MathBrain.git
cd MathBrain
pip install -e .
```

### Train A module (v2, recommended)

```bash
# Default config (N=4, D=8) on tinystories_10
python experiments/train_a.py

# Scaled config (N=16, D=32) on tinystories_200
python experiments/train_a.py --corpus datasets/tinystories_200.txt --N 16 --D 32 --epochs 640

# Save trained model
python experiments/train_a.py --N 16 --D 32 --epochs 640 --save model.pt

# Sweep N/D configs to find the right scale
python experiments/train_a.py --corpus datasets/tinystories_200.txt --sweep
```

Or use the Python API directly:

```python
import numpy as np
from mathbrain import MathBrain, MathBrainConfig
from mathbrain.trainer import MathBrainTrainer

config = MathBrainConfig(
    N=16,
    RHO=tuple(np.power(0.5, 1.0 / np.geomspace(0.6, 210, 16)).tolist()),
    D_PHI=32,
)

model = MathBrain(config)
trainer = MathBrainTrainer(model, device='auto')  # auto: cuda > mps > cpu
corpus = open('datasets/tinystories_200.txt').read().strip().split('\n')

trainer.fit(corpus, epochs=640)
print(trainer.evaluate(corpus))
trainer.save('model.pt')
```

### Train with Wake-Sleep (v1, original)

```bash
# Single wake-sleep cycle on 10 TinyStories
python train.py --corpus datasets/tinystories_10.txt --mode cycle --cycles 1

# Multi-cycle training
python train.py --corpus datasets/tinystories_20.txt --mode cycle --cycles 5
```

## Project Structure

```
MathBrain/
├── train.py                     # Training CLI (wake, sleep, cycle modes)
├── mathbrain/                   # Core package
│   ├── model.py                 #   Main MathBrain class
│   ├── config.py                #   All hyperparameters (N, D, CP_RANK, etc.)
│   ├── trainer.py               #   [v2] PyTorch A-module trainer (CUDA/MPS/CPU)
│   ├── retina.py                #   Hash-based lexical coding (HashRetina)
│   ├── q_state.py               #   Multi-timescale EMA state
│   ├── phi_encoder_chaos.py     #   Cosine-chaos feature map (edge of chaos)
│   ├── a_knowledge_v2.py        #   Long-term slice-wise low-rank factorization
│   ├── b_memory_hashtable_v2.py #   Short-term sparse memory (experimental)
│   ├── sleep_v2_mlx.py          #   Wake-sleep consolidation (MLX backend)
│   └── inference_fast.py        #   Sparse matrix decoder for slot→word
├── datasets/                    # Demo corpora
├── experiments/                 # Experiment scripts
│   ├── train_a.py               #   [v2] CLI for A-module training + N/D sweep
├── paper/                       # ArXiv draft (.tex + .pdf)
└── docs/                        # Mathematical specification (Chinese)
```

## Key Experiments

### 1. Gold-Teacher Sleep: A-Only Training (v2)

**Key discovery:** Bypassing B entirely and training A directly on gold (context, next_word) pairs produces dramatically better results than the B→A wake-sleep pipeline.

| Method | tinystories_10 |
|--------|---------------|
| v1: B wake → A sleep (B as teacher) | 84-86% A-only |
| v1: A + B combined | 92-93% |
| **v2: A trained directly on gold** | **99.41%** |

This reveals that A's architecture (EMA + CosineChaos + HashRetina + low-rank factorization + independent voting) is itself a powerful sequence model. B's noisy teacher signal was the bottleneck, not A's capacity.

### 2. Width Scaling: N and D

**Key discovery:** When accuracy drops on larger corpora (200 stories: 71% with default N=4, D=8), the fix is **extending EMA timescales**, not adding layers.

N (EMA scales) and D (φ dimension) must be scaled together -- D without more N doesn't add information, because the information source is the N-dimensional EMA state.

| N | D | Max EMA half-life | tinystories_200 |
|---|---|-------------------|-----------------|
| 4 | 8 | ~14 steps | 71.14% |
| 16 | 32 | ~210 steps | **97.84%** |

**Why not multi-layer?** We investigated multi-layer stacking but found it unnecessary. The CP_RANK-dimensional low-rank space already provides implicit abstraction (each source projects through a 384-dim virtual space before voting on targets). The real bottleneck was temporal context, solved by wider EMA.

### 3. Tree-Parallel Wake Training (v1)

**Script:** `experiments/run_parallel_wake.py`

Because wake writes are local edge updates, they can be parallelized. The trainer freezes B, farms out shards in parallel, then tree-merges sparse deltas. Per-round cost drops from O(N) to O(k log N).

## Current Status

This is an active research project.

**Working:**
- Core architecture (HashRetina, EMA, CosineChaos, low-rank A) -- validated
- PyTorch trainer with CUDA/MPS/CPU support, model save/load
- 99.41% on tinystories_10 (N=4, D=8), 97.84% on tinystories_200 (N=16, D=32)
- Width scaling (N, D) confirmed as the right axis for context extension

**Open directions:**
- Redesigning B as episodic memory that tracks both input and output trajectories
- Scaling to larger corpora (1000+ stories, full TinyStories, PTB)
- Systematic baselines against n-gram / LSTM / Transformer at matched parameter counts
- Generalization evaluation (train/test split, not just memorization)

## Paper

For the full theoretical motivation -- why categorical voting is a viable first principle, why bounded state becomes possible once you give up exact history reconstruction, and how memory-as-trigger differs from memory-as-retrieval -- please read the paper:

> **[Bounded-State Categorical Voting for Online Sequence Learning: A White-Box Alternative to Length-Growing Black-Box Models](paper/mathbrain_arxiv_draft.pdf)**

The paper covers the mathematical formulation in detail, including the complexity analysis, the wake update derivation, the sleep consolidation objective, and the biological correspondence argument. The README gives the intuition; the paper gives the proofs.

For the research journey that led to this architecture -- from BubbleStream to GPT-2 memory analysis to the categorical voting insight to EMA encoding -- see [docs/research_log.md](docs/research_log.md).

## Requirements

- Python >= 3.8
- NumPy >= 1.21.0
- PyTorch >= 2.0.0 (required for v2 trainer; supports CUDA, MPS, and CPU)

Optional:
- [MLX](https://github.com/ml-explore/mlx) >= 0.5.0 (Apple Silicon, for v1 wake-sleep)

## A Note on Code Quality

This codebase grew out of a solo research project. The author is a researcher first and not a professional software engineer -- the code works and the experiments are reproducible, but it is not as clean or well-organized as a production library. Apologies for any rough edges. If something is confusing, please open an issue and I'll be happy to explain.

## Contributing

This is an active research program with many open directions. I would love to collaborate:

- **Discussions**: If you find the categorical-voting premise interesting (or flawed!), let's talk. Open an issue or reach out directly.
- **Experiments**: Scaling to larger corpora, systematic baselines against n-gram/LSTM/Transformer -- all need more hands.
- **Theory**: The connections to information geometry, compressed sensing, and neural coding theory are largely unexplored.
- **Engineering**: Better implementations, GPU kernels, edge deployment -- the sparse voting structure should be very hardware-friendly.

PRs, issues, and ideas are all welcome.

If this project interests you, a :star: on GitHub would be greatly appreciated -- it helps others discover this work!

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
