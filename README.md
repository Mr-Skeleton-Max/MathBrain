# MathBrain

**Bounded-State Categorical Voting for Online Sequence Learning**

MathBrain is an experimental sequence prediction architecture that does not store or attend over token history. Instead, it compresses all context into a bounded EMA state and predicts the next token through explicit memory voting -- achieving O(1) inference and O(logN) train(experimental,but stable), online learning, and full interpretability by design.

> **TL;DR**: Transformers grow state with sequence length. MathBrain doesn't. It treats next-token prediction as a categorical voting problem over bounded-state features, with constant memory and inference cost regardless of how many tokens it has seen.

## Key Ideas

- **Sequence context can be rotated from "temporal horizontal" to "model vertical".** Instead of attending over token history (O(N)), MathBrain compresses context into per-slot EMA states -- turning temporal information into bounded feature vectors. This encoding is orthogonal to the downstream predictor and compatible with any model architecture.
- **Per-slot independence is critical.** Each slot maintains its own EMA state independently. Summing across slots would collapse the architecture. Independent slots enable independent voting, which tolerates compression noise.
- **Nonlinear feature expansion at edge of chaos.** Raw EMA states have very low discriminability. Cosine-chaos folding amplifies micro-differences between contexts into distinguishable features, while remaining bounded. The Lyapunov exponent controls the tradeoff between amplification and stability.
- **The predictor is a modular component.** The current bilinear low-rank voter (E_src, E_tgt) is one choice. It could be replaced by any classifier -- MLP, Transformer, etc. The encoding is the core contribution, not the predictor.
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

The system has four modular components. Components 1-3 form a **fixed, deterministic position encoder** (no learned parameters). Component 4 is the only learned part.

```
  1. HashRetina          word → sparse slot IDs           O(1), deterministic
  2. Multi-timescale EMA  slot × time → Q ∈ R^N           O(1)/step, per-slot independent
  3. Cosine-Chaos Map     Q → φ ∈ R^D                     O(1), nonlinear expansion
  4. Predictor            {(slot, φ)} → next_word          learned (bilinear voter, or any classifier)
```

```
                              ┌─────────────────────────────┐
  Token ──→ Hash Retina ──→   │  Q: Multi-timescale EMA     │ ──→ Active slots
            (n-gram → slots)  │  N decay rates: ρ₁..ρ_N    │     + Q values
                              │  bounded state, O(1) memory │
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  φ: Cosine-Chaos Feature Map │
                              │  P mixing + L recursive folds│
                              │  bounded output ∈ [-1,1]^D   │
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │  Predictor (modular)         │
                              │  Current: bilinear low-rank  │
                              │  E_src ⊙ φ → E_tgt → score  │
                              │  Could be: MLP, Transformer  │
                              └──────────────────────────────┘
```

### Why This Works: EMA as Lossless Temporal Encoding

The key mathematical insight: for any sequence of token activations, the multi-timescale EMA state Q ∈ R^N provides a **unique encoding** of the activation history. The decomposition of a signal onto exponential bases (ρ₁^t, ρ₂^t, ..., ρ_N^t) is unique -- this ensures the encoding does not collapse into mere frequency statistics.

Two properties are critical:
- **Near-field priority, far-field persistence.** Fast-decaying scales (small ρ) give high weight to recent tokens; slow-decaying scales (large ρ) retain distant context. This is not a sliding window -- all history contributes, with exponentially graded importance.
- **Per-slot independence.** Each slot maintains its own Q vector. Slots are never summed or averaged. This preserves the identity of individual n-gram patterns and enables independent voting downstream.

### Cosine-Chaos: Amplifying Micro-Differences

Raw EMA states Q have very low discriminability -- different contexts produce nearly identical Q vectors (effective rank ~3-4 out of N=32 dimensions). The cosine-chaos map amplifies these micro-differences:

```
E = P @ (α_scale ⊙ Q).T       ← linear mixing (P provides cross-scale coupling)
M₀ = 0
Mₜ = cos(α · (shift(Mₜ₋₁) + E))   ← nonlinear folding, L iterations
φ = M_L
```

Experimental findings on the φ encoding:
- **P's mixing role is essential.** Removing P drops accuracy by ~10%. P provides full-connectivity across EMA scales that cyclic shift alone cannot efficiently achieve.
- **P's singular value distribution matters.** Non-uniform singular values (as in random Gaussian P) create multi-resolution features: large singular values → high-frequency discrimination, small → low-frequency smoothness. This improves φ effective rank from ~21 to ~26.
- **Folding depth L matters more than shift.** L=N (full cycle) gives the best results. With sufficient L, the chaos dynamics amplify even tiny Q differences into distinguishable φ vectors.
- **The Lyapunov exponent should be slightly positive** (weak chaos, not critical). α=2.0 with L=N gives λ≈+0.16, which outperforms the theoretical critical point λ=0.

### Current Predictor: Bilinear Low-Rank Voter

The predictor uses a **slice-wise low-rank factorization**: each slot `s` has factor matrices `E_src[s]` and `E_tgt[s]` in `ℝ^{CP_RANK×D}`. Prediction:

```
score[tgt] = Σ_src ⟨E_tgt[tgt], E_src[src] ⊙ φ(src)⟩_F / n_active
```

φ acts as an element-wise gate on E_src -- the same slot votes differently in different contexts. The CP_RANK-dimensional space provides implicit abstraction (a hidden layer without explicit stacking).

Trained by **dream sleep**: batch AdamW on pre-extracted (φ, gold_word) pairs. Backpropagation is used within the prediction layer only -- not through time steps (no BPTT). The entire temporal encoding pipeline is deterministic and fixed.

**This predictor is modular.** It could be replaced by an MLP, a small Transformer, or any classifier that takes {(slot_id, φ_vector)} as input. The encoding (components 1-3) is the core contribution.

## How It Differs from Transformers and SSMs

| | **Transformer** | **SSM (S4/Mamba)** | **MathBrain** |
|---|---|---|---|
| **Context encoding** | Attend over all tokens O(N²) | Global state vector O(1) | Per-slot EMA O(V) |
| **Information bottleneck** | None (full history) | State dimension | None (per-slot independent) |
| **Inference cost** | O(N) or O(N²) | O(1) | O(V) per step |
| **Online learning** | Requires retraining | Requires retraining | Native (predictor only) |
| **Interpretability** | Post-hoc attention maps | Opaque state | Every vote traceable |
| **Training** | BPTT through full sequence | BPTT through full sequence | Backprop in predictor only; no BPTT |

The key distinction from SSMs: SSMs compress the entire sequence into a single shared state vector -- all tokens mix into one representation. MathBrain maintains **per-slot independent** EMA states. This avoids the information bottleneck at the cost of O(V) rather than O(1) state size.

The tradeoff: MathBrain's cost scales with vocabulary/slot count O(V) rather than sequence length O(N). For bounded vocabularies, this is constant. It excels at continual, interpretable, bounded-state operation.

## Results

All results are train-set accuracy (memorization) on TinyStories subsets. The claim is not SOTA on broad benchmarks, but viability of bounded-state encoding as a replacement for attention-based context aggregation.

### v2: Gold-Teacher Sleep (A-only, direct training)

| Setting | N | D | CP_RANK | Accuracy | Note |
|---------|---|---|---------|----------|------|
| tinystories_10 | 4 | 8 | 384 | **99.41%** | Near-perfect; ~10 errors are genuine ambiguity |
| tinystories_60 | 4 | 8 | 384 | **99.06%** | Capacity holds at 6x corpus |
| tinystories_200 | 4 | 8 | 384 | 71.14% | Context window too short (half-life ~14 steps) |
| tinystories_200 | 16 | 32 | 384 | **97.84%** | Extending EMA scales solves context bottleneck |
| tinystories_200 | 8(sparse) | 32 | 384 | **97.13%** | Sparse RHO: fewer scales, wider coverage |

### φ Encoding Analysis (tinystories_60, N=D=32, CP_RANK=32)

| P matrix | Singular values | φ effective rank | Accuracy |
|----------|----------------|-----------------|----------|
| Random Gaussian | Non-uniform (cond=50) | 25.9 | 97.96% |
| Ortho × Gaussian SV | Non-uniform (cond=50) | 26.2 | **98.08%** |
| Hadamard × Gaussian SV | Non-uniform (cond=50) | 24.9 | 97.90% |
| Random Orthogonal | Uniform (cond=1) | 21.7 | 96.93% |
| Hadamard | Uniform (cond=1) | 20.5 | 96.59% |
| No P (D=N, chaos only) | N/A | ~18 | 87.22% |

**Key finding:** P's non-uniform singular values create multi-resolution features that increase φ effective rank by ~25%. This is the dominant factor in encoding quality.

| Folding config | shift | L | φ effective rank | Accuracy |
|----------------|-------|---|-----------------|----------|
| P + shift, L=1 (≈RFF) | yes | 1 | 24.9 | 97.42% |
| P + shift, L=3 (default) | yes | 3 | 25.9 | 97.96% |
| P + shift, L=N=32 | yes | 32 | 26.8 | **98.60%** |
| P + no_shift, L=1 | no | 1 | 24.9 | 97.48% |
| P + no_shift, L=N=32 | no | 32 | 26.7 | 98.26% |

**Key finding:** A single cos(α·P@Q) (Random Fourier Features) already achieves 97.5%. Chaos folding with L=N adds ~1% by deepening the nonlinear kernel. Cyclic shift contributes ~0.3% on top of that.

### v1: Wake-Sleep with B (original architecture)

| Setting | Accuracy | Note |
|---------|----------|------|
| tinystories_10, A+B | 92-93% | Combined dual memory |
| tinystories_10, A-only after sleep | 84-86% | B as teacher, A as student |
| tinystories_11_20, wake-only B | 87.0% | Pure online learning |

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

Bypassing B entirely and training A directly on gold (context, next_word) pairs produces dramatically better results than the B→A wake-sleep pipeline. This reveals that the encoding pipeline (EMA + CosineChaos + HashRetina) is itself a powerful feature extractor -- B's noisy teacher signal was the bottleneck, not A's capacity.

### 2. φ Encoding: What Makes It Work

We systematically decomposed the φ encoding pipeline to understand each component's contribution:

- **P matrix (linear mixing):** Essential. Provides cross-scale coupling that cyclic shift cannot efficiently achieve. Non-uniform singular values (condition number ~50) create multi-resolution features, improving φ effective rank by ~25% over uniform orthogonal P.
- **Cosine-chaos folding (nonlinear expansion):** L=N iterations give the best results (+1% over single cos). The chaos dynamics amplify micro-differences in Q into distinguishable φ features. Weak chaos (α=2.0, λ≈+0.16) outperforms exact criticality (λ=0).
- **Cyclic shift (inter-dimension coupling):** Small contribution (~0.3%) when P already provides full mixing. Can be removed with minimal impact.
- **Baseline: cos(α·P@Q) (Random Fourier Features):** Already achieves 97.5%. This is the irreducible core -- a single nonlinear projection of the mixed EMA state.

### 3. Width Scaling: N and D

When accuracy drops on larger corpora, the fix is extending EMA timescales (N), not adding layers. Sparse RHO distributions (few scales, wide coverage) are more parameter-efficient than dense distributions.

### 4. Tree-Parallel Wake Training (v1)

**Script:** `experiments/run_parallel_wake.py`

Because wake writes are local edge updates, they can be parallelized. The trainer freezes B, farms out shards in parallel, then tree-merges sparse deltas.

## Current Status

This is an active research project.

**Working:**
- Core encoding pipeline (HashRetina → EMA → CosineChaos) validated as O(1) position encoder
- PyTorch trainer with CUDA/MPS/CPU support, configurable N/D/CP_RANK/RHO
- 97-99% train accuracy on TinyStories subsets (10-200 stories)
- φ encoding mechanism understood: P mixing + nonlinear chaos folding + multi-resolution singular values

**Open directions:**
- Replacing the bilinear voter with stronger predictors (MLP, small Transformer) to test encoding quality
- Online self-distillation: compressing φ (context memory) into parameters during inference
- Scaling to larger corpora and evaluating generalization (train/test split)
- Optimizing P as a learnable component (Cayley-parameterized orthogonal + learnable singular values)
- Systematic baselines against n-gram / LSTM / Transformer at matched parameter counts

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
