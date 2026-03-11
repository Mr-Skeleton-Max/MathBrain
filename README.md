# MathBrain

**Bounded-State Categorical Voting for Online Sequence Learning**

MathBrain is an experimental sequence prediction architecture that does not store or attend over token history. Instead, it compresses all context into a bounded EMA state and predicts the next token through explicit memory voting -- achieving O(1) inference, online learning, and full interpretability by design.

> **TL;DR**: Transformers grow state with sequence length. MathBrain doesn't. It treats next-token prediction as a voting problem over two explicit memory systems (fast B + slow A), with constant memory regardless of how many tokens it has seen.

## Key Ideas

- **Prediction is voting.** Every next-token prediction is a categorical competition. The architecture is designed around this from the ground up.
- **History should be compressed, not stored.** Multi-timescale EMA states capture "when did I last see this pattern" without keeping the full sequence.
- **Voting tolerates compression.** You don't need exact replay to rank the correct candidate above competitors -- sufficient evidence is enough.
- **Two memories, one goal.** Fast episodic memory B (hippocampus) writes instantly; slow structured memory A (neocortex) consolidates during sleep. A can take over most of B's function.
- **Everything is inspectable.** Every prediction traces back to specific memory entries, source slots, and feature vectors. No black box.

## How It Works

Consider predicting the next word after *"the cat sat on the"*. A transformer must attend over all previous tokens. MathBrain works differently:

```
"the cat sat on the"

  1. Hash each word into sparse slot IDs           "the" → {3712, 8891, ...}
     (deterministic character n-gram hashing)       "cat" → {1204, 5567, ...}

  2. Compress history into bounded EMA state        Q[3712] = [0.95, 0.82, 0.41, 0.12]
     (4 timescales: fast → slow decay)                        ← recent ──── distant →

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
            (n-gram → slots)  │  4 decay rates: ρ₁..ρ₄     │     + Q values
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

### Why Two Memories?

This is the core epistemic division:

| | **B** (Short-Term) | **A** (Long-Term) |
|---|---|---|
| **What it stores** | Exact co-activation traces | Generalized factored structure |
| **How it learns** | Online delta-rule, one pass | Offline sleep consolidation |
| **Generalization** | Weak (memorization) | Strong (pattern extraction) |
| **Speed** | Instant write | Slow batch optimization |
| **Biological analog** | Hippocampus | Neocortex |

The most important empirical finding: **A can take over most of B's predictive function.** After sleep consolidation, A-only reaches 84-86% while A+B reaches 92-93%. Sleep is not cosmetic -- it transfers predictive structure into compressed long-term knowledge.

### Wake-Sleep Learning

```
  ┌─── Wake ────────────────────────────────────────────┐
  │  Online, one token at a time                        │
  │  B writes exact traces: Δb = η · gap · φ̂           │
  │  gap = 1 - (γ·score_A + score_B)                   │
  │  → self-limiting: stops writing when A+B is enough  │
  └─────────────────────────┬───────────────────────────┘
                            │ B accumulates traces
                            ▼
  ┌─── Sleep ───────────────────────────────────────────┐
  │  Offline, batch optimization (AdamW)                │
  │  Target: y = γ·A_old + B  (blend old knowledge      │
  │           with new traces)                          │
  │  Distill exact traces → generalized structure       │
  │  Optional rehearsal to prevent catastrophic forget   │
  └─────────────────────────────────────────────────────┘
```

## How It Differs from Transformers

| | **Transformer** | **MathBrain** |
|---|---|---|
| **Inference cost** | O(L) or O(L²) in sequence length | **O(1)** after EMA saturation |
| **KV cache** | Grows with every token | No cache; bounded EMA state |
| **Online learning** | Requires full retraining | Native: wake writes are instant |
| **Interpretability** | Post-hoc attention maps | Every vote is an explicit memory trace |
| **Memory model** | Implicit in weights | Explicit dual system with inspectable entries |
| **Training** | Backprop through time | No BPTT; local delta-rule + offline sleep |

The tradeoff: MathBrain's prediction cost scales with vocabulary size O(V) rather than sequence length. It excels at continual, interpretable, bounded-state operation -- not at replacing large-scale language models on broad benchmarks.

## Early Results

| Setting | Accuracy | Note |
|---------|----------|------|
| tinystories_10, A+B | 92-93% | Combined dual memory |
| tinystories_11_20, wake-only | 87.0% | Pure online learning |
| tinystories_11_20, A-only after sleep | 85.6% | Long-term module alone |
| tinystories_10, A-only | 84-86% | After 10 wake-sleep cycles |
| PTB first 200 lines | 85% | Standard benchmark |
| 230-word memorization test | 100% | Perfect recall |
| Zero-shot: dog → wolf | positive | Ecological generalization via shared slots |

These are early results on small-scale corpora. The claim is not SOTA on broad benchmarks, but viability of a fundamentally different regime: bounded state, voting-style memory, online learning, and full interpretability.

## Quick Start

### Install

```bash
git clone https://github.com/Mr-Skeleton-Max/MathBrain.git
cd MathBrain
pip install -e .
```

### Train

```bash
# Single wake-sleep cycle on 10 TinyStories
python train.py --corpus datasets/tinystories_10.txt --mode cycle --cycles 1

# Multi-cycle training
python train.py --corpus datasets/tinystories_20.txt --mode cycle --cycles 5

# Wake-only (no sleep consolidation)
python train.py --corpus datasets/tinystories_10.txt --mode wake
```

### Python API

```python
from mathbrain import MathBrain, MathBrainConfig

config = MathBrainConfig()
model = MathBrain(config)

# Online learning: feed tokens one by one
for token in tokens:
    prediction = model.forward(token)  # learns and predicts simultaneously
```

## Project Structure

```
MathBrain/
├── train.py                     # Training CLI (wake, sleep, cycle modes)
├── mathbrain/                   # Core package
│   ├── model.py                 #   Main MathBrain class
│   ├── config.py                #   All hyperparameters
│   ├── retina.py                #   Hash-based lexical coding
│   ├── q_state.py               #   Multi-timescale EMA state
│   ├── phi_encoder_chaos.py     #   Cosine-chaos feature map
│   ├── b_memory_hashtable_v2.py #   Short-term sparse memory
│   ├── a_knowledge_v2.py        #   Long-term slice-wise factorization
│   ├── sleep_v2.py              #   Wake-sleep consolidation (PyTorch)
│   ├── sleep_v2_mlx.py          #   Wake-sleep consolidation (MLX)
│   └── docs/                    #   Detailed architecture docs (Chinese)
├── datasets/                    # Demo corpora + test sets
├── experiments/                 # Reproducible experiment scripts
├── paper/                       # ArXiv draft (.tex + .pdf)
└── docs/                        # Full mathematical specification (Chinese)
```

## Experiments

The `experiments/` directory contains three lines of investigation beyond the basic wake-sleep loop.

### 1. Tree-Parallel Wake Training

**Script:** `run_parallel_wake.py` | **Insight: O(k log N) training**

Standard wake training processes tokens sequentially -- O(N) in corpus length. But because wake writes are **local edge updates** (each source-target pair is independent), they can be parallelized:

```
  Corpus: [sent₁, sent₂, ..., sentₙ]
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Shard₁       Shard₂      Shard₃      ← parallel wake on frozen B snapshot
    (local Δb)   (local Δb)  (local Δb)
        └───────────┼───────────┘
                    ▼
            Tree-merge Δb               ← weighted sum on coincident slot pairs
                    │
                    ▼
              Update B globally
```

The trainer freezes B at the start of each round, farms out shards in parallel, then tree-merges the sparse delta updates. Per-round cost drops from O(N) to O(k log N) where k = shard size. This is a genuine architectural property of the sparse voting update -- not an approximation hack.

```bash
# 16 rounds of tree-parallel wake on 20 stories
python experiments/run_parallel_wake.py \
    --corpus datasets/tinystories_20.txt --rounds 16 --backend mlx

# Full wake-sleep with parallel wake
python experiments/run_parallel_wake.py \
    --corpus datasets/tinystories_10.txt --rounds 16 --cycles 5 --wake-sleep --backend mlx
```

### 2. Dream Sleep: Can A Learn Without Seeing the Original Text?

**Scripts:** `step1_build_dream_data.py` → `step2_sleep_onehot.py`, `export_dream_dataset.py`, `train_a_from_qdist.py`, `exp_sleep_ce.py`

**Core insight:** Since each voting edge in MathBrain is independent, sleep does not need the original corpus. If we can generate *any* Q-state activations and ask the current A+B system for its top-1 response, that is sufficient supervision for A to learn.

The dream-sleep pipeline:
1. Generate synthetic Q-state activations (random slot groups driven through EMA, or replay corpus structure)
2. Query the current A+B teacher for its top-1 prediction at each state (Dirichlet-sharpened one-hot target)
3. Train A to match these one-hot targets

**Key finding:** What A actually learns during sleep is a **low-rank factorization of B's edge map** -- not a distribution over next tokens. Attempting to fit the full soft distribution (rather than the sharpened top-1) causes catastrophic performance collapse. This reveals that sleep's primary function is **improving B's signal-to-noise ratio** through compression: A captures the dominant pattern, and B only needs to store the residual.

```bash
# Step 1: wake + export dream dataset (Q-states → teacher scores)
python experiments/step1_build_dream_data.py

# Step 2: train A from one-hot teacher targets
python experiments/step2_sleep_onehot.py

# Cross-entropy sleep variant
python experiments/exp_sleep_ce.py
```

**Open question:** In theory, B should continue to improve accuracy on top of A (storing what A missed). In practice, the residual B improvement after sleep is currently small. Better designs for the B-on-top-of-A residual path remain an active research direction.

### 3. Multi-Layer MathBrain: From Linear Classifier to MLP

**Scripts:** `step12_two_layer.py`, `step12b_two_layer_fix.py` | **Result: 93% → 97%**

**Motivation:** The current single-layer MathBrain is essentially a **super linear classifier** -- all computation happens in the interaction space between source features and target embeddings. It cannot form abstract intermediate representations. The natural extension: stack layers, like going from a single linear layer to an MLP.

```
Layer 1: token → Q₁ → φ₁ → B₁ → prediction distribution P₁
                                        │
                              top-k winners + actual word (residual connection)
                                        │
                                        ▼
Layer 2: P₁ → Q₂ → φ₂ → B₂ → final prediction
```

Three variants tested:
- **Hard attention**: L2 sees only L1's winner prediction
- **Residual**: L2 sees actual word + L1's winner (best)
- **Residual + soft**: L2 sees actual word + L1's top-3 softmax

| Variant | Accuracy (tinystories_10) |
|---------|--------------------------|
| Single-layer baseline | ~93% |
| Two-layer, hard attention | ~91% |
| Two-layer, residual | **~97%** |
| Two-layer, residual + soft | ~96% |

The residual variant works best: Layer 2 processes the *combination* of what actually happened and what Layer 1 predicted would happen. This is the most promising direction for scaling MathBrain -- each additional layer adds a new level of abstraction while preserving the voting-based, interpretable architecture.

## Current Status

This is an active research project in early-to-mid stage:

- Core architecture (hash retina, EMA state, phi encoding, B memory, A knowledge, wake-sleep) -- implemented and working
- Single-layer system reaches 87-93% next-token accuracy on small corpora (TinyStories, PTB)
- Multi-layer extension shows 93% → 97% improvement -- most promising scaling direction
- Dream sleep (corpus-free consolidation) -- working but residual B improvement still small
- Large-scale evaluation (10K+ stories, systematic baselines vs n-gram/LSTM/Transformer) -- not yet done
- Abstraction beyond linear interaction space -- open problem (multi-layer is the current approach)

The architecture is validated at small scale. The open question is how far it can go.

## Paper

For the full theoretical motivation -- why categorical voting is a viable first principle, why bounded state becomes possible once you give up exact history reconstruction, and how memory-as-trigger differs from memory-as-retrieval -- please read the paper:

> **[Bounded-State Categorical Voting for Online Sequence Learning: A White-Box Alternative to Length-Growing Black-Box Models](paper/mathbrain_arxiv_draft.pdf)**

The paper covers the mathematical formulation in detail, including the complexity analysis, the wake update derivation, the sleep consolidation objective, and the biological correspondence argument. The README gives the intuition; the paper gives the proofs.

For the research journey that led to this architecture -- from BubbleStream to GPT-2 memory analysis to the categorical voting insight to EMA encoding -- see [docs/research_log.md](docs/research_log.md).

## Requirements

- Python >= 3.8
- NumPy >= 1.21.0
- SciPy >= 1.7.0

Optional accelerators:
- [MLX](https://github.com/ml-explore/mlx) >= 0.5.0 (Apple Silicon)
- [PyTorch](https://pytorch.org/) >= 2.0.0 (CUDA)

## A Note on Code Quality

This codebase grew out of a solo research project. The author is a researcher first and not a professional software engineer -- the code works and the experiments are reproducible, but it is not as clean or well-organized as a production library. Apologies for any rough edges. If something is confusing, please open an issue and I'll be happy to explain.

## Contributing

This is an active research program with many open directions. I would love to collaborate:

- **Discussions**: If you find the categorical-voting premise interesting (or flawed!), let's talk. Open an issue or reach out directly.
- **Experiments**: Scaling to larger corpora, systematic baselines against n-gram/LSTM/Transformer, multi-layer architectures -- all need more hands.
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
