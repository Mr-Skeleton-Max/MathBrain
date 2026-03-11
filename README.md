# MathBrain

**Bounded-State Categorical Voting for Online Sequence Learning**

A white-box, neuroscience-inspired architecture for online sequence prediction with complementary learning systems. MathBrain treats sequence modeling as bounded-state categorical voting driven by explicit memories, not black-box function approximation over growing history.

## Key Properties

- **Constant-time inference**: After bounded-state compression, inference time and memory are O(1) with respect to processed sequence length
- **Continual online learning**: No replay buffer or growing context window required
- **Fully interpretable**: Every prediction can be traced to specific memory entries and voting evidence
- **Dual memory system**: Fast short-term memory (B) + slow long-term knowledge (A), inspired by hippocampal-neocortical complementary learning systems

## Architecture Overview

```
Token → Hash Retina → Q (EMA state) → φ (distance features)
                                          ↓
                              B (sparse short-term memory)
                              A (low-rank long-term knowledge)
                                          ↓
                              Categorical voting → Prediction
```

| Component | Role | Analogy |
|-----------|------|---------|
| Hash Retina | Deterministic n-gram lexical coding | Sensory encoding |
| Q State | Multi-timescale EMA compression (N=4 scales) | Temporal context |
| φ Encoder | Bounded positional feature map (cos+sin chaos) | Distance representation |
| B Memory | Sparse hash-table voting memory, fast delta-rule write | Hippocampus |
| A Knowledge | Low-rank slice-wise factorization, slow consolidation | Neocortex |
| Sleep | Pseudorehearsal transfer from B → A | Memory consolidation |

## Performance

Early experimental results on TinyStories and PTB:

| Setting | Accuracy |
|---------|----------|
| Wake-only (B+A) on tinystories_11_20 | 87.0% top-1 |
| A-only after sleep on tinystories_10 | 84-86% top-1 |
| A+B combined on tinystories_10 | 92-93% top-1 |
| First 200 lines of PTB | 85% top-1 |
| 230-word memorization test | 100% top-1 |

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/MathBrain.git
cd MathBrain
pip install -e .
```

### Train on demo data

```bash
# Single wake-sleep cycle on 10 TinyStories
python train.py --corpus datasets/tinystories_10.txt --mode cycle --cycles 1

# Multi-cycle training on 20 stories
python train.py --corpus datasets/tinystories_20.txt --mode cycle --cycles 5

# Wake-only (no sleep consolidation)
python train.py --corpus datasets/tinystories_10.txt --mode wake
```

### Python API

```python
from mathbrain import MathBrain, MathBrainConfig

config = MathBrainConfig()
model = MathBrain(config)

# Online learning: feed tokens one at a time
for token in tokens:
    prediction = model.forward(token)
```

## Project Structure

```
MathBrain/
├── train.py                 # Training CLI
├── mathbrain/               # Core package
│   ├── config.py            # All hyperparameters
│   ├── model.py             # Main MathBrain class
│   ├── retina.py            # Hash-based lexical coding
│   ├── q_state.py           # Multi-timescale EMA state
│   ├── phi_encoder.py       # Distance feature encoding
│   ├── phi_encoder_chaos.py # Cosine chaos encoder
│   ├── b_memory.py          # Short-term B memory variants
│   ├── a_knowledge_v2.py    # Long-term A knowledge (slice-wise decomposition)
│   ├── sleep_v2.py          # Wake-sleep consolidation
│   └── ...
├── datasets/                # Demo datasets
├── experiments/             # Reproducible experiments
├── paper/                   # ArXiv draft
└── docs/                    # Specification (Chinese)
```

## Experiments

The `experiments/` directory contains scripts for reproducing key results:

- **`trace_tiny_wake_sleep.py`** — End-to-end wake-sleep demo with accuracy tracing
- **`run_parallel_wake.py`** — Parallel tree-approximation wake training
- **`step1_build_dream_data.py`** + **`step2_sleep_onehot.py`** — Dream-sleep pipeline
- **`exp_sleep_ce.py`** — Cross-entropy sleep training
- **`step12_two_layer.py`** — Two-layer architecture experiments

## Requirements

- Python >= 3.8
- NumPy >= 1.21.0
- SciPy >= 1.7.0

Optional:
- [MLX](https://github.com/ml-explore/mlx) >= 0.5.0 (Apple Silicon acceleration)
- [PyTorch](https://pytorch.org/) >= 2.0.0 (GPU acceleration)

## Citation

```bibtex
@article{mathbrain2026,
  title={Bounded-State Categorical Voting for Online Sequence Learning:
         A White-Box Alternative to Length-Growing Black-Box Models},
  author={Anonymous},
  year={2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
