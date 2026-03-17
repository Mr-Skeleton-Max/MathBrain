#!/usr/bin/env python3
"""Train MathBrain A module via gold-teacher dream sleep.

N (EMA timescales) and D (phi dimension) must be scaled together:
- N controls how many decay rates the EMA uses; more scales → longer context
- D controls the phi encoding dimension; D ≈ 2*N keeps information density
- RHO is auto-generated: half-lives from ~0.6 to max_halflife, log-spaced

Usage:
    # Default: N=4, D=8 on tinystories_10
    python experiments/train_a.py

    # Scaled: N=16, D=32 on tinystories_200
    python experiments/train_a.py --corpus datasets/tinystories_200.txt --N 16 --D 32 --epochs 640

    # Sweep multiple N/D configs
    python experiments/train_a.py --corpus datasets/tinystories_200.txt --sweep

    # Save trained model
    python experiments/train_a.py --N 16 --D 32 --epochs 640 --save model.pt
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.trainer import MathBrainTrainer


def load_corpus(filepath: str) -> list[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def make_rho(N: int, min_halflife: float = 0.6, max_halflife: float = None) -> tuple:
    """Generate N EMA decay rates with log-spaced half-lives.

    half_life = log(0.5) / log(rho)  =>  rho = 0.5^(1/half_life)
    """
    if max_halflife is None:
        max_halflife = 14.0 * (15.0 ** ((N - 4) / 12.0))

    half_lives = np.geomspace(min_halflife, max_halflife, N)
    rho = np.clip(np.power(0.5, 1.0 / half_lives), 0.05, 0.9999)
    print(f"  RHO ({N} scales):")
    for i, (r, hl) in enumerate(zip(rho, half_lives)):
        print(f"    [{i}] rho={r:.6f}  half-life={hl:.1f} steps")
    return tuple(rho.tolist())


def make_config(N: int, D: int, cp_rank: int = 384) -> MathBrainConfig:
    return MathBrainConfig(N=N, RHO=make_rho(N), D_PHI=D, CP_RANK=cp_rank)


def run_single(corpus, N, D, cp_rank, epochs, batch_size, device, save_path=None):
    print(f"\n{'='*60}")
    print(f"N={N}, D={D}, CP_RANK={cp_rank}, epochs={epochs}")
    print(f"{'='*60}")

    cfg = make_config(N, D, cp_rank)
    model = MathBrain(cfg)
    trainer = MathBrainTrainer(model, device=device)

    result = trainer.fit(corpus, epochs=epochs, batch_size=batch_size)
    eval_result = trainer.evaluate(corpus)
    print(f"  Accuracy: {eval_result['correct']}/{eval_result['total']} "
          f"= {eval_result['accuracy']:.2f}%")

    if save_path:
        trainer.save(save_path)

    return {'N': N, 'D': D, 'cp_rank': cp_rank, 'epochs': epochs,
            **result, **eval_result}


def main():
    parser = argparse.ArgumentParser(
        description='Train MathBrain A module via gold-teacher dream sleep')
    parser.add_argument('--corpus', default='datasets/tinystories_10.txt',
                        help='Path to corpus file (one sentence per line)')
    parser.add_argument('--N', type=int, default=None, help='EMA timescales')
    parser.add_argument('--D', type=int, default=None,
                        help='Phi dimension (default: 2*N)')
    parser.add_argument('--cp-rank', type=int, default=384,
                        help='Low-rank factorization rank')
    parser.add_argument('--epochs', type=int, default=320)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='auto',
                        help='auto, cuda, mps, or cpu')
    parser.add_argument('--save', default=None, metavar='PATH',
                        help='Save trained model to file')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep N/D configs: (4,8), (8,16), (12,24), (16,32)')
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    print(f"Corpus: {args.corpus}, {len(corpus)} sentences")

    if args.sweep:
        configs = [(4, 8), (8, 16), (12, 24), (16, 32)]
        results = []
        for N, D in configs:
            r = run_single(corpus, N, D, args.cp_rank, args.epochs,
                           args.batch_size, args.device)
            results.append(r)

        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'N':>4} {'D':>4} {'rank':>5} {'epochs':>6} | "
              f"{'accuracy':>8} {'correct':>7}/{'total':<5} {'loss':>10}")
        print("-" * 65)
        for r in results:
            print(f"{r['N']:4d} {r['D']:4d} {r['cp_rank']:5d} "
                  f"{r['epochs']:6d} | {r['accuracy']:7.2f}% "
                  f"{r['correct']:7d}/{r['total']:<5d} "
                  f"{r['best_loss']:10.6f}")
    else:
        N = args.N or 4
        D = args.D or (2 * N)
        run_single(corpus, N, D, args.cp_rank, args.epochs,
                   args.batch_size, args.device, args.save)


if __name__ == '__main__':
    main()
