#!/usr/bin/env python3
"""Train MathBrain A module via gold-teacher dream sleep.

Usage:
    # Default: N=4, D=8 on tinystories_10
    python experiments/train_a.py

    # Scaled: N=16, D=32 on tinystories_200
    python experiments/train_a.py --corpus datasets/tinystories_200.txt --N 16 --D 32 --epochs 640

    # Custom RHO (sparse):
    python experiments/train_a.py --rho 0.3,0.85,0.97,0.995,0.999,0.9997 --D 32 --epochs 640

    # Adjust CP_RANK:
    python experiments/train_a.py --N 8 --D 32 --cp-rank 192 --epochs 640

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
    return tuple(rho.tolist())


def print_rho(rho):
    rho_arr = np.array(rho)
    hl = np.log(0.5) / np.log(rho_arr)
    print(f"  RHO ({len(rho)} scales):")
    for i, (r, h) in enumerate(zip(rho_arr, hl)):
        print(f"    [{i}] rho={r:.6f}  half-life={h:.1f} steps")


def run_single(corpus, rho, D, cp_rank, epochs, batch_size, device,
               save_path=None, log_interval=20, lr=0.01,
               predictor='linear', d_model=128, n_heads=4, n_layers=2,
               phi_mode='chaos', learnable_P=False,
               cache_dir='cache/', corpus_path=None,
               storage_dtype='fp32'):
    N = len(rho)
    hl = np.log(0.5) / np.log(np.array(rho))

    print(f"\n{'='*60}")
    print(f"N={N}, D={D}, CP_RANK={cp_rank}, epochs={epochs}, predictor={predictor}")
    print(f"  max half-life={hl[-1]:.0f}, phi_mode={phi_mode}")
    if predictor == 'transformer':
        print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    print_rho(rho)
    print(f"{'='*60}")

    cfg = MathBrainConfig(N=N, RHO=tuple(rho), D_PHI=D, CP_RANK=cp_rank)
    model = MathBrain(cfg)
    trainer = MathBrainTrainer(model, device=device, predictor=predictor,
                               d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                               phi_mode=phi_mode, learnable_P=learnable_P)

    result = trainer.fit(corpus, epochs=epochs, batch_size=batch_size,
                         log_interval=log_interval, lr=lr,
                         cache_dir=cache_dir, corpus_path=corpus_path,
                         storage_dtype=storage_dtype)
    eval_result = trainer.evaluate(corpus, cache_dir=cache_dir,
                                       corpus_path=corpus_path)
    print(f"  Accuracy: {eval_result['correct']}/{eval_result['total']} "
          f"= {eval_result['accuracy']:.2f}%")

    if save_path:
        trainer.save(save_path)

    return {'N': N, 'D': D, 'cp_rank': cp_rank, 'epochs': epochs,
            'max_hl': hl[-1], 'predictor': predictor, **result, **eval_result}


def main():
    parser = argparse.ArgumentParser(
        description='Train MathBrain A module via gold-teacher dream sleep')
    parser.add_argument('--corpus', default='datasets/tinystories_10.txt',
                        help='Path to corpus file (one sentence per line)')
    parser.add_argument('--N', type=int, default=None,
                        help='EMA timescales (ignored if --rho is set)')
    parser.add_argument('--D', type=int, default=None,
                        help='Phi dimension (default: 2*N)')
    parser.add_argument('--rho', type=str, default=None,
                        help='Comma-separated RHO values (overrides --N)')
    parser.add_argument('--cp-rank', type=int, default=384,
                        help='Low-rank factorization rank (default: 384)')
    parser.add_argument('--epochs', type=int, default=320)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--device', default='auto',
                        help='auto, cuda, mps, or cpu')
    parser.add_argument('--save', default=None, metavar='PATH',
                        help='Save trained model to file')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep N/D configs: (4,8), (8,16), (12,24), (16,32)')
    # Predictor options
    parser.add_argument('--predictor', default='linear',
                        choices=['linear', 'transformer'],
                        help='Predictor type (default: linear)')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Transformer d_model (default: 128)')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Transformer attention heads (default: 4)')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Transformer encoder layers (default: 2)')
    parser.add_argument('--phi-mode', default='chaos',
                        choices=['chaos', 'raw'],
                        help='Phi encoding: chaos (CosineChaos) or raw (alpha_scale*Q)')
    parser.add_argument('--learnable-P', action='store_true',
                        help='Make P matrix learnable (Cayley parametrization)')
    parser.add_argument('--cache-dir', default='cache/',
                        help='Directory for preprocessed binary cache (default: cache/)')
    parser.add_argument('--storage-dtype', choices=['fp32', 'bf16'],
                        default='fp32',
                        help='Cache storage precision: fp32 (default) or bf16 (~50%% space saving)')
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    print(f"Corpus: {args.corpus}, {len(corpus)} sentences")

    if args.sweep:
        configs = [(4, 8), (8, 16), (12, 24), (16, 32)]
        results = []
        for N, D in configs:
            rho = make_rho(N)
            r = run_single(corpus, rho, D, args.cp_rank, args.epochs,
                           args.batch_size, args.device,
                           log_interval=args.log_interval,
                           cache_dir=args.cache_dir,
                           corpus_path=args.corpus)
            results.append(r)

        print(f"\n{'='*60}")
        print("Summary:")
        print(f"{'N':>4} {'D':>4} {'rank':>5} {'epochs':>6} {'max_hl':>7} | "
              f"{'accuracy':>8} {'correct':>7}/{'total':<5} {'loss':>10}")
        print("-" * 75)
        for r in results:
            print(f"{r['N']:4d} {r['D']:4d} {r['cp_rank']:5d} "
                  f"{r['epochs']:6d} {r['max_hl']:7.0f} | {r['accuracy']:7.2f}% "
                  f"{r['correct']:7d}/{r['total']:<5d} "
                  f"{r['best_loss']:10.6f}")
    else:
        # 确定 RHO
        if args.rho:
            rho = tuple(float(x) for x in args.rho.split(','))
            N = len(rho)
        else:
            N = args.N or 4
            rho = make_rho(N)

        D = args.D or (2 * N)
        run_single(corpus, rho, D, args.cp_rank, args.epochs,
                   args.batch_size, args.device, args.save,
                   log_interval=args.log_interval,
                   lr=args.lr,
                   predictor=args.predictor,
                   d_model=args.d_model,
                   n_heads=args.n_heads,
                   n_layers=args.n_layers,
                   phi_mode=args.phi_mode,
                   learnable_P=args.learnable_P,
                   cache_dir=args.cache_dir,
                   corpus_path=args.corpus,
                   storage_dtype=args.storage_dtype)


if __name__ == '__main__':
    main()
