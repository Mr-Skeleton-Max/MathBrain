#!/usr/bin/env python3
"""Quick training benchmark: Triton vs Eager

用法: python experiments/train_bench.py --corpus datasets/tinystories_200.txt --epochs 10
"""
import sys, time, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MathBrain import MathBrainConfig, MathBrainTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='datasets/tinystories_200.txt')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    cfg = MathBrainConfig(
        N=8,
        RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999),
        D_PHI=32,
        CP_RANK=32,
    )

    corpus = [l.strip() for l in open(args.corpus) if l.strip()]
    print(f"Corpus: {len(corpus)} sentences")

    trainer = MathBrainTrainer(cfg, device='auto')
    result = trainer.fit(
        corpus,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_interval=1,
    )
    trainer.evaluate(corpus)


if __name__ == '__main__':
    main()
