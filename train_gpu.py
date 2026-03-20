#!/usr/bin/env python3
"""MathBrain GPU 训练 CLI

用法:
  # 训练
  python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 100

  # 训练 + 评估
  python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 100 --eval-every 20

  # 自定义参数
  python train_gpu.py --corpus data.txt --epochs 200 --batch-size 8192 --lr 0.005
"""
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from MathBrain.config import MathBrainConfig
from MathBrain.trainer import MathBrainTrainer


def main():
    parser = argparse.ArgumentParser(description='MathBrain GPU Training')
    parser.add_argument('--corpus', required=True, help='训练语料文件')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.01, help='最大学习率')
    parser.add_argument('--min-lr', type=float, default=0.0,
                        help='最终学习率 (默认: lr/25000)')
    parser.add_argument('--warmup-pct', type=float, default=0.05,
                        help='warmup 占比 (默认: 0.05 = 5%%)')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--eval-every', type=int, default=0,
                        help='每 N epoch 评估 (0=仅结束时)')
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--D', type=int, default=32, help='D_PHI')
    parser.add_argument('--rank', type=int, default=32, help='CP_RANK')
    parser.add_argument('--save', type=str, help='保存模型路径')
    args = parser.parse_args()

    corpus = [l.strip() for l in open(args.corpus) if l.strip()]

    cfg = MathBrainConfig(
        N=args.N,
        RHO=(0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999)[:args.N],
        D_PHI=args.D,
        CP_RANK=args.rank,
    )

    trainer = MathBrainTrainer(cfg, device='cuda')
    result = trainer.fit(
        corpus,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_pct=args.warmup_pct,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        eval_corpus=corpus if args.eval_every > 0 else None,
        eval_every=args.eval_every,
    )

    # Final evaluation
    print("\n── Final Evaluation ──")
    trainer.evaluate(corpus)

    if args.save:
        trainer.save(args.save)
        print(f"\nModel saved to {args.save}")


if __name__ == '__main__':
    main()
