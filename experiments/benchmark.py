#!/usr/bin/env python3
"""MathBrain 公平对比实验：Linear vs Transformer × Chaos vs Raw × 数据规模

统一条件：
- 输出路径: slot_scores → word_proj → CrossEntropy
- 输入: HashRetina → EMA → phi (chaos 或 raw)
- 优化器: AdamW + CosineAnnealing

已知结果 (tinystories_10, 200 epochs, N=8 sparse):
  linear + chaos  lr=0.01  → 99.4%
  linear + raw    lr=0.01  → 95.2%
  transformer + chaos lr=0.003 → 99.4%
  transformer + raw   lr=0.001 → 99.4%

待验证：在 tinystories_60 和 tinystories_200 上是否保持？

用法:
  python experiments/benchmark.py --corpus datasets/tinystories_10.txt
  python experiments/benchmark.py --corpus datasets/tinystories_60.txt --epochs 480
  python experiments/benchmark.py --corpus datasets/tinystories_200.txt --epochs 640
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.trainer import MathBrainTrainer


# ── 默认 RHO ──────────────────────────────────────────────
SPARSE_RHO_8 = (0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999)
D_CHAOS = 32   # CosineChaos 输出维度
N = 8          # EMA 尺度数


def run(corpus, predictor, phi_mode, lr, cp_rank=32, learnable_P=False,
        d_model=128, n_heads=4, n_layers=2, epochs=200, batch_size=128):
    """单次实验。返回 (accuracy, best_loss, elapsed)。"""
    D_eff = N if phi_mode == 'raw' else D_CHAOS
    cfg = MathBrainConfig(N=N, RHO=SPARSE_RHO_8, D_PHI=D_eff, CP_RANK=cp_rank)
    model = MathBrain(cfg)
    trainer = MathBrainTrainer(
        model, device='auto', predictor=predictor,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        phi_mode=phi_mode, learnable_P=learnable_P,
    )
    t0 = time.time()
    result = trainer.fit(corpus, epochs=epochs, batch_size=batch_size,
                         lr=lr, log_interval=max(1, epochs // 5))
    ev = trainer.evaluate(corpus)
    elapsed = time.time() - t0
    return ev['accuracy'], result['best_loss'], elapsed


# ── 实验配置 ──────────────────────────────────────────────
# (name, predictor, phi_mode, lr, cp_rank, learnable_P, d_model, n_heads, n_layers)
EXPERIMENTS = [
    ("linear+chaos",       "linear",      "chaos", 0.01,  32, False, 128, 4, 2),
    ("linear+raw",         "linear",      "raw",   0.01,  32, False, 128, 4, 2),
    ("linear+learnP",      "linear",      "chaos", 0.003, 32, True,  128, 4, 2),
    ("transformer+chaos",  "transformer", "chaos", 0.003, 32, False, 128, 4, 2),
    ("transformer+raw",    "transformer", "raw",   0.001, 32, False, 128, 4, 2),
]


def main():
    parser = argparse.ArgumentParser(description='MathBrain benchmark')
    parser.add_argument('--corpus', default='datasets/tinystories_10.txt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--only', type=str, default=None,
                        help='Comma-separated experiment names to run. '
                             'Available: linear+chaos, linear+raw, '
                             'transformer+chaos, transformer+raw. '
                             'Default: run all.')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate for all experiments')
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--cp-rank', type=int, default=32)
    args = parser.parse_args()

    corpus_path = args.corpus
    corpus = [l.strip() for l in open(corpus_path) if l.strip()]
    corpus_name = Path(corpus_path).stem
    print(f"Corpus: {corpus_path} ({len(corpus)} sentences)")
    print(f"N={N}, D_chaos={D_CHAOS}, epochs={args.epochs}")
    print(f"Output: slot_scores → word_proj → CrossEntropy (unified)\n")

    # 过滤实验
    exps = EXPERIMENTS
    if args.only:
        selected = set(s.strip() for s in args.only.split(','))
        exps = [e for e in exps if e[0] in selected]
        if not exps:
            print(f"No matching experiments. Available: "
                  f"{', '.join(e[0] for e in EXPERIMENTS)}")
            return

    results = []
    for name, pred, phi, default_lr, default_cp, default_lP, default_dm, default_nh, default_nl in exps:
        lr = args.lr if args.lr is not None else default_lr
        cp_rank = args.cp_rank
        dm = args.d_model
        nh = args.n_heads
        nl = args.n_layers
        lP = default_lP
        print(f"── {name} (lr={lr}, cp_rank={cp_rank}, learnP={lP}) ──")
        acc, loss, elapsed = run(
            corpus, pred, phi, lr, cp_rank=cp_rank, learnable_P=lP,
            d_model=dm, n_heads=nh, n_layers=nl,
            epochs=args.epochs, batch_size=args.batch_size,
        )
        print(f"   acc={acc:.2f}%  loss={loss:.6f}  time={elapsed:.0f}s\n")
        results.append((name, pred, phi, lr, acc, loss, elapsed))

    # ── 汇总 ──
    print(f"\n{'='*65}")
    print(f"Benchmark: {corpus_name}, {len(corpus)} sentences, {args.epochs} epochs")
    print(f"{'name':>22} {'predictor':>12} {'phi':>6} {'lr':>7} | {'acc':>7} {'loss':>10} {'time':>6}")
    print("-" * 75)
    for name, pred, phi, lr, acc, loss, elapsed in results:
        print(f"{name:>22} {pred:>12} {phi:>6} {lr:7.4f} | {acc:6.2f}% {loss:10.6f} {elapsed:5.0f}s")

    # ── 关键对比 ──
    print(f"\nKey comparisons:")
    accs = {name: acc for name, _, _, _, acc, _, _ in results}
    if 'linear+chaos' in accs and 'linear+raw' in accs:
        delta = accs['linear+chaos'] - accs['linear+raw']
        print(f"  CosineChaos value (linear):      {delta:+.1f}%  "
              f"({accs['linear+chaos']:.1f}% vs {accs['linear+raw']:.1f}%)")
    if 'transformer+chaos' in accs and 'transformer+raw' in accs:
        delta = accs['transformer+chaos'] - accs['transformer+raw']
        print(f"  CosineChaos value (transformer):  {delta:+.1f}%  "
              f"({accs['transformer+chaos']:.1f}% vs {accs['transformer+raw']:.1f}%)")
    if 'linear+chaos' in accs and 'transformer+chaos' in accs:
        delta = accs['transformer+chaos'] - accs['linear+chaos']
        print(f"  Transformer value (chaos):        {delta:+.1f}%  "
              f"({accs['transformer+chaos']:.1f}% vs {accs['linear+chaos']:.1f}%)")
    if 'linear+raw' in accs and 'transformer+raw' in accs:
        delta = accs['transformer+raw'] - accs['linear+raw']
        print(f"  Transformer value (raw):          {delta:+.1f}%  "
              f"({accs['transformer+raw']:.1f}% vs {accs['linear+raw']:.1f}%)")


if __name__ == '__main__':
    main()
