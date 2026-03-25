#!/usr/bin/env python3
"""MathBrain GPU 训练 CLI

用法:
  # 训练
  python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 100

  # 训练 + 评估
  python train_gpu.py --corpus datasets/tinystories_1000.txt --epochs 100 --eval-every 20

  # 从 checkpoint 继续训练
  python train_gpu.py --corpus datasets/tinystories_2000.txt --epochs 320 \
      --resume model.pt --save model_v2.pt

  # 自定义参数
  python train_gpu.py --corpus data.txt --epochs 200 --batch-size 8192 --lr 0.005
"""
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mathbrain.config import MathBrainConfig
from mathbrain.trainer import MathBrainTrainer
from mathbrain.data import BPETokenizer, set_tokenizer


def main():
    parser = argparse.ArgumentParser(description='MathBrain GPU Training')
    parser.add_argument('--corpus', required=True, help='训练语料文件')
    parser.add_argument('--val-corpus', type=str, default=None,
                        help='验证集文件 (独立文件, 与 --val-split 互斥)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.01, help='最大学习率')
    parser.add_argument('--min-lr', type=float, default=0.0,
                        help='最终学习率 (默认: lr/25000)')
    parser.add_argument('--warmup-pct', type=float, default=0.05,
                        help='warmup 占比 (默认: 0.05 = 5%%)')
    parser.add_argument('--scheduler', type=str, default='constant',
                        choices=['constant', 'cosine', 'step', 'exp'],
                        help='LR调度: constant|cosine|step|exp (默认: constant)')
    parser.add_argument('--noise-sigma', type=float, default=0.0,
                        help='ctx 高斯噪声 σ (0=关, >0=sleep-during-wake)')
    parser.add_argument('--val-split', type=float, default=0.0,
                        help='验证集比例 (0=不切分, 0.1=10%%)')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--eval-every', type=int, default=0,
                        help='每 N epoch 评估 val (0=仅结束时)')
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--retina', type=str, default='hash',
                        choices=['hash', 'identity', 'bpe'],
                        help='Retina 模式: hash (n-gram) | identity (1 word=1 slot) | bpe (1 BPE token=1 slot)')
    parser.add_argument('--vicreg', type=float, default=0.0,
                        help='Anti-collapse (VICReg) weight, 0=off, try 0.01~1.0')
    parser.add_argument('--tie-weights', action='store_true',
                        help='Weight tying: E_src as lm_head (slot embed = output proj)')
    parser.add_argument('--pred-mode', type=str, default='ce',
                        choices=['ce', 'innovation', 'hybrid'],
                        help='Prediction mode: ce | innovation (MSE) | hybrid (CE + λ·MSE)')
    parser.add_argument('--innovation-weight', type=float, default=0.1,
                        help='MSE innovation weight λ for hybrid mode (default 0.1)')
    parser.add_argument('--half-life', type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'),
                        help='半衰期范围，对数均匀生成 N 个 ρ (例: --half-life 1 1000)')
    parser.add_argument('--D', type=int, default=32, help='D_PHI')
    parser.add_argument('--rank', type=int, default=32, help='CP_RANK')
    parser.add_argument('--alpha', type=float, default=2.1,
                        help='Cosine Chaos α (默认 2.1, 越大混沌越强)')
    parser.add_argument('--residual', action='store_true',
                        help='φ→φ+1 残差: 保留裸语义信号')
    parser.add_argument('--mlp-phi', action='store_true',
                        help='用 MLP 替代 CosineChaos φ 编码器')
    parser.add_argument('--mlp-hidden', type=int, default=64,
                        help='MLP φ 隐藏层维度 (默认 64)')
    parser.add_argument('--mlp-ctx', action='store_true',
                        help='在 ctx 聚合后加 MLP 非线性层 (Position C)')
    parser.add_argument('--ctx-hidden', type=int, default=256,
                        help='ctx MLP 隐藏层维度 (默认 256)')
    parser.add_argument('--fourier-phi', action='store_true',
                        help='用 Fourier PE 替代 CosineChaos φ 编码器')
    parser.add_argument('--K', type=int, default=8,
                        help='Fourier PE 频率数 (默认 8)')
    parser.add_argument('--transformer', action='store_true',
                        help='用 Slot Transformer decoder 替代 bilinear')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Transformer d_model (默认 128)')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='Transformer 层数 (默认 2)')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Transformer attention heads (默认 4)')
    parser.add_argument('--pe-mode', type=str, default='fourier',
                        choices=['fourier', 'linear'],
                        help='PE 编码方式: fourier | linear (默认 fourier)')
    parser.add_argument('--q-transform', type=str, default='none',
                        choices=['none', 'log', 'norm'],
                        help='Q 变换: none | log (Weber-Fechner) | norm (L2归一化)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 加载继续训练')
    parser.add_argument('--save', type=str, help='保存模型路径')
    parser.add_argument('--tokenizer', type=str, default='word',
                        choices=['word', 'bpe'],
                        help='分词模式: word (默认) | bpe (GPT-2 BPE)')
    parser.add_argument('--bpe-model', type=str, default=None,
                        help='BPE 模型路径 (.model), 搭配 --tokenizer bpe 使用')
    args = parser.parse_args()

    corpus = [l.strip() for l in open(args.corpus) if l.strip()]
    val_corpus_ext = None
    if args.val_corpus:
        val_corpus_ext = [l.strip() for l in open(args.val_corpus) if l.strip()]
        print(f"Train: {len(corpus)}, Val: {len(val_corpus_ext)} (from {args.val_corpus})")

    # Setup tokenizer
    if args.tokenizer == 'bpe':
        bpe_tok = BPETokenizer(model_path=args.bpe_model)
        set_tokenizer(bpe_tok)
        args.retina = 'bpe'  # BPE: 1 token = 1 slot, no hash
        print(f"Tokenizer: BPE ({bpe_tok.backend}, vocab={bpe_tok.vocab_size})")
    else:
        set_tokenizer(None)
        print("Tokenizer: word-level")

    if args.resume:
        # Resume: load checkpoint, config comes from checkpoint
        dummy_cfg = MathBrainConfig()
        trainer = MathBrainTrainer(dummy_cfg, device='cuda')
        trainer.load(args.resume)
        print(f"\n── Resume training from {args.resume} ──\n")
    else:
        # Generate RHO
        if args.half_life:
            h_min, h_max = args.half_life
            import numpy as np
            half_lives = np.logspace(np.log10(h_min), np.log10(h_max), args.N)
            rho_vals = tuple(float(0.5 ** (1.0 / h)) for h in half_lives)
            print(f"  ρ from half-life [{h_min}, {h_max}]: {rho_vals}")
        else:
            rho_vals = (0.3, 0.75, 0.93, 0.98, 0.995, 0.999, 0.9995, 0.9999)[:args.N]
        cfg = MathBrainConfig(
            N=args.N,
            RHO=rho_vals,
            RETINA_MODE=args.retina,
            D_PHI=args.D,
            CP_RANK=args.rank,
            CHAOS_ALPHA=args.alpha,
            MLP_PHI_HIDDEN=args.mlp_hidden if args.mlp_phi else 0,
            MLP_CTX_HIDDEN=args.ctx_hidden if args.mlp_ctx else 0,
            FOURIER_PHI_K=args.K if args.fourier_phi else 0,
            TRANSFORMER_D_MODEL=args.d_model if args.transformer else 0,
            TRANSFORMER_NHEAD=args.n_heads,
            TRANSFORMER_LAYERS=args.n_layers,
            TRANSFORMER_PE_MODE=args.pe_mode,
            TRANSFORMER_Q_TRANSFORM=args.q_transform,
            VICREG_WEIGHT=args.vicreg,
            TRANSFORMER_TIE_WEIGHTS=args.tie_weights,
            TRANSFORMER_PRED_MODE=args.pred_mode,
            INNOVATION_WEIGHT=args.innovation_weight,
        )
        trainer = MathBrainTrainer(cfg, device='cuda')

    result = trainer.fit(
        corpus,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_pct=args.warmup_pct,
        scheduler=args.scheduler,
        noise_sigma=args.noise_sigma,
        val_split=(args.val_split if args.val_split > 0
                   else (0.1 if args.eval_every > 0 and not val_corpus_ext else 0.0)
                   ) if not val_corpus_ext else 0.0,
        val_corpus_ext=val_corpus_ext,
        phi_residual=args.residual,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
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
