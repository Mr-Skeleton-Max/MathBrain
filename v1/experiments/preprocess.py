#!/usr/bin/env python3
"""独立预处理脚本: corpus → binary cache

用法:
  python experiments/preprocess.py \\
      --corpus datasets/tinystories_2000.txt \\
      --rho 0.3,0.75,0.93,0.98,0.995,0.999,0.9995,0.9999 \\
      --D 32 --cp-rank 32 \\
      --storage-dtype bf16 \\
      --cache-dir cache/

说明:
  预处理可在无 GPU 的 CPU 机器上运行。
  生成的 .bin + .meta.npz 文件拷贝到 GPU 服务器后，
  train_a.py 会自动检测并加载（跳过预处理）。
"""

import argparse
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mathbrain import MathBrain, MathBrainConfig
from mathbrain.data_cache import preprocess_corpus


def parse_rho(s):
    """Parse comma-separated rho string."""
    return np.array([float(x) for x in s.split(',')], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description='MathBrain 预处理: corpus → binary cache',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--corpus', required=True,
                        help='语料文件路径 (每行一句)')
    parser.add_argument('--rho', type=parse_rho,
                        default='0.3,0.75,0.93,0.98',
                        help='EMA 衰减系数 (逗号分隔)')
    parser.add_argument('--D', type=int, default=8,
                        help='Phi 编码器输出维度 D_PHI')
    parser.add_argument('--cp-rank', type=int, default=384,
                        help='CP 分解秩')
    parser.add_argument('--cache-dir', default='cache/',
                        help='缓存目录 (default: cache/)')
    parser.add_argument('--storage-dtype', choices=['fp32', 'bf16'],
                        default='fp32',
                        help='存储精度: fp32 (默认) 或 bf16 (省约50%%空间)')
    parser.add_argument('--phi-mode', default='chaos',
                        choices=['chaos', 'raw'],
                        help='Phi 编码模式')
    parser.add_argument('--force', action='store_true',
                        help='强制重新预处理 (忽略缓存)')

    args = parser.parse_args()

    # Load corpus
    with open(args.corpus) as f:
        corpus = [line.strip() for line in f if line.strip()]
    print(f"Corpus: {args.corpus}, {len(corpus)} sentences")

    # Build config & model
    rho = args.rho
    N = len(rho)
    hl = np.log(0.5) / np.log(rho)

    print(f"\n{'='*60}")
    print(f"N={N}, D={args.D}, CP_RANK={args.cp_rank}")
    print(f"  max half-life={hl[-1]:.0f}, phi_mode={args.phi_mode}")
    print(f"  storage_dtype={args.storage_dtype}")
    print(f"  RHO ({N} scales):")
    for i, (r, h) in enumerate(zip(rho, hl)):
        print(f"    [{i}] rho={r:.6f}  half-life={h:.1f} steps")
    print(f"{'='*60}")

    cfg = MathBrainConfig(N=N, RHO=tuple(rho), D_PHI=args.D,
                          CP_RANK=args.cp_rank)
    model = MathBrain(cfg)

    # Force re-preprocess if requested
    if args.force:
        from mathbrain.data_cache import _cache_filename
        from pathlib import Path
        cache_path = str(Path(args.cache_dir) / _cache_filename(args.corpus, cfg))
        for ext in ['', '.meta.npz']:
            p = cache_path + ext
            if os.path.exists(p):
                os.remove(p)
                print(f"  删除旧缓存: {p}")

    # Run preprocessing
    t0 = time.time()
    cache_path = preprocess_corpus(
        corpus, model,
        cache_dir=args.cache_dir,
        corpus_path=args.corpus,
        phi_mode=args.phi_mode,
        storage_dtype=args.storage_dtype,
    )
    elapsed = time.time() - t0

    # Print stats
    file_size = os.path.getsize(cache_path)
    meta_size = os.path.getsize(cache_path + '.meta.npz')
    print(f"\n  缓存文件: {cache_path}")
    print(f"  大小: {file_size/1e9:.3f} GB  (+meta {meta_size/1e6:.1f} MB)")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  吞吐量: {len(corpus)/max(elapsed, 0.001):.0f} 句/秒")


if __name__ == '__main__':
    main()
