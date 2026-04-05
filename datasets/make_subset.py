"""
Create subsets of a text dataset (one article per line) for scaling experiments.

Usage:
    python datasets/make_subset.py datasets/wikitext103_train.txt --ratios 0.1 0.3 0.5

Output:
    datasets/wikitext103_train_10pct.txt
    datasets/wikitext103_train_30pct.txt
    datasets/wikitext103_train_50pct.txt
"""
import argparse
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input text file (one doc per line)')
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.1, 0.3, 0.5],
                        help='Subset ratios (e.g., 0.1 0.3 0.5)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]

    n_total = len(lines)
    base, ext = os.path.splitext(args.input)

    random.seed(args.seed)
    shuffled = list(range(n_total))
    random.shuffle(shuffled)

    for ratio in sorted(args.ratios):
        n = int(n_total * ratio)
        subset_indices = sorted(shuffled[:n])  # keep original order
        out_path = f"{base}_{int(ratio*100)}pct{ext}"

        with open(out_path, 'w', encoding='utf-8') as f:
            for i in subset_indices:
                f.write(lines[i])

        size_mb = os.path.getsize(out_path) / 1024**2
        print(f"{int(ratio*100)}%: {n}/{n_total} articles → {out_path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
