#!/usr/bin/env python3
"""训练 BPE 词汇表 (sentencepiece)

用法:
  # 从语料训练 8K vocab 的 BPE
  python train_bpe.py --corpus datasets/wikitext2_train.txt --vocab-size 8000

  # 自定义输出路径
  python train_bpe.py --corpus data.txt --vocab-size 16000 --output tokenizers/my_bpe

  # 包含验证集语料（让 vocab 覆盖更全）
  python train_bpe.py --corpus datasets/wikitext2_train.txt \
      --extra-corpus datasets/wikitext2_validation.txt \
      --vocab-size 8000
"""

import argparse
import os
import tempfile

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--corpus', required=True, help='训练语料 (一行一句)')
    parser.add_argument('--extra-corpus', type=str, default=None,
                        help='额外语料 (如验证集, 用于扩展 vocab 覆盖)')
    parser.add_argument('--vocab-size', type=int, default=8000,
                        help='BPE 词汇表大小 (默认 8000)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出前缀 (默认: tokenizers/bpe_{vocab_size})')
    parser.add_argument('--character-coverage', type=float, default=1.0,
                        help='字符覆盖率 (默认 1.0, 英文够用)')
    parser.add_argument('--model-type', type=str, default='bpe',
                        choices=['bpe', 'unigram'],
                        help='分词算法 (默认 bpe)')
    args = parser.parse_args()

    # Output path
    if args.output is None:
        os.makedirs('tokenizers', exist_ok=True)
        args.output = f'tokenizers/bpe_{args.vocab_size}'

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Merge corpus files if extra-corpus provided
    if args.extra_corpus:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            merged_path = f.name
            for path in [args.corpus, args.extra_corpus]:
                with open(path) as src:
                    for line in src:
                        if line.strip():
                            f.write(line)
        input_path = merged_path
        print(f"Merged: {args.corpus} + {args.extra_corpus}")
    else:
        input_path = args.corpus

    # Count lines
    with open(input_path) as f:
        n_lines = sum(1 for l in f if l.strip())
    print(f"Training BPE: {n_lines:,} lines, vocab_size={args.vocab_size}")

    # Train
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        # Preserve whitespace as part of tokens (like GPT-2)
        split_by_whitespace=True,
        add_dummy_prefix=False,
        # Special tokens
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        # Don't split digits
        split_digits=False,
        # Byte fallback for OOV characters
        byte_fallback=True,
    )

    # Verify
    sp = spm.SentencePieceProcessor()
    sp.load(f'{args.output}.model')
    print(f"\n✅ Saved: {args.output}.model ({sp.get_piece_size()} tokens)")
    print(f"   Vocab: {args.output}.vocab")

    # Test
    test_sents = [
        "The cat sat on the mat .",
        "Neural networks are universal function approximators .",
        "In 2024, large language models achieved remarkable results .",
    ]
    for s in test_sents:
        tokens = sp.encode_as_pieces(s)
        ids = sp.encode_as_ids(s)
        print(f"\n  Input:  {s}")
        print(f"  Tokens: {tokens}")
        print(f"  IDs:    {ids}")

    # Cleanup temp file
    if args.extra_corpus:
        os.unlink(merged_path)


if __name__ == '__main__':
    main()
