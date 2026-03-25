#!/usr/bin/env python3
import argparse
import os
import sentencepiece as spm

def train_bpe(corpus_paths: list, vocab_size: int, output_dir: str = "tokenizers", model_prefix: str = None):
    """
    Trains a SentencePiece BPE tokenizer on the given corpus files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if model_prefix is None:
        model_prefix = f"bpe_{vocab_size}"

    output_path = os.path.join(output_dir, model_prefix)

    input_files = ",".join(corpus_paths)
    print(f"Training BPE Tokenizer (vocab_size={vocab_size}) on {input_files}...")
    
    # SentencePiece parameters
    # - model_type: bpe
    # - pad_id: 0, unk_id: 1, bos_id: 2, eos_id: 3 (Standard setup)
    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=output_path,
        vocab_size=vocab_size,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2, 
        eos_id=3,
        character_coverage=1.0, # 1.0 for small character sets (like pure English), 0.9995 for rich text
        train_extremely_large_corpus=False
    )

    print(f"✅ Training complete! Model saved to {output_path}.model and {output_path}.vocab")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE model")
    parser.add_argument('--corpus', type=str, required=True, help="Path to the main training text file")
    parser.add_argument('--extra-corpus', type=str, nargs='*', default=[], help="Additional text files (like validation sets) to include in BPE training")
    parser.add_argument('--vocab', type=int, default=8000, help="Vocabulary size (default: 8000)")
    parser.add_argument('--out-dir', type=str, default="tokenizers", help="Output directory")
    
    args = parser.parse_args()
    
    all_corpora = [args.corpus] + args.extra_corpus
    train_bpe(all_corpora, args.vocab, args.out_dir)
