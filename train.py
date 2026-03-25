#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from mathbrain.config import MathBrainConfig
from mathbrain.retina import IdentityRetina, BPERetina
from mathbrain.data import get_dataloader
from mathbrain.trainer import MathBrainTrainer

def main():
    parser = argparse.ArgumentParser(description="MathBrain Clean Rewrite Training")
    parser.add_argument('--corpus', type=str, required=True, help="Path to training text file")
    parser.add_argument('--val-corpus', type=str, help="Path to validation text file")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--seq-len', type=int, default=256, help="Sequence length")
    parser.add_argument('--bpe-model', type=str, default=None, help="BPE model file")
    parser.add_argument('--vocab', type=int, default=8000, help="Vocab size")
    
    args = parser.parse_args()

    print(f"Loading corpus: {args.corpus}")
    with open(args.corpus, 'r', encoding='utf-8') as f:
        train_text = f.read()

    val_text = None
    if args.val_corpus:
        with open(args.val_corpus, 'r', encoding='utf-8') as f:
            val_text = f.read()

    if args.bpe_model:
        print(f"Using BPE Retina with model {args.bpe_model}")
        retina = BPERetina(args.bpe_model)
        retina_mode = 'bpe'
    else:
        print("Using Identity Retina (word-level)")
        retina = IdentityRetina()
        retina_mode = 'identity'

    print("Tokenizing training data...")
    t0 = time.time()
    train_ids = retina.encode(train_text)
    print(f"Tokenized {len(train_ids)} tokens in {time.time()-t0:.2f}s")
    
    val_ids = None
    if val_text:
        val_ids = retina.encode(val_text)
        print(f"Tokenized validation {len(val_ids)} tokens")

    vocab_size = retina.vocab_size
    print(f"Setting vocab_size to {vocab_size} (from config arg --vocab {args.vocab})")
    # If using IdentityRetina, vocab_size gets updated during encode
    if retina_mode == 'identity':
        vocab_size = retina.vocab_size
        print(f"Identity Retina actual tokens: {vocab_size}")
    else:
        vocab_size = args.vocab

    # Match paper's exact hyperparams for WikiText-2:
    # d_model=64, n_layers=2, n_heads=8, N=64, half_life=1..1000
    cfg = MathBrainConfig(
        vocab_size=vocab_size,
        retina_mode=retina_mode,
        N=64,
        d_model=64,
        n_layers=2,
        n_heads=8,
        d_ff=256,
        phi_mode='linear'
    )

    train_loader = get_dataloader(train_ids, batch_size=args.batch_size, seq_len=args.seq_len, shuffle=True)
    val_loader = None
    if val_ids:
        val_loader = get_dataloader(val_ids, batch_size=args.batch_size, seq_len=args.seq_len, shuffle=False)

    trainer = MathBrainTrainer(cfg, device=None)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

if __name__ == '__main__':
    main()
