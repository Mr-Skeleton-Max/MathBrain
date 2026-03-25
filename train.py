#!/usr/bin/env python3
import argparse
import time
import os
from pathlib import Path
from mathbrain.config import MathBrainConfig
from mathbrain.data import get_precomputed_dataloader
from mathbrain.trainer import MathBrainTrainer

def main():
    parser = argparse.ArgumentParser(description="MathBrain Clean Rewrite Training (Offline Precomputed)")
    parser.add_argument('--dataset', type=str, required=True, help="Path to precomputed .pt dataset")
    parser.add_argument('--val-dataset', type=str, help="Path to validation precomputed .pt dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--seq-len', type=int, default=256, help="Sequence length for decoder training chunks")
    parser.add_argument('--vocab', type=int, default=8000, help="Vocab size")
    
    args = parser.parse_args()

    if not args.dataset.endswith('.pt'):
        print(f"Error: Expected a `.pt` precomputed dataset. Got {args.dataset}")
        print("Please run `python -m mathbrain.gpu_preprocessor --corpus your.txt --output your_cached.pt` first!")
        return
        
    print(f"Loading precomputed dataset: {args.dataset}")

    # Match paper's exact hyperparams for WikiText-2:
    # d_model=64, n_layers=2, n_heads=8, N=64, half_life=1..1000
    cfg = MathBrainConfig.from_half_lives(
        h_min=1,
        h_max=1000,
        vocab_size=args.vocab,
        retina_mode='bpe', # Precomputed features already mapped tokens to slots
        N=64,
        d_model=64,
        n_layers=2,
        n_heads=8,
        d_ff=256,
        phi_mode='linear'
    )

    train_loader = get_precomputed_dataloader(args.dataset, batch_size=args.batch_size, seq_len=args.seq_len, shuffle=True)
    val_loader = None
    if args.val_dataset:
        val_loader = get_precomputed_dataloader(args.val_dataset, batch_size=args.batch_size, seq_len=args.seq_len, shuffle=False)

    trainer = MathBrainTrainer(cfg, device=None)
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

if __name__ == '__main__':
    main()
