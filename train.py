"""
MathBrain Project: SlotTransformerLM / RoPETransformerLM Training Script
========================================================================

Key concepts:
  --seq_len     RoPE: context window (model capacity).
                EMA: max chunk size for splitting long docs (set large, e.g. 10000).
  --token_budget  Recommended: total tokens per batch (GPU memory control).
                  Dynamic B and L per batch. 0 = fallback to fixed batch_size.
  --batch_size    Fallback: fixed samples per batch (only if token_budget=0).

Usage:

1. EMA on WikiText-103 (token budget, dynamic batching):
  python train.py --model ema \\
      --train_file datasets/wikitext103_train.txt \\
      --val_file datasets/wikitext103_val.txt \\
      --tokenizer gpt2 --d_model 512 --n_layers 6 --n_heads 8 \\
      --seq_len 10000 --token_budget 65536 \\
      --lr 3e-4 --epochs 15

2. RoPE baseline (same data, fixed context window):
  python train.py --model rope \\
      --train_file datasets/wikitext103_train.txt \\
      --val_file datasets/wikitext103_val.txt \\
      --tokenizer gpt2 --d_model 512 --n_layers 6 --n_heads 8 \\
      --seq_len 1024 --token_budget 65536 \\
      --lr 3e-4 --epochs 15
"""

import os
import time
import math
import argparse
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mathbrain.data import preprocess_corpus, WikiTextMMapDataset, dynamic_collate_fn, TokenBudgetSampler
from mathbrain.model import SlotTransformerLM
from mathbrain.baseline import RoPETransformerLM

def get_peak_memory_mb(device):
    if device == 'cuda':
        return torch.cuda.max_memory_allocated() / (1024**2)
    elif device == 'mps':
        # MPS doesn't natively expose peak memory historically like CUDA, just current
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0

def reset_memory(device):
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    elif device == 'mps':
        torch.mps.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ema', choices=['ema', 'rope'])
    parser.add_argument('--train_file', type=str, required=True, help='Path to the raw train tokens/text file')
    parser.add_argument('--val_file', type=str, default=None, help='Path to the raw valid tokens/text file')
    parser.add_argument('--out_dir', type=str, default='processed_data', help='Directory to cache preprocessed tensors')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Fixed samples per batch (only used if --token_budget=0)')
    parser.add_argument('--seq_len', type=int, default=2048,
                        help='RoPE: context window size. EMA: max chunk size for splitting long docs (set large, e.g. 10000)')
    parser.add_argument('--token_budget', type=int, default=0,
                        help='Total tokens per batch (recommended). 0=fallback to fixed batch_size×seq_len')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--min_hl', type=float, default=1.0, help='Minimum EMA half-life in tokens')
    parser.add_argument('--max_hl', type=float, default=2048.0, help='Maximum EMA half-life in tokens')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--tokenizer', type=str, default='gpt2', choices=['gpt2', 'custom'], help='Type of BPE tokenizer')
    parser.add_argument('--vocab_size', type=int, default=50304, help='Vocab size for custom tokenizer')
    parser.add_argument('--log_interval', type=int, default=10, help='Epochs between logging')
    parser.add_argument('--step_log_interval', type=int, default=0, help='Steps between intra-epoch logging (0 = disabled)')
    parser.add_argument('--linear_gating', action='store_true', help='Disable SiLU gating and use pure linear associative retrieval')
    parser.add_argument('--ema_dropout', type=float, default=0.0, help='Dropout on EMA temporal states (test generalization hypothesis)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() for auto-fusion')
    parser.add_argument('--no_log', action='store_true', help='Disable auto-logging and checkpointing')
    parser.add_argument('--exp_dir', type=str, default='experiments', help='Base directory for experiments')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    
    # ---------------- 0. Tokenizer Initialization ----------------
    if args.tokenizer == 'gpt2':
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab
        def encode(text): return enc.encode(text)
    else:
        from tokenizers import ByteLevelBPETokenizer, Tokenizer
        tok_path = os.path.join(args.out_dir, "custom_bpe.json")
        
        if not os.path.exists(tok_path):
            tokenizer = ByteLevelBPETokenizer()
            os.makedirs(args.out_dir, exist_ok=True)
            print(f"Training custom BPE tokenizer on {args.train_file}...")
            tokenizer.train(files=[args.train_file], vocab_size=args.vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"])
            tokenizer.save(tok_path)
            print("Finished training.")
        else:
            tokenizer = Tokenizer.from_file(tok_path)
            
        vocab_size = tokenizer.get_vocab_size()
        def encode(text): return tokenizer.encode(text).ids
        
    # ---------------- 0. Logging & Checkpointing Setup ----------------
    run_dir = None
    if not args.no_log:
        import datetime, json, logging, sys
        os.makedirs(args.exp_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{args.model}"
        run_dir = os.path.join(args.exp_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save exact command and arguments
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            config = vars(args)
            config['command'] = " ".join(sys.argv)
            json.dump(config, f, indent=4)
            
        # Set up logger
        logger = logging.getLogger("Trainer")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(run_dir, 'train.log'))
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        def lprint(*pargs, **kwargs):
            msg = " ".join(map(str, pargs))
            logger.info(msg)
    else:
        def lprint(*pargs, **kwargs):
            print(*pargs, **kwargs)
            
    # ---------------- 1. Caching & Data Prep ----------------
    def prep_dataset(file_path, cache_dir):
        if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
            if not os.path.exists(cache_dir): os.makedirs(cache_dir)
            lprint(f"Preprocessing {file_path}. This operates ONCE and caches securely to disk.")
            documents = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        documents.append(encode(line))
            preprocess_corpus(documents, cache_dir, N=args.N, min_hl=args.min_hl, max_hl=args.max_hl)
        else:
            lprint(f"Loading pre-computed cached logic from {cache_dir}...")
            
        log_base = torch.log(torch.tensor(2.0))
        scales = torch.logspace(math.log10(args.min_hl), math.log10(args.max_hl), args.N)
        rhos = torch.exp(-log_base / scales).cpu()
        
        dataset = WikiTextMMapDataset(cache_dir, block_size=args.seq_len, rhos=rhos, in_memory=True)
        use_pin = (device == 'cuda')
        n_workers = 32 if use_pin else 0

        if args.token_budget > 0:
            sampler = TokenBudgetSampler(dataset, token_budget=args.token_budget, shuffle=True)
            loader = DataLoader(dataset, batch_sampler=sampler,
                                collate_fn=dynamic_collate_fn,
                                pin_memory=use_pin, num_workers=n_workers,
                                persistent_workers=(n_workers > 0))
        else:
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                collate_fn=dynamic_collate_fn, drop_last=False,
                                pin_memory=use_pin, num_workers=n_workers,
                                persistent_workers=(n_workers > 0))
        return loader
        
    train_base = os.path.splitext(os.path.basename(args.train_file))[0]
    cache_suffix = f"{args.tokenizer}_V{vocab_size}_N{args.N}_hl{args.min_hl}-{args.max_hl}"
    train_dir = os.path.join(args.out_dir, f"{train_base}_{cache_suffix}")
    train_loader = prep_dataset(args.train_file, train_dir)

    val_loader = None
    if args.val_file is not None:
        val_base = os.path.splitext(os.path.basename(args.val_file))[0]
        val_dir = os.path.join(args.out_dir, f"{val_base}_{cache_suffix}")
        val_loader = prep_dataset(args.val_file, val_dir)

    lprint(f"\n=== INITIALIZING TRAINER ===")
    lprint(f"Device: {device}")
    lprint(f"Model: {args.model.upper()}")
    if args.token_budget > 0:
        lprint(f"Batching: token_budget={args.token_budget} (dynamic B and L)")
    else:
        lprint(f"Batching: batch_size={args.batch_size}, seq_len={args.seq_len} (fixed)")
    if args.model == 'ema':
        lprint(f"seq_len={args.seq_len} (max chunk size for long docs, NOT context window)")
    else:
        lprint(f"seq_len={args.seq_len} (context window)")
    lprint(f"Train: {len(train_loader)} batches/epoch")
    if val_loader: lprint(f"Val: {len(val_loader)} batches")
    lprint("============================\n")

    # ---------------- 2. Model Prep ----------------
    if args.model == 'ema':
        model = SlotTransformerLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            N=args.N,
            use_silu=not args.linear_gating,
            ema_dropout=args.ema_dropout,
            min_hl=args.min_hl,
            max_hl=args.max_hl
        )
    else:
        model = RoPETransformerLM(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads
        )
    
    model.to(device)
    
    if args.compile:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine LR with linear warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(200, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device == 'cuda')

    # ---------------- 3. Evaluation Loop ----------------
    @torch.no_grad()
    def evaluate(loader, prefix="VAL"):
        model.eval()
        total_loss = 0.0
        for i, batch in enumerate(loader):
            if i > 50: break # Quick eval
            chunks, unique_slots, pad_mask, inverse_indices, c_bases, K_max, doc_ids = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
            ]
            
            x = chunks[:, :-1].contiguous()
            y = chunks[:, 1:].contiguous()
            doc_ids_x = doc_ids[:, :-1].contiguous()
            inv_idx_x = inverse_indices[:, :-1].contiguous()

            doc_ids_y = doc_ids[:, 1:].contiguous()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                if args.model == 'ema':
                    logits = model(x, unique_slots=unique_slots, inverse_indices=inv_idx_x, c_base=c_bases, doc_ids=doc_ids_x, pad_mask=pad_mask)
                else:
                    logits = model(x)
                y_eval = y.clone()
                y_eval[doc_ids_y == -1] = -1
                y_eval[doc_ids_x != doc_ids_y] = -1  # mask cross-document predictions
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_eval.view(-1), ignore_index=-1)
            total_loss += loss.detach()

        avg_loss = (total_loss / min(len(loader), 51)).item()
        lprint(f"\n--- {prefix} Loss: {avg_loss:.4f} | {prefix} PPL: {math.exp(avg_loss):.2f} ---\n")
        model.train()

    # ---------------- 4. Training Loop ----------------
    lprint("\nStarting Training...")
    best_loss = float('inf')
    
    model.train()
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        epoch_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            chunks, unique_slots, pad_mask, inverse_indices, c_bases, K_max, doc_ids = [
                b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch
            ]

            x = chunks[:, :-1].contiguous()
            y = chunks[:, 1:].contiguous()
            doc_ids_x = doc_ids[:, :-1].contiguous()
            doc_ids_y = doc_ids[:, 1:].contiguous()
            inv_idx_x = inverse_indices[:, :-1].contiguous()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                if args.model == 'ema':
                    logits = model(x, unique_slots=unique_slots, inverse_indices=inv_idx_x, c_base=c_bases, doc_ids=doc_ids_x, pad_mask=pad_mask)
                else:
                    logits = model(x)

                # Filter: mask padding AND cross-document boundary predictions
                y_target = y.clone()
                y_target[doc_ids_y == -1] = -1
                y_target[doc_ids_x != doc_ids_y] = -1  # cross-document boundary
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_target.view(-1), ignore_index=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.detach()

            if args.step_log_interval > 0 and (i + 1) % args.step_log_interval == 0:
                step_loss = loss.item()
                step_ppl = math.exp(step_loss) if step_loss < 10 else float('inf')
                global_step = epoch * len(train_loader) + i + 1
                lprint(f"  Step {global_step:6d} [{i+1}/{len(train_loader)}] | Loss: {step_loss:.4f} | PPL: {step_ppl:.2f}")

        epoch_t1 = time.time()
        avg_loss = (epoch_loss / len(train_loader)).item()  # single sync per epoch, not per step
        
        if epoch % args.log_interval == 0:
            mem = get_peak_memory_mb(device)
            ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
            cur_lr = scheduler.get_last_lr()[0]
            lprint(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {cur_lr:.2e} | Mem: {mem:.1f} MB | Time: {epoch_t1 - epoch_t0:.2f} s")
            
        if val_loader and epoch % args.eval_interval == 0 and epoch > 0:
            evaluate(val_loader)
            
        # Checkpointing
        if not args.no_log and epoch > 0 and epoch % args.log_interval == 0:
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'args': vars(args)
                }
                torch.save(ckpt, os.path.join(run_dir, 'model_best.pt'))
                
    if not args.no_log:
        final_ckpt = {
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'args': vars(args)
        }
        torch.save(final_ckpt, os.path.join(run_dir, 'model_final.pt'))
        lprint(f"\nTraining complete. Artifacts saved to {run_dir}/")


if __name__ == "__main__":
    main()
