#!/usr/bin/env python3
"""Standard Transformer LM Baseline — 对比 MathBrain

标准 word-level Transformer 语言模型:
  word embedding + sinusoidal PE → causal Transformer → LM head → logits

用法:
  python baseline_transformer.py \
    --corpus datasets/dailydialog_utt_5000_train.txt \
    --val-corpus datasets/dailydialog_utt_5000_val.txt \
    --epochs 640 --d-model 128 --n-layers 2 --n-heads 4
"""

import argparse
import math
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Tokenizer (简单 word-level, 与 MathBrain 对齐)
# ═══════════════════════════════════════════════════════════════

def build_vocab(sentences: List[str], min_freq: int = 1):
    """Build word-level vocab."""
    from collections import Counter
    counter = Counter()
    for s in sentences:
        counter.update(s.lower().split())

    vocab = ['<pad>', '<unk>', '<bos>', '<eos>']
    for w, c in counter.most_common():
        if c >= min_freq:
            vocab.append(w)

    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i


def tokenize(sentence: str, w2i: dict) -> List[int]:
    """Tokenize a sentence to word ids."""
    unk = w2i['<unk>']
    bos = w2i['<bos>']
    eos = w2i['<eos>']
    tokens = [bos] + [w2i.get(w, unk) for w in sentence.lower().split()] + [eos]
    return tokens


def make_dataset(sentences: List[str], w2i: dict, max_len: int = 128):
    """Create padded tensor dataset for LM training."""
    all_tokens = []
    for s in sentences:
        toks = tokenize(s, w2i)
        if len(toks) > max_len:
            toks = toks[:max_len]
        all_tokens.append(toks)

    # Pad to max length in batch
    max_l = max(len(t) for t in all_tokens)
    pad_id = w2i['<pad>']
    padded = torch.full((len(all_tokens), max_l), pad_id, dtype=torch.long)
    for i, toks in enumerate(all_tokens):
        padded[i, :len(toks)] = torch.tensor(toks, dtype=torch.long)

    return padded


# ═══════════════════════════════════════════════════════════════
# Standard Transformer LM
# ═══════════════════════════════════════════════════════════════

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StandardTransformerLM(nn.Module):
    """Standard causal Transformer language model."""

    def __init__(self, V: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, d_ffn: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(V, d_model, padding_idx=0)
        self.pe = SinusoidalPE(d_model)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_ffn, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, V, bias=False)

        # Weight tying
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x, pad_mask=None):
        """x: (B, T) → logits: (B, T, V)"""
        T = x.size(1)
        # Causal mask
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        h = self.drop(self.pe(self.embedding(x) * math.sqrt(self.d_model)))
        h = self.encoder(h, mask=causal, src_key_padding_mask=pad_mask, is_causal=True)
        logits = self.lm_head(self.norm(h))
        return logits


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, data, optimizer, pad_id, device, batch_size=64):
    model.train()
    total_loss = 0
    total_tokens = 0
    perm = torch.randperm(data.size(0))

    for i in range(0, data.size(0), batch_size):
        batch = data[perm[i:i+batch_size]].to(device)
        x = batch[:, :-1]  # input
        y = batch[:, 1:]   # target

        pad_mask = (x == pad_id)
        logits = model(x, pad_mask=pad_mask)

        # Flatten and compute loss (ignore padding)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_id,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n_tokens = (y != pad_id).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, data, pad_id, device, batch_size=64):
    model.eval()
    total_loss = 0
    total_tokens = 0
    correct = 0

    for i in range(0, data.size(0), batch_size):
        batch = data[i:i+batch_size].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        pad_mask = (x == pad_id)
        logits = model(x, pad_mask=pad_mask)

        valid = (y != pad_id)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=pad_id,
            reduction='sum',
        )
        total_loss += loss.item()
        n = valid.sum().item()
        total_tokens += n
        preds = logits.argmax(-1)
        correct += (preds == y)[valid].sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    acc = correct / max(total_tokens, 1) * 100
    ppl = math.exp(min(avg_loss, 20))
    return {'loss': avg_loss, 'ppl': ppl, 'acc': acc,
            'correct': correct, 'total': total_tokens}


def main():
    parser = argparse.ArgumentParser(description='Standard Transformer LM Baseline')
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--val-corpus', default=None)
    parser.add_argument('--epochs', type=int, default=640)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--d-ffn', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-every', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_sents = [l.strip() for l in open(args.corpus) if l.strip()]
    val_sents = None
    if args.val_corpus:
        val_sents = [l.strip() for l in open(args.val_corpus) if l.strip()]
        print(f"Train: {len(train_sents)}, Val: {len(val_sents)}")
    else:
        print(f"Train: {len(train_sents)}")

    # Build vocab from training data
    vocab, w2i = build_vocab(train_sents)
    V = len(vocab)
    pad_id = w2i['<pad>']
    print(f"Vocab: {V} words")

    # Tokenize
    train_data = make_dataset(train_sents, w2i)
    val_data = make_dataset(val_sents, w2i) if val_sents else None

    total_train_tokens = (train_data != pad_id).sum().item()
    print(f"Train tokens: {total_train_tokens:,}")
    if val_data is not None:
        total_val_tokens = (val_data != pad_id).sum().item()
        print(f"Val tokens: {total_val_tokens:,}")

    # Build model
    model = StandardTransformerLM(
        V=V, d_model=args.d_model, nhead=args.n_heads,
        num_layers=args.n_layers, d_ffn=args.d_ffn,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: d_model={args.d_model}, layers={args.n_layers}, "
          f"heads={args.n_heads}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Train
    print(f"\n── Training ({args.epochs} epochs) ──\n")
    best_val_ppl = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_data, optimizer, pad_id, device,
                                 batch_size=args.batch_size)
        scheduler.step()
        dt = time.time() - t0

        if epoch % args.eval_every == 0:
            train_ppl = math.exp(min(train_loss, 20))
            lr = optimizer.param_groups[0]['lr']
            print(f"  epoch {epoch:4d}: loss={train_loss:.6f}, ppl={train_ppl:.1f} "
                  f"({dt*1000:.0f}ms, lr={lr:.2e})")

            if val_data is not None:
                val_res = evaluate(model, val_data, pad_id, device)
                if val_res['ppl'] < best_val_ppl:
                    best_val_ppl = val_res['ppl']
                print(f"         val: acc={val_res['acc']:.1f}%, "
                      f"ppl={val_res['ppl']:.1f}, loss={val_res['loss']:.4f}")

    # Final eval
    print(f"\n── Final Results ──")
    train_res = evaluate(model, train_data, pad_id, device)
    print(f"  Train: acc={train_res['acc']:.1f}%, ppl={train_res['ppl']:.1f}")
    if val_data is not None:
        val_res = evaluate(model, val_data, pad_id, device)
        print(f"  Val:   acc={val_res['acc']:.1f}%, ppl={val_res['ppl']:.1f}")
        print(f"  Best Val PPL: {best_val_ppl:.1f}")


if __name__ == '__main__':
    main()
