"""诊断分析脚本 — 深入探究 EMA SlotTransformer 的泛化失败

用 Wakhloo et al. 的几何框架分析：
1. 预测分布: train vs val 的 logit/confidence 分布
2. 错误类型: val 上的错误是"不确定"还是"确定性错误"？
3. Q 值几何: participation ratio, 有效维度
4. 注意力/表示分析: train vs val 的隐层表示距离
"""

import sys
import os
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mathbrain.config import MathBrainConfig
from mathbrain.trainer import MathBrainTrainer
from mathbrain.data import tokenize
from mathbrain.gpu_preprocessor import gpu_preprocess, iter_batches_gpu
from mathbrain.slot_transformer import flat_to_padded


def load_model_and_data(model_path, train_file, val_file, device='cuda'):
    """加载模型和数据"""
    cfg = MathBrainConfig()
    trainer = MathBrainTrainer(cfg, device=device)
    trainer.load(model_path)
    
    train_corpus = [l.strip() for l in open(train_file) if l.strip()]
    val_corpus = [l.strip() for l in open(val_file) if l.strip()]
    
    return trainer, train_corpus, val_corpus


@torch.no_grad()
def detailed_eval(trainer, corpus, label='eval', max_positions=50000):
    """详细评估 — 收集每个 position 的 logit, confidence, correct/wrong"""
    device = trainer.device
    cfg = trainer.config
    vocab = trainer._vocab
    
    xfr = trainer._slot_xfr
    xfr.eval()
    
    eval_data = gpu_preprocess(corpus, vocab, cfg, device, verbose=False)
    
    all_logits = []
    all_targets = []
    all_preds = []
    all_probs = []  # prob of correct word
    all_top1_probs = []  # prob of top-1 prediction
    all_entropies = []
    all_Q_stats = []  # Q value statistics per position
    all_top5_preds = []  # top-5 predictions per position
    all_slot_info = []   # active slot IDs per position
    
    total = 0
    for b in iter_batches_gpu(eval_data, 4096, device, shuffle=False):
        Bi = b.counts.shape[0]
        
        ret = flat_to_padded(
            b.flat_slots, b.flat_Q, b.pos_lo, b.counts, device,
            flat_is_latest=b.flat_is_latest)
        if len(ret) == 4:
            padded_slots, padded_Q, pad_mask, is_latest = ret
        else:
            padded_slots, padded_Q, pad_mask = ret
            is_latest = None
        logits = xfr(padded_slots, padded_Q, pad_mask, is_latest=is_latest)
        
        # Softmax
        probs = F.softmax(logits, dim=-1)
        targets = b.targets
        preds = logits.argmax(dim=1)
        
        # Top-5 predictions for each position
        top5_vals, top5_idx = probs.topk(5, dim=1)
        
        # Per-position stats
        for i in range(Bi):
            target_idx = targets[i].item()
            pred_idx = preds[i].item()
            
            prob_correct = probs[i, target_idx].item()
            prob_top1 = probs[i, pred_idx].item()
            
            # Entropy of prediction distribution
            log_probs = F.log_softmax(logits[i], dim=0)
            entropy = -(probs[i] * log_probs).sum().item()
            
            # Logit stats
            max_logit = logits[i].max().item()
            mean_logit = logits[i].mean().item()
            
            # Q value stats for this position
            lo = b.pos_lo[i].item()
            n_active = b.counts[i].item()
            q_vals = b.flat_Q[lo:lo+n_active]  # (n_active, N)
            q_norms = q_vals.norm(dim=1)  # per-slot Q norm
            slot_ids = b.flat_slots[lo:lo+n_active].cpu().tolist()
            
            all_targets.append(target_idx)
            all_preds.append(pred_idx)
            all_probs.append(prob_correct)
            all_top1_probs.append(prob_top1)
            all_entropies.append(entropy)
            all_top5_preds.append([
                (top5_idx[i, j].item(), top5_vals[i, j].item())
                for j in range(5)
            ])
            all_slot_info.append(slot_ids)
            all_Q_stats.append({
                'n_active': n_active,
                'q_norm_mean': q_norms.mean().item(),
                'q_norm_max': q_norms.max().item(),
                'q_max': q_vals.max().item(),
                'max_logit': max_logit,
                'mean_logit': mean_logit,
            })
        
        total += Bi
        if total >= max_positions:
            break
    
    results = {
        'targets': np.array(all_targets),
        'preds': np.array(all_preds),
        'prob_correct': np.array(all_probs),
        'prob_top1': np.array(all_top1_probs),
        'entropy': np.array(all_entropies),
        'q_stats': all_Q_stats,
        'top5_preds': all_top5_preds,
        'slot_info': all_slot_info,
    }
    
    # Summary
    correct = (results['targets'] == results['preds']).sum()
    acc = correct / len(results['targets']) * 100
    avg_loss = -np.log(np.clip(results['prob_correct'], 1e-10, 1.0)).mean()
    ppl = math.exp(min(avg_loss, 20))
    
    print(f"\n{'='*60}")
    print(f" {label}: {correct}/{len(results['targets'])} = {acc:.1f}%")
    print(f" Loss = {avg_loss:.4f}, PPL = {ppl:.1f}")
    print(f"{'='*60}")
    
    return results


def analyze_confidence(train_res, val_res, vocab_list):
    """分析置信度差异 — 这是 PPL 爆炸的直接原因"""
    print("\n" + "="*60)
    print(" 1. 置信度分析 (Confidence Analysis)")
    print("="*60)
    
    for name, res in [("Train", train_res), ("Val", val_res)]:
        correct_mask = res['targets'] == res['preds']
        wrong_mask = ~correct_mask
        
        print(f"\n[{name}]")
        print(f"  Accuracy: {correct_mask.mean()*100:.1f}%")
        print(f"  Mean entropy: {res['entropy'].mean():.3f}")
        
        print(f"\n  当预测正确时:")
        if correct_mask.sum() > 0:
            print(f"    P(correct): mean={res['prob_correct'][correct_mask].mean():.4f}, "
                  f"median={np.median(res['prob_correct'][correct_mask]):.4f}")
            print(f"    P(top1):    mean={res['prob_top1'][correct_mask].mean():.4f}")
            print(f"    Entropy:    mean={res['entropy'][correct_mask].mean():.3f}")
        
        print(f"  当预测错误时:")
        if wrong_mask.sum() > 0:
            print(f"    P(correct): mean={res['prob_correct'][wrong_mask].mean():.6f}, "
                  f"median={np.median(res['prob_correct'][wrong_mask]):.6f}")
            print(f"    P(top1):    mean={res['prob_top1'][wrong_mask].mean():.4f}")
            print(f"    Entropy:    mean={res['entropy'][wrong_mask].mean():.3f}")
            
            # P(correct) 分布
            pc = res['prob_correct'][wrong_mask]
            print(f"    P(correct) 分位数:")
            for pct in [1, 5, 10, 25, 50, 75]:
                print(f"      {pct}%: {np.percentile(pc, pct):.8f}")
    
    # 比较: val 上错误预测的 confidence vs train 上错误预测
    print(f"\n  关键对比:")
    train_wrong = train_res['prob_top1'][train_res['targets'] != train_res['preds']]
    val_wrong = val_res['prob_top1'][val_res['targets'] != val_res['preds']]
    if len(train_wrong) > 0:
        print(f"    Train 错误时的 top1 confidence: {train_wrong.mean():.4f}")
    print(f"    Val   错误时的 top1 confidence: {val_wrong.mean():.4f}")
    print(f"    → 模型在 val 上犯错时{'过度自信' if val_wrong.mean() > 0.5 else '不太自信'}")


def analyze_error_types(train_res, val_res, vocab_list):
    """分析错误类型 — val 上到底错在哪？"""
    print("\n" + "="*60)
    print(" 2. 错误类型分析 (Error Type Analysis)")
    print("="*60)
    
    wrong_mask = val_res['targets'] != val_res['preds']
    
    # 高频错误词
    target_counts = defaultdict(int)
    error_counts = defaultdict(int)
    
    for i in range(len(val_res['targets'])):
        t = val_res['targets'][i]
        target_counts[t] += 1
        if wrong_mask[i]:
            error_counts[t] += 1
    
    # 按错误率排序
    print(f"\n  Val 上错误率最高的词 (至少出现 5 次):")
    word_errors = []
    for t, count in target_counts.items():
        if count >= 5:
            err_rate = error_counts.get(t, 0) / count
            word_errors.append((vocab_list[t], count, err_rate))
    
    word_errors.sort(key=lambda x: -x[2])
    for w, cnt, rate in word_errors[:15]:
        print(f"    '{w}': {rate*100:.0f}% wrong ({cnt} occurrences)")
    
    print(f"\n  Val 上错误率最低的词:")
    word_errors.sort(key=lambda x: x[2])
    for w, cnt, rate in word_errors[:10]:
        print(f"    '{w}': {rate*100:.0f}% wrong ({cnt} occurrences)")


def analyze_q_geometry(train_res, val_res):
    """Q 值几何分析 — 用 Wakhloo 框架"""
    print("\n" + "="*60)
    print(" 3. Q 值几何分析 (Wakhloo Geometric Framework)")
    print("="*60)
    
    for name, res in [("Train", train_res), ("Val", val_res)]:
        q_stats = res['q_stats']
        
        norms = [s['q_norm_mean'] for s in q_stats]
        maxes = [s['q_max'] for s in q_stats]
        n_actives = [s['n_active'] for s in q_stats]
        max_logits = [s['max_logit'] for s in q_stats]
        
        print(f"\n[{name}]")
        print(f"  Active slots: mean={np.mean(n_actives):.1f}, "
              f"max={max(n_actives)}, min={min(n_actives)}")
        print(f"  Q norm (mean per pos): mean={np.mean(norms):.3f}, "
              f"std={np.std(norms):.3f}")
        print(f"  Q max value: mean={np.mean(maxes):.3f}, "
              f"max={max(maxes):.3f}")
        print(f"  Max logit: mean={np.mean(max_logits):.1f}, "
              f"max={max(max_logits):.1f}")
    
    # 对比
    train_logits = [s['max_logit'] for s in train_res['q_stats']]
    val_logits = [s['max_logit'] for s in val_res['q_stats']]
    print(f"\n  Logit 温度对比:")
    print(f"    Train max logit: {np.mean(train_logits):.1f} ± {np.std(train_logits):.1f}")
    print(f"    Val   max logit: {np.mean(val_logits):.1f} ± {np.std(val_logits):.1f}")
    ratio = np.mean(val_logits) / max(np.mean(train_logits), 0.01)
    print(f"    比值: {ratio:.2f}x")
    if ratio > 0.8:
        print(f"    → 模型在 val 上产生了和 train 类似大小的 logit")
        print(f"      说明它对 val 也'很确信'，但确信的是错误答案 → PPL 爆炸")


@torch.no_grad()  
def analyze_representation_overlap(trainer, train_corpus, val_corpus):
    """分析 train vs val 的隐层表示是否存在分布差异"""
    print("\n" + "="*60)
    print(" 4. 表示分布分析 (Representation Distribution)")
    print("="*60)
    
    device = trainer.device
    cfg = trainer.config
    vocab = trainer._vocab
    xfr = trainer._slot_xfr
    xfr.eval()
    
    # 收集 context vectors (通过完整 forward with return_ctx)
    def get_contexts(corpus, max_n=2000):
        data = gpu_preprocess(corpus, vocab, cfg, device, verbose=False)
        contexts = []
        total = 0
        for b in iter_batches_gpu(data, 4096, device, shuffle=False):
            ret = flat_to_padded(
                b.flat_slots, b.flat_Q, b.pos_lo, b.counts, device,
                flat_is_latest=b.flat_is_latest)
            if len(ret) == 4:
                padded_slots, padded_Q, pad_mask, is_latest = ret
            else:
                padded_slots, padded_Q, pad_mask = ret
                is_latest = None
            _, ctx = xfr(padded_slots, padded_Q, pad_mask,
                         is_latest=is_latest, return_ctx=True)
            
            contexts.append(ctx.cpu())
            total += ctx.shape[0]
            if total >= max_n:
                break
        
        return torch.cat(contexts, dim=0)[:max_n]
    
    print("  Computing train contexts...", end=' ')
    train_ctx = get_contexts(train_corpus)
    print(f"shape={train_ctx.shape}")
    
    print("  Computing val contexts...", end=' ')
    val_ctx = get_contexts(val_corpus)
    print(f"shape={val_ctx.shape}")
    
    # 1. Participation Ratio (有效维度)
    def participation_ratio(X):
        """PR = (sum λ_i)^2 / sum λ_i^2"""
        X_centered = X - X.mean(dim=0, keepdim=True)
        cov = (X_centered.T @ X_centered) / X.shape[0]
        eigenvalues = torch.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues.clamp(min=0)
        pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
        return pr.item(), eigenvalues
    
    train_pr, train_eig = participation_ratio(train_ctx)
    val_pr, val_eig = participation_ratio(val_ctx)
    
    print(f"\n  Participation Ratio (有效维度, Wakhloo PR):")
    print(f"    Train: PR = {train_pr:.1f} / {train_ctx.shape[1]}")
    print(f"    Val:   PR = {val_pr:.1f} / {val_ctx.shape[1]}")
    
    # 2. Context vector norm
    train_norms = train_ctx.norm(dim=1)
    val_norms = val_ctx.norm(dim=1)
    print(f"\n  Context vector norm:")
    print(f"    Train: {train_norms.mean():.3f} ± {train_norms.std():.3f}")
    print(f"    Val:   {val_norms.mean():.3f} ± {val_norms.std():.3f}")
    
    # 3. Cosine similarity between train and val centers
    train_center = train_ctx.mean(dim=0)
    val_center = val_ctx.mean(dim=0) 
    cos_sim = F.cosine_similarity(train_center.unsqueeze(0), 
                                   val_center.unsqueeze(0)).item()
    print(f"\n  Train vs Val center cosine similarity: {cos_sim:.4f}")
    
    # 4. Eigenspectrum comparison
    train_eig_sorted = train_eig.flip(0)[:20].numpy()
    val_eig_sorted = val_eig.flip(0)[:20].numpy()
    total_train = train_eig.sum().item()
    total_val = val_eig.sum().item()
    
    print(f"\n  Top-10 eigenvalue variance explained:")
    print(f"    {'Rank':<6} {'Train%':>8} {'Val%':>8} {'Ratio':>8}")
    for i in range(10):
        t_pct = train_eig_sorted[i] / total_train * 100
        v_pct = val_eig_sorted[i] / total_val * 100
        print(f"    {i+1:<6} {t_pct:>7.1f}% {v_pct:>7.1f}% {v_pct/max(t_pct,0.01):>7.2f}x")


def analyze_predictions_detail(val_res, vocab_list, n_show=30):
    """展示具体预测样例 — 看模型的预测是否合理"""
    print("\n" + "="*60)
    print(" 5. 预测样例分析 (Prediction Examples)")
    print("="*60)
    
    wrong_mask = val_res['targets'] != val_res['preds']
    correct_mask = ~wrong_mask
    wrong_idx = np.where(wrong_mask)[0]
    correct_idx = np.where(correct_mask)[0]
    
    def show_sample(idx, tag):
        target_w = vocab_list[val_res['targets'][idx]]
        pred_w = vocab_list[val_res['preds'][idx]]
        p_correct = val_res['prob_correct'][idx]
        entropy = val_res['entropy'][idx]
        n_slots = val_res['q_stats'][idx]['n_active']
        
        # Active slot words (for identity retina, slot_id = word_id)
        slot_ids = val_res['slot_info'][idx]
        slot_words = [vocab_list[s] if s < len(vocab_list) else f'<{s}>' 
                      for s in slot_ids]
        
        # Top-5 predictions
        top5 = val_res['top5_preds'][idx]
        top5_str = ', '.join(
            f"'{vocab_list[w]}' {p:.3f}" if w < len(vocab_list) 
            else f"<{w}> {p:.3f}"
            for w, p in top5
        )
        
        context_str = ' '.join(slot_words[:20])  # show up to 20 slots
        if len(slot_words) > 20:
            context_str += f' ... (+{len(slot_words)-20})'
        
        mark = '✓' if tag == 'correct' else '✗'
        print(f"  {mark} [{n_slots} slots] {context_str}")
        print(f"    正确: '{target_w}'  预测: '{pred_w}'  "
              f"P(正确)={p_correct:.4f}  H={entropy:.2f}")
        print(f"    Top5: {top5_str}")
    
    # Show wrong predictions (most confident first — worst errors)
    print(f"\n  ── Val 错误预测 (按置信度从高到低排序) ──")
    if len(wrong_idx) > 0:
        # Sort by P(top1) descending — most confident wrong predictions first
        wrong_conf = val_res['prob_top1'][wrong_idx]
        sorted_wrong = wrong_idx[np.argsort(-wrong_conf)]
        for i, idx in enumerate(sorted_wrong[:n_show]):
            show_sample(idx, 'wrong')
        print(f"  ... 共 {len(wrong_idx)} 个错误")
    
    # Show correct predictions (least confident — hardest correct ones)
    print(f"\n  ── Val 正确预测 (最不确定的) ──")
    if len(correct_idx) > 0:
        correct_conf = val_res['prob_top1'][correct_idx]
        sorted_correct = correct_idx[np.argsort(correct_conf)]
        for i, idx in enumerate(sorted_correct[:10]):
            show_sample(idx, 'correct')
    
    # Confusion matrix: most common wrong-prediction pairs
    print(f"\n  ── 最常见的 (正确→预测) 混淆对 ──")
    confusion = defaultdict(int)
    for idx in wrong_idx:
        pair = (val_res['targets'][idx], val_res['preds'][idx])
        confusion[pair] += 1
    
    top_pairs = sorted(confusion.items(), key=lambda x: -x[1])[:20]
    for (t, p), cnt in top_pairs:
        tw = vocab_list[t] if t < len(vocab_list) else f'<{t}>'
        pw = vocab_list[p] if p < len(vocab_list) else f'<{p}>'
        print(f"    '{tw}' → '{pw}': {cnt} 次")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='EMA SlotTransformer 诊断分析')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--val', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("="*60)
    print(" EMA SlotTransformer 泛化诊断")
    print(f" Model: {args.model}")
    print("="*60)
    
    trainer, train_corpus, val_corpus = load_model_and_data(
        args.model, args.train, args.val, args.device)
    
    vocab_list = trainer._vocab.vocab_list
    
    # 1. 详细评估
    print("\n>>> Evaluating on TRAIN...")
    train_res = detailed_eval(trainer, train_corpus, 'TRAIN', max_positions=5000)
    
    print("\n>>> Evaluating on VAL...")
    val_res = detailed_eval(trainer, val_corpus, 'VAL')
    
    # 2. 置信度分析
    analyze_confidence(train_res, val_res, vocab_list)
    
    # 3. 错误类型
    analyze_error_types(train_res, val_res, vocab_list)
    
    # 4. Q 值几何
    analyze_q_geometry(train_res, val_res)
    
    # 5. 预测样例分析 (NEW)
    analyze_predictions_detail(val_res, vocab_list)
    
    # 6. 表示分布
    analyze_representation_overlap(trainer, train_corpus, val_corpus)
    
    print("\n" + "="*60)
    print(" 分析完成")
    print("="*60)


if __name__ == '__main__':
    main()
