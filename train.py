#!/usr/bin/env python3
"""MathBrain 统一训练 CLI

功能：
- 支持多种训练模式（简单训练、Wake-Sleep 循环、记忆测试）
- 支持 JIT 优化
- 支持模型保存/加载
- 支持详细的训练诊断
"""

import argparse
import time
import random
import pickle
import os
import sys
import numpy as np

from mathbrain import MathBrain, MathBrainConfig


class PreprocessedCorpus:
    """预处理的语料库 - 提前完成分词和编码"""

    def __init__(self, sentences, model):
        self.sentences = []
        self.tokenized = []
        self.encoded_slots = []
        self.wake_positions = []

        print(f"预处理语料: {len(sentences)} 句")
        t0 = time.time()
        q_cache = model.q.__class__(model.config)

        for sentence in sentences:
            words = [w.strip() for w in
                     sentence.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').split()
                     if w.strip()]

            if len(words) < 2:
                continue

            # 确保所有词在词汇表中
            for word in words:
                model._ensure_word(word)

            # 编码所有词
            slots_list = []
            for word in words:
                slots = model.retina.encode(word)
                slots_list.append(slots)

            self.sentences.append(sentence)
            self.tokenized.append(words)
            self.encoded_slots.append(slots_list)

            # 预计算 Wake 阶段的 active_slots / phi / phi_hat，避免每轮重复算
            q_cache.reset()
            positions = []
            for i in range(len(words) - 1):
                q_cache.update(slots_list[i])
                active_slots, Q_vals = q_cache.get_active()
                if len(active_slots) == 0:
                    continue
                phi = model.phi_encoder.encode(Q_vals)
                norms = np.linalg.norm(phi, axis=1, keepdims=True)
                phi_hat = phi / (norms + 1e-8)
                positions.append({
                    'active_slots': active_slots.astype(np.int32, copy=False),
                    'phi': phi.astype(np.float32, copy=False),
                    'phi_hat': phi_hat.astype(np.float32, copy=False),
                    'target_counts': slots_list[i + 1],
                    'target_slots': np.fromiter(slots_list[i + 1].keys(), dtype=np.int32),
                })
            self.wake_positions.append(positions)

        preprocess_time = (time.time() - t0) * 1000
        print(f"  预处理完成: {preprocess_time:.2f}ms ({preprocess_time/len(self.sentences):.4f}ms/句)\n")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.tokenized[idx], self.encoded_slots[idx]

    def get_wake_positions(self, idx):
        return self.wake_positions[idx]

    # 注: 无需 precompute_cycle_projection，新架构 D = d，无 P 投影矩阵



def tokenize(text):
    """分词"""
    return [w.strip() for w in
            text.lower().replace('.', ' .').replace('?', ' ?').replace(',', ' ,').split()
            if w.strip()]


def load_corpus(filepath):
    """加载语料文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def evaluate_sentence(model, sentence, verbose=False):
    """评估单个句子"""
    words = tokenize(sentence)
    if len(words) < 2:
        return 0, 0, []

    correct = 0
    total = 0
    errors = []

    for i in range(len(words) - 1):
        context = words[:i+1]
        expected = words[i+1]

        preds = model.predict_next(context, k=5)
        pred = preds[0][0] if preds else "?"

        total += 1
        if pred == expected:
            correct += 1
        else:
            errors.append({
                'context': ' '.join(context),
                'expected': expected,
                'predicted': pred
            })

    return correct, total, errors


def evaluate_last_word(model, sentence, verbose=False):
    """评估单个句子：只测试最后一个词的预测"""
    words = tokenize(sentence)
    if len(words) < 2:
        return 0, 0, []

    # 只测试最后一个词
    context = words[:-1]
    expected = words[-1]

    preds = model.predict_next(context, k=5)
    pred = preds[0][0] if preds else "?"

    if pred == expected:
        return 1, 1, []
    else:
        return 0, 1, [{
            'context': ' '.join(context),
            'expected': expected,
            'predicted': pred
        }]


def evaluate_recall(model, corpus, verbose=False):
    """评估记忆准确度（逐词预测）"""
    correct = 0
    total = 0
    errors = []

    for sent in corpus:
        words = tokenize(sent)
        if len(words) < 2:
            continue

        for i in range(len(words) - 1):
            context = words[:i+1]
            expected = words[i+1]

            preds = model.predict_next(context, k=5)
            pred = preds[0][0] if preds else "?"

            total += 1
            if pred == expected:
                correct += 1
            else:
                errors.append({
                    'context': ' '.join(context),
                    'expected': expected,
                    'predicted': pred,
                    'top5': [w for w, _ in preds[:5]]
                })

    accuracy = correct / total * 100 if total > 0 else 0

    if verbose and errors:
        print(f"\n错误样例（前 10 个）:")
        for err in errors[:10]:
            print(f"  '{err['context']}' → 预期: {err['expected']}, "
                  f"预测: {err['predicted']}, Top-5: {err['top5']}")

    return accuracy, correct, total, errors


def run_simple_tests(model):
    """运行简单测试集"""
    tests = [
        (["the", "dog", "eat"], "meat"),
        (["the", "cat", "eat"], "fish"),
        (["a", "bird", "can"], "fly"),
        (["a", "mouse", "can"], "squeak"),
    ]
    correct = 0
    for ctx, expected in tests:
        preds = model.predict_next(ctx)
        pred = preds[0][0] if preds else "?"
        ok = pred == expected
        if ok:
            correct += 1
        mark = "✓" if ok else "✗"
        print(f"  '{' '.join(ctx)}' → {pred} {mark}")
    print(f"  准确率: {correct}/{len(tests)} = {100*correct/len(tests):.0f}%")
    return correct, len(tests)


def train_simple(model, corpus, epochs=3, sleep_every=0, eval_every=1, verbose=True):
    """简单训练模式（每个 epoch 训练一次语料）"""
    for epoch in range(1, epochs + 1):
        if verbose:
            print(f"{'='*70}")
            print(f"Epoch {epoch}/{epochs}")
            print(f"{'='*70}")

        random.shuffle(corpus)
        t0 = time.time()

        for sent in corpus:
            model.train_sentence(sent)

        elapsed = (time.time() - t0) * 1000
        ms_per = elapsed / len(corpus)

        stats = model.get_stats()
        if verbose:
            print(f"Wake: {len(corpus)} 句, {ms_per:.2f} ms/句, "
                  f"B={stats['b']['n_entries']}, vocab={stats['vocab_size']}")

        # 评估
        if eval_every and epoch % eval_every == 0:
            if verbose:
                print("评估:")
            run_simple_tests(model)

        # Sleep
        if sleep_every and epoch % sleep_every == 0:
            if verbose:
                print(f"\n--- Sleep 巩固 ---")
            t0 = time.time()
            result = model.sleep()
            sleep_time = (time.time() - t0) * 1000

            if result['success']:
                if verbose:
                    print(f"Sleep: {result['n_new']} 新 + {result['n_rehearsal']} 排练, "
                          f"{result['epochs']} epochs, loss={result['final_loss']:.6f}, "
                          f"耗时 {sleep_time:.0f}ms")
                    print("Sleep 后评估:")
                run_simple_tests(model)
            else:
                if verbose:
                    print(f"Sleep 失败: {result.get('error', 'unknown')}")

        if verbose:
            print()


def train_wake_sleep_cycle(model, corpus, cycles=10, wake_repeats=100, verbose=True, early_stop=False, b_retention=0.0):
    """Wake-Sleep 循环训练模式（用于结构化数据集）

    Args:
        model: MathBrain 模型
        corpus: 句子列表（每个元素是一个句子）
        cycles: Wake-Sleep 循环次数
        wake_repeats: 每个 cycle 中重复训练的次数
        verbose: 是否显示详细信息
        early_stop: 是否在达到100%准确率时早停（默认False）
        b_retention: Sleep后B Memory保留比例（0-1，默认0）
    """
    if verbose:
        print(f"语料: {len(corpus)} 句")
        print()

    # 预处理语料（总是执行，避免重复编码）
    if verbose:
        print("预处理语料...")
    preprocessed = PreprocessedCorpus(corpus, model)


    for cycle in range(1, cycles + 1):
        if verbose:
            print(f"{'='*70}")
            print(f"Cycle {cycle}/{cycles}")
            print(f"{'='*70}")

        # Wake: 重复训练多次
        t0 = time.time()
        for ep in range(wake_repeats):
            # 串行训练（使用预处理的数据）
            indices = list(range(len(preprocessed)))
            random.shuffle(indices)
            for idx in indices:
                wake_positions = preprocessed.get_wake_positions(idx)
                model.train_cached_positions(wake_positions)

            if verbose and ep in [0, 9, 49, 99] and ep < wake_repeats:
                bs = model.get_stats()['b']
                print(f"  Wake ep={ep+1}: B entries={bs['n_entries']}, "
                      f"norm={bs['norm_mean']:.4f}")

        wake_time = (time.time() - t0) * 1000

        # 评估（只在最后一个cycle或小数据集时评估）
        # if len(corpus) <= 20 or cycle == cycles:
        total_correct = 0
        total_words = 0
        for sentence in corpus:
            correct, total, _ = evaluate_sentence(model, sentence, verbose=False)
            total_correct += correct
            total_words += total

        accuracy = total_correct / total_words * 100 if total_words > 0 else 0

        if verbose:
            print(f"  Wake 结果: {total_correct}/{total_words} = {accuracy:.1f}% "
                    f"(耗时 {wake_time:.0f}ms)")
        # else:
        #     if verbose:
        #         print(f"  Wake 完成 (耗时 {wake_time:.0f}ms, 跳过评估以节省时间)")
        #     accuracy = 0  # 占位符

        # A 诊断（Wake 后、Sleep 前）
        a_stats = model.a.stats()
        if a_stats.get('has_knowledge'):
            print(f"  [A诊断] Wake后: {a_stats['n_embeddings']} slots, "
                  f"E_src norm={a_stats['E_src_norm_mean']:.4f}, "
                  f"E_tgt norm={a_stats['E_tgt_norm_mean']:.4f}")

        # 裁剪 B（在 Sleep 前）
        if hasattr(model.b, 'prune_by_norm'):
            stats_before = model.get_stats()['b']
            if verbose:
                print(f"  开始裁剪: {stats_before['n_entries']:,} 条目...")
            t_prune = time.time()
            n_pruned = model.b.prune_by_norm()
            prune_time = (time.time() - t_prune) * 1000
            if n_pruned and verbose:
                stats_after = model.get_stats()['b']
                print(f"  裁剪完成: {stats_before['n_entries']:,} → {stats_after['n_entries']:,} "
                      f"(删除 {n_pruned:,} 条, 耗时 {prune_time:.0f}ms)")

        # Sleep
        if verbose:
            print(f"  Sleep 巩固...")

        t0 = time.time()
        result = model.sleep(b_retention=b_retention)
        sleep_time = (time.time() - t0) * 1000

        if verbose:
            print(f"  Sleep: loss={result['final_loss']:.6f}, "
                  f"epochs={result['epochs']}, n_new={result['n_new']}, "
                  f"n_rehearsal={result['n_rehearsal']}, "
                  f"耗时 {sleep_time:.0f}ms")

        # A 诊断（Sleep 后）
        a_stats_after = model.a.stats()
        if a_stats_after.get('has_knowledge') and verbose:
            print(f"  [A诊断] Sleep后: {a_stats_after['n_embeddings']} slots, "
                  f"E_src norm={a_stats_after['E_src_norm_mean']:.4f}, "
                  f"E_tgt norm={a_stats_after['E_tgt_norm_mean']:.4f}")

        # Sleep 后评估
        total_correct = 0
        total_words = 0
        for sentence in corpus:
            correct, total, _ = evaluate_sentence(model, sentence, verbose=False)
            total_correct += correct
            total_words += total

        accuracy = total_correct / total_words * 100 if total_words > 0 else 0

        if verbose:
            print(f"  Sleep 后: {total_correct}/{total_words} = {accuracy:.1f}%")
            print()

        # 如果启用早停且达到100%准确度，提前结束
        if early_stop and accuracy >= 100.0:
            if verbose:
                print(f"✅ 达到 100% 准确度！在 Cycle {cycle} 完成\n")
            break

    # 最终测试
    if verbose:
        print(f"{'='*70}")
        print("最终测试")
        print(f"{'='*70}")

    total_correct = 0
    total_words = 0
    for sentence in corpus:
        correct, total, _ = evaluate_last_word(model, sentence, verbose=False)
        total_correct += correct
        total_words += total

    accuracy = total_correct / total_words * 100 if total_words > 0 else 0

    if verbose:
        print(f"\n最终准确度: {total_correct}/{total_words} = {accuracy:.1f}%")

        if accuracy >= 95.0:
            print("\n✅ 成功！达到 95% 准确度")
        elif accuracy >= 80.0:
            print(f"\n⚠️  接近目标（{accuracy:.1f}%）")
        else:
            print(f"\n❌ 未达到目标（{accuracy:.1f}%）")

    return accuracy, total_correct, total_words


def main():
    parser = argparse.ArgumentParser(
        description='MathBrain 统一训练 CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
训练模式:
  simple    简单训练（默认）- 每个 epoch 训练一次语料
  cycle     Wake-Sleep 循环训练 - 用于记忆测试，每个 cycle 重复训练多次

默认优化:
  默认启用哈希表优化版 B Memory (10x 加速，大规模稳定)
  默认启用并行共识合并训练（Parallel Consensus Merge）和预处理加速
  使用 --no-parallel 禁用并行训练，使用 --no-batch 禁用批量训练
  使用 --optimized 启用 Numba 优化版 B Memory (需要裁剪，适合小规模)

裁剪参数:
  --prune-ratio   裁剪保留比例 (默认: 0.8, 保留 Top 80%)
  --prune-max     裁剪最大条目数 (默认: 200000)
  裁剪在每个 Wake-Sleep 循环的 Sleep 前自动执行
  注意：哈希表版本不需要裁剪，可处理百万级条目

示例:
  # 简单训练（默认使用哈希表优化）
  python train.py --corpus data.txt --epochs 10 --sleep-every 2

  # Wake-Sleep 循环训练（推荐用于大规模数据）
  python train.py --corpus tinystories_200.txt --mode cycle --cycles 10

  # 使用 Cosine Chaos 编码器 (L=3, α=2, 临界点)
  python train.py --corpus tinystories_10.txt --mode cycle --cycles 10 \
                  --phi-mode chaos --chaos-n-folds 3 --chaos-alpha 2.0

  # 使用 Numba 优化版（适合小规模+裁剪场景）
  python train.py --corpus story.txt --mode cycle --optimized --cycles 10 \
                  --wake-repeats 100 --prune-ratio 0.7 --prune-max 100000

  # 使用串行训练（禁用优化）
  python train.py --corpus data.txt --no-batch --epochs 10

  # 使用 JIT 优化
  python train.py --corpus data.txt --jit --epochs 10

  # 运行内置 demo
  python train.py --demo
        """
    )

    # 基本参数
    parser.add_argument('--corpus', type=str, help='语料文件路径')
    parser.add_argument('--mode', type=str, default='simple',
                        choices=['simple', 'cycle'],
                        help='训练模式 (默认: simple)')

    # 简单训练模式参数
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮次 (simple 模式)')
    parser.add_argument('--sleep-every', type=int, default=0,
                        help='每 N 轮 Sleep (simple 模式, 0=不 Sleep)')
    parser.add_argument('--eval-every', type=int, default=1,
                        help='每 N 轮评估 (simple 模式)')

    # Wake-Sleep 循环模式参数
    parser.add_argument('--cycles', type=int, default=10,
                        help='Wake-Sleep 循环次数 (cycle 模式)')
    parser.add_argument('--wake-repeats', type=int, default=10,
                        help='每个 cycle 重复训练次数 (cycle 模式)')

    # 模型配置
    parser.add_argument('--b-mode', type=str, default='energy',
                        choices=['hebbian', 'lms', 'delta', 'energy'],
                        help='B 通道模式 (默认: energy)')
    parser.add_argument('--ngram-scales', type=str, default='3',
                        help='N-gram scales (逗号分隔, 默认: 3)')

    # Hash Retina 参数
    parser.add_argument('--K', type=int, default=65536,
                        help='Hash槽位数 (默认: 65536)')
    parser.add_argument('--ngram-size', type=int, default=3,
                        help='N-gram大小 (默认: 3)')

    # EMA 参数
    parser.add_argument('--N', type=int, default=4,
                        help='时间尺度数量 (默认: 4)')
    parser.add_argument('--rho', type=str, default='0.3,0.6,0.85,0.95',
                        help='EMA衰减率 (逗号分隔, 默认: 0.3,0.6,0.85,0.95)')

    # Phi 编码器参数
    parser.add_argument('--phi-mode', type=str, default='chaos',
                        choices=['fourier', 'chaos'],
                        help='位置编码模式 (默认: chaos)')
    parser.add_argument('--d-phi', type=int, default=8,
                        help='随机投影维度 (random/chaos模式, 默认: 8)')
    parser.add_argument('--phi-sigma', type=float, default=0.5,
                        help='随机投影标准差 (random/chaos模式, 默认: 0.5)')

    # Cosine Chaos 编码器参数 (chaos 模式)
    parser.add_argument('--chaos-n-folds', type=int, default=3,
                        help='Cosine Chaos 折叠次数 (chaos模式, 默认: 3, 推荐1-3)')
    parser.add_argument('--chaos-alpha', type=float, default=2.0,
                        help='Cosine Chaos 混沌参数 (chaos模式, 默认: 2.0, 临界点)')

    # 确定性傅里叶特征参数（fourier模式: D = d = F_ALPHA × N × 2，自动计算）
    parser.add_argument('--F-alpha', type=int, default=8,
                        help='频率基值个数 (fourier模式, 默认: 8, D=F*N*2)')
    parser.add_argument('--alpha-min', type=float, default=0.5,
                        help='最小频率 (fourier模式, 默认: 0.5)')
    parser.add_argument('--alpha-max', type=float, default=200.0,
                        help='最大频率 (fourier模式, 默认: 200.0)')

    # B 通道参数
    parser.add_argument('--eta', type=float, default=0.5,
                        help='B学习率 (默认: 0.5)')
    parser.add_argument('--lambda-b', type=float, default=1e-4,
                        help='B全局衰减率 (默认: 1e-4)')
    parser.add_argument('--neg-suppress', type=float, default=0.1,
                        help='负样本抑制强度 (默认: 0.1)')

    # A / Sleep 参数
    parser.add_argument('--cp-rank', type=int, default=64,
                        help='A 模块 CP rank (默认: 64)')
    parser.add_argument('--lambda-wd', type=float, default=1e-4,
                        help='AdamW weight decay (默认: 1e-4)')
    parser.add_argument('--sleep-global-decay', type=float, default=1.0,
                        help='Sleep前对全部A边强度做全局衰减 (默认: 1.0=关闭)')
    parser.add_argument('--gamma-decay', type=float, default=0.9,
                        help='Sleep衰减因子 (默认: 0.9, 推荐0.9-0.98)')
    parser.add_argument('--no-gamma-adaptive', action='store_true',
                        help='禁用自适应衰减，使用固定衰减 (默认: 启用自适应)')
    parser.add_argument('--gamma-initial', type=float, default=0.98,
                        help='自适应衰减初始保留率 (默认: 0.99)')
    parser.add_argument('--gamma-min', type=float, default=0.9,
                        help='自适应衰减最小保留率 (默认: 0.9)')
    parser.add_argument('--gamma-decay-rate', type=float, default=0.9,
                        help='自适应衰减速率 (默认: 0.9)')
    parser.add_argument('--lambda-ortho', type=float, default=0.3,
                        help='正交约束强度 (默认: 0.3)')
    parser.add_argument('--sleep-lr', type=float, default=0.01,
                        help='Sleep学习率 (默认: 0.01)')
    parser.add_argument('--sleep-max-epochs', type=int, default=1000,
                        help='Sleep最大轮次 (默认: 1000)')
    parser.add_argument('--sleep-min-epochs', type=int, default=0,
                        help='Sleep最小轮次 (默认: 0, 防止warm-start过早停止)')
    parser.add_argument('--sleep-patience', type=int, default=200,
                        help='Sleep早停patience (默认: 200)')
    parser.add_argument('--sleep-rel-tol', type=float, default=0.0001,
                        help='Sleep早停相对容差 (默认: 0.0001, 设0可禁用)')
    parser.add_argument('--sleep-memory-limit-gb', type=float, default=20.0,
                        help='MLX Sleep mini-batch 显存上限 (GB, 默认: 20.0)')
    parser.add_argument('--sleep-memory-utilization', type=float, default=0.9,
                        help='MLX Sleep mini-batch 启用/目标显存占用比例 (默认: 0.9)')
    parser.add_argument('--sleep-loss', type=str, default='mse',
                        choices=['mse', 'huber'],
                        help='Sleep损失函数 (默认: mse, huber可抑制异常值)')
    parser.add_argument('--sleep-huber-delta', type=float, default=1.0,
                        help='Huber loss的δ阈值 (默认: 1.0, 仅在--sleep-loss huber时生效)')
    parser.add_argument('--sleep-solver', type=str, default='adamw',
                        choices=['adamw', 'als', 'dream'],
                        help='Sleep求解器 (默认: adamw)')
    parser.add_argument('--sleep-als-iters', type=int, default=6,
                        help='ALS Sleep 最大迭代轮数 (默认: 6)')
    parser.add_argument('--sleep-als-ridge', type=float, default=1e-3,
                        help='ALS ridge 正则强度 (默认: 1e-3)')
    parser.add_argument('--sleep-als-prox', type=float, default=1.0,
                        help='ALS 贴近旧A的 proximal 强度 (默认: 1.0)')
    parser.add_argument('--sleep-dream-samples', type=int, default=512,
                        help='Dream Sleep 采样的梦境数 (默认: 512)')
    parser.add_argument('--sleep-dream-active', type=int, default=8,
                        help='每个 Dream 激活的 source 数 (默认: 8)')
    parser.add_argument('--sleep-dream-topk', type=int, default=32,
                        help='Dream teacher 候选 top-k (默认: 32)')
    parser.add_argument('--sleep-dream-phases', type=int, default=8,
                        help='Dream 周期性 phase bank 大小 (默认: 8)')
    parser.add_argument('--sleep-dream-max-proto', type=int, default=4,
                        help='每个 source 保留的 prototype 数 (默认: 4)')
    parser.add_argument('--sleep-dream-batch-size', type=int, default=128,
                        help='Dream Sleep batch size (默认: 128)')
    parser.add_argument('--sleep-dream-epochs', type=int, default=200,
                        help='Dream Sleep 最大轮次 (默认: 200)')
    parser.add_argument('--sleep-dream-temperature', type=float, default=1.5,
                        help='Dream distillation temperature (默认: 1.5)')
    parser.add_argument('--sleep-dream-prox', type=float, default=1e-3,
                        help='Dream Sleep proximal 强度 (默认: 1e-3)')

    # 裁剪参数
    parser.add_argument('--prune-ratio', type=float, default=0.8,
                        help='裁剪保留比例 (默认: 0.8, 保留 Top 80%%)')
    parser.add_argument('--prune-max', type=int, default=200000,
                        help='裁剪最大条目数 (默认: 200000)')

    # 训练控制参数
    parser.add_argument('--early-stop', action='store_true',
                        help='达到100%%准确率时早停 (默认: 不早停)')
    parser.add_argument('--b-retention', type=float, default=0.0,
                        help='Sleep后B Memory保留比例 (0-1, 默认: 0, 即清空)')

    # 模型保存/加载
    parser.add_argument('--save', type=str, help='保存模型路径')
    parser.add_argument('--load', type=str, help='加载模型路径')

    # 其他
    parser.add_argument('--demo', action='store_true', help='运行内置 demo')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--quiet', action='store_true', help='减少输出')

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 加载或创建模型
    if args.load and os.path.exists(args.load):
        with open(args.load, 'rb') as f:
            model = pickle.load(f)
        if not args.quiet:
            print(f"✅ 已加载模型: {args.load}\n")
    else:
        # 配置模型
        cfg = MathBrainConfig()

        # 基本参数
        cfg.B_MODE = args.b_mode
        cfg.K = args.K
        cfg.NGRAM_SIZE = args.ngram_size
        cfg.N = args.N

        # 解析 ngram_scales
        try:
            scales = tuple(int(x.strip()) for x in args.ngram_scales.split(','))
            cfg.NGRAM_SCALES = scales
        except ValueError:
            print(f"错误: 无效的 ngram-scales: {args.ngram_scales}")
            return 1

        # 解析 rho
        try:
            rho_values = tuple(float(x.strip()) for x in args.rho.split(','))
            if len(rho_values) != args.N:
                print(f"错误: rho值数量({len(rho_values)})必须等于N({args.N})")
                return 1
            cfg.RHO = rho_values
        except ValueError:
            print(f"错误: 无效的 rho: {args.rho}")
            return 1

        # Phi 编码器参数
        cfg.PHI_MODE = args.phi_mode
        cfg.D_PHI = args.d_phi
        cfg.PHI_SIGMA = args.phi_sigma

        # Cosine Chaos 编码器参数
        cfg.CHAOS_N_FOLDS = args.chaos_n_folds
        cfg.CHAOS_ALPHA = args.chaos_alpha

        # 确定性傅里叶特征参数（仅在 fourier 模式下使用）
        cfg.F_ALPHA = args.F_alpha
        cfg.ALPHA_MIN = args.alpha_min
        cfg.ALPHA_MAX = args.alpha_max

        # B 通道参数
        cfg.ETA = args.eta
        cfg.LAMBDA_B = args.lambda_b
        cfg.NEG_SUPPRESS = args.neg_suppress

        # A / Sleep 参数
        cfg.CP_RANK = args.cp_rank
        cfg.LAMBDA_WD = args.lambda_wd
        cfg.SLEEP_GLOBAL_DECAY = args.sleep_global_decay
        cfg.GAMMA_DECAY = args.gamma_decay
        cfg.GAMMA_ADAPTIVE = not args.no_gamma_adaptive  # 默认启用，除非显式禁用
        cfg.GAMMA_INITIAL = args.gamma_initial
        cfg.GAMMA_MIN = args.gamma_min
        cfg.GAMMA_DECAY_RATE = args.gamma_decay_rate
        cfg.LAMBDA_ORTHO = args.lambda_ortho
        cfg.SLEEP_LR = args.sleep_lr
        cfg.SLEEP_MAX_EPOCHS = args.sleep_max_epochs
        cfg.SLEEP_MIN_EPOCHS = args.sleep_min_epochs
        cfg.SLEEP_PATIENCE = args.sleep_patience
        cfg.SLEEP_REL_TOL = args.sleep_rel_tol
        cfg.SLEEP_MINIBATCH_MEMORY_LIMIT_GB = args.sleep_memory_limit_gb
        cfg.SLEEP_MINIBATCH_UTILIZATION = args.sleep_memory_utilization
        cfg.SLEEP_LOSS = args.sleep_loss
        cfg.SLEEP_HUBER_DELTA = args.sleep_huber_delta
        cfg.SLEEP_SOLVER = args.sleep_solver
        cfg.SLEEP_ALS_ITERS = args.sleep_als_iters
        cfg.SLEEP_ALS_RIDGE = args.sleep_als_ridge
        cfg.SLEEP_ALS_PROX = args.sleep_als_prox
        cfg.SLEEP_DREAM_SAMPLES = args.sleep_dream_samples
        cfg.SLEEP_DREAM_ACTIVE = args.sleep_dream_active
        cfg.SLEEP_DREAM_TOPK = args.sleep_dream_topk
        cfg.SLEEP_DREAM_PHASES = args.sleep_dream_phases
        cfg.SLEEP_DREAM_MAX_PROTO = args.sleep_dream_max_proto
        cfg.SLEEP_DREAM_BATCH_SIZE = args.sleep_dream_batch_size
        cfg.SLEEP_DREAM_EPOCHS = args.sleep_dream_epochs
        cfg.SLEEP_DREAM_TEMPERATURE = args.sleep_dream_temperature
        cfg.SLEEP_DREAM_PROX = args.sleep_dream_prox

        cfg._build_derived()

        model = MathBrain(cfg)

    # 运行训练
    if args.demo:
        # 内置 demo
        if not args.quiet:
            print("="*70)
            print("MathBrain Demo")
            print("="*70)

        templates = [
            "the {animal} eat {food} .",
            "a {animal} can {action} .",
            "{animal} is {property} .",
        ]
        data = {
            'animal': ['dog', 'cat', 'bird', 'fish'],
            'food': ['meat', 'fish', 'seeds', 'food'],
            'action': ['run', 'jump', 'fly', 'swim'],
            'property': ['big', 'small', 'fast', 'slow'],
        }
        corpus = []
        for template in templates:
            for animal in data['animal']:
                for key in ['food', 'action', 'property']:
                    if '{' + key + '}' in template:
                        for val in data[key]:
                            corpus.append(template.format(animal=animal, **{key: val}))
        corpus = list(set(corpus))

        if not args.quiet:
            print(f"语料: {len(corpus)} 句\n")

        train_simple(model, corpus, epochs=3, sleep_every=1, eval_every=1,
                     verbose=not args.quiet)

    elif args.corpus:
        # 从文件加载语料
        if not os.path.exists(args.corpus):
            print(f"错误: 语料文件不存在: {args.corpus}")
            return 1

        corpus = load_corpus(args.corpus)

        if not args.quiet:
            print("="*70)
            print("MathBrain 训练")
            print("="*70)
            print(f"语料文件: {args.corpus}")
            print(f"语料: {len(corpus)} 句")
            print(f"训练模式: {args.mode}")
            print()

        if args.mode == 'simple':
            train_simple(model, corpus, args.epochs, args.sleep_every,
                        args.eval_every, verbose=not args.quiet)

        elif args.mode == 'cycle':
            # Wake-Sleep 循环模式


            # 设置裁剪参数
            model.b._prune_keep_ratio = args.prune_ratio
            model.b._prune_max_entries = args.prune_max

            if not args.quiet:
                print(f"  裁剪参数: 保留 Top {args.prune_ratio:.0%}, 最多 {args.prune_max:,} 条")

            train_wake_sleep_cycle(model, corpus, args.cycles, args.wake_repeats,
                                  verbose=not args.quiet, early_stop=args.early_stop,
                                  b_retention=args.b_retention)

    else:
        parser.print_help()
        return 0

    # 保存模型
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        with open(args.save, 'wb') as f:
            pickle.dump(model, f)
        if not args.quiet:
            print(f"\n✅ 模型已保存: {args.save}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
