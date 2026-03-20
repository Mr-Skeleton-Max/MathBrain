# train.py 命令行使用指南

## 1. 基本用法

```bash
python train.py [选项]
```

## 2. 训练模式

### 2.1 Simple 模式 (简单训练)

**用途**: 快速实验，每个 epoch 训练一次语料

```bash
python train.py --corpus data.txt --mode simple --epochs 10
```

**参数**:
- `--epochs`: 训练轮次 (默认: 3)
- `--sleep-every`: 每 N 轮 Sleep (默认: 0, 不 Sleep)
- `--eval-every`: 每 N 轮评估 (默认: 1)

**示例**:
```bash
# 训练 10 轮，每 2 轮 Sleep 一次
python train.py --corpus data.txt --mode simple \
                --epochs 10 --sleep-every 2
```

### 2.2 Cycle 模式 (Wake-Sleep 循环)

**用途**: 记忆测试，每个 cycle 重复训练多次

```bash
python train.py --corpus data.txt --mode cycle --cycles 10
```

**参数**:
- `--cycles`: Wake-Sleep 循环次数 (默认: 10)
- `--wake-repeats`: 每个 cycle 重复训练次数 (默认: 10)
- `--early-stop`: 达到 100% 准确率时早停
- `--b-retention`: Sleep 后 B 保留比例 (默认: 0)

**示例**:
```bash
# 10 个循环，每个循环训练 100 次
python train.py --corpus tinystories_10.txt --mode cycle \
                --cycles 10 --wake-repeats 100

# 启用早停
python train.py --corpus data.txt --mode cycle \
                --cycles 20 --early-stop

# Sleep 后保留 30% 的 B
python train.py --corpus data.txt --mode cycle \
                --b-retention 0.3
```

---

## 3. 模型配置

### 3.1 Hash Retina

```bash
--K 65536              # 槽位数 (默认: 65536)
--ngram-size 3         # N-gram 大小 (默认: 3)
--ngram-scales 3       # N-gram 尺度 (默认: 3, 可用 1,3,5)
```

**示例**:
```bash
# 使用多尺度 N-gram
python train.py --corpus data.txt --ngram-scales 1,3,5
```

### 3.2 EMA 上下文

```bash
--N 4                  # 时间尺度数 (默认: 4)
--rho 0.3,0.6,0.85,0.95  # EMA 衰减率 (默认)
```

**示例**:
```bash
# 使用偏慢的时间尺度
python train.py --corpus data.txt --rho 0.5,0.7,0.9,0.97
```

### 3.3 位置编码

#### Random 模式 (默认)

```bash
--phi-mode random      # 随机投影
--d-phi 8              # φ 维度 (默认: 8)
--phi-sigma 0.5        # 投影标准差 (默认: 0.5)
```

#### Fourier 模式

```bash
--phi-mode fourier     # 确定性傅里叶特征
--F-alpha 32           # 频率基值个数 (默认: 32)
--alpha-min 0.5        # 最小频率 (默认: 0.5)
--alpha-max 200.0      # 最大频率 (默认: 200.0)
```

**输出维度**: D = F_ALPHA × N × 2

#### Cosine Chaos 模式 (新)

```bash
--phi-mode chaos       # Cosine Chaos 编码器
--d-phi 8              # φ 维度 (默认: 8)
--chaos-n-folds 3      # 折叠次数 (默认: 3, 推荐 1-3)
--chaos-alpha 2.0      # 混沌参数 (默认: 2.0, 临界点)
```

**示例**:
```bash
# 使用 Cosine Chaos (L=3, α=2, 临界点)
python train.py --corpus data.txt --mode cycle \
                --phi-mode chaos --chaos-n-folds 3 --chaos-alpha 2.0

# 使用 L=1, α=π (频率缩放)
python train.py --corpus data.txt --mode cycle \
                --phi-mode chaos --chaos-n-folds 1 --chaos-alpha 3.14
```

### 3.4 B Memory

```bash
--b-mode energy        # B 模式 (默认: energy)
--eta 0.5              # 学习率 (默认: 0.5)
--lambda-b 1e-4        # 全局衰减率 (默认: 1e-4)
--neg-suppress 0.1     # 负样本抑制强度 (默认: 0.1)
```

**B 模式**:
- `energy`: Energy-based delta rule (推荐)
- `delta`: 简单 delta rule
- `lms`: LMS 算法
- `hebbian`: Hebbian 学习

**示例**:
```bash
# 增加学习率和衰减
python train.py --corpus data.txt --eta 0.8 --lambda-b 1e-3
```

### 3.5 A Knowledge

```bash
--cp-rank 64           # Slice-wise 秩 (默认: 64)
--lambda-wd 1e-4       # AdamW weight decay (默认: 1e-4)
```

**示例**:
```bash
# 增加 A 的表达能力
python train.py --corpus data.txt --cp-rank 128
```

### 3.6 Sleep 参数

```bash
--gamma-decay 0.9      # 衰减因子 (默认: 0.9)
--sleep-lr 0.01        # 学习率 (默认: 0.01)
--sleep-max-epochs 1000  # 最大轮次 (默认: 1000)
--sleep-min-epochs 0   # 最小轮次 (默认: 0)
--sleep-patience 200   # 早停 patience (默认: 200)
--sleep-rel-tol 0.0001 # 早停相对容差 (默认: 0.0001)
```

**自适应衰减** (推荐):
```bash
--gamma-initial 0.98   # 初始保留率 (默认: 0.98)
--gamma-min 0.9        # 最小保留率 (默认: 0.9)
--gamma-decay-rate 0.9 # 衰减速率 (默认: 0.9)
--no-gamma-adaptive    # 禁用自适应 (使用固定 gamma-decay)
```

**损失函数**:
```bash
--sleep-loss mse       # MSE 损失 (默认)
--sleep-loss huber     # Huber 损失 (抑制异常值)
--sleep-huber-delta 1.0  # Huber δ 阈值 (默认: 1.0)
```

**示例**:
```bash
# 使用 Huber 损失
python train.py --corpus data.txt --mode cycle \
                --sleep-loss huber --sleep-huber-delta 0.5

# 禁用自适应衰减
python train.py --corpus data.txt --mode cycle \
                --no-gamma-adaptive --gamma-decay 0.95
```

### 3.7 裁剪参数

```bash
--prune-ratio 0.8      # 保留比例 (默认: 0.8, 保留 Top 80%)
--prune-max 200000     # 最大条目数 (默认: 200000)
```

**触发时机**: 每个 Wake-Sleep 循环的 Sleep 前

**示例**:
```bash
# 更激进的裁剪
python train.py --corpus data.txt --mode cycle \
                --prune-ratio 0.5 --prune-max 100000
```

---

## 4. 模型保存/加载

### 4.1 保存模型

```bash
python train.py --corpus data.txt --mode cycle --save model.pkl
```

**保存内容**:
- B Memory (所有条目和权重)
- A Knowledge (E_src, E_tgt 嵌入)
- 词汇表 (vocab, word_to_slots)
- Q State (当前上下文)
- φ Encoder (投影矩阵)

### 4.2 加载模型

```bash
python train.py --load model.pkl --corpus test.txt --mode cycle
```

**用途**:
- 断点续训
- 模型评估
- 迁移学习

### 4.3 示例

```bash
# 训练并保存
python train.py --corpus train.txt --mode cycle --cycles 10 --save model.pkl

# 加载并继续训练
python train.py --load model.pkl --corpus train.txt --mode cycle --cycles 5

# 加载并评估
python train.py --load model.pkl --corpus test.txt --mode cycle --cycles 1
```

---

## 5. 其他选项

### 5.1 随机种子

```bash
--seed 42              # 随机种子 (默认: 42)
```

**用途**: 保证实验可复现

### 5.2 输出控制

```bash
--quiet                # 减少输出
```

**示例**:
```bash
# 静默模式
python train.py --corpus data.txt --mode cycle --quiet
```

### 5.3 内置 Demo

```bash
python train.py --demo
```

**功能**: 运行内置的动物-食物-动作模板数据集

---

## 6. 完整示例

### 6.1 小语料记忆测试

```bash
python train.py \
  --corpus datasets/tinystories/tinystories_10.txt \
  --mode cycle \
  --cycles 10 \
  --wake-repeats 100 \
  --phi-mode chaos \
  --chaos-n-folds 3 \
  --chaos-alpha 2.0 \
  --cp-rank 64 \
  --save models/tinystories_10.pkl
```

### 6.2 大语料训练

```bash
python train.py \
  --corpus datasets/tinystories/tinystories_200.txt \
  --mode cycle \
  --cycles 20 \
  --wake-repeats 10 \
  --phi-mode random \
  --d-phi 8 \
  --cp-rank 128 \
  --prune-ratio 0.7 \
  --prune-max 500000 \
  --save models/tinystories_200.pkl
```

### 6.3 实验对比

```bash
# 基线 (Random 编码器)
python train.py --corpus data.txt --mode cycle --cycles 10 \
                --phi-mode random --save baseline.pkl

# Cosine Chaos (L=3, α=2)
python train.py --corpus data.txt --mode cycle --cycles 10 \
                --phi-mode chaos --chaos-n-folds 3 --chaos-alpha 2.0 \
                --save chaos_l3_a2.pkl

# Cosine Chaos (L=1, α=π)
python train.py --corpus data.txt --mode cycle --cycles 10 \
                --phi-mode chaos --chaos-n-folds 1 --chaos-alpha 3.14 \
                --save chaos_l1_api.pkl
```

### 6.4 调试配置

```bash
python train.py \
  --corpus data.txt \
  --mode cycle \
  --cycles 5 \
  --wake-repeats 10 \
  --sleep-max-epochs 100 \
  --sleep-patience 20 \
  --b-retention 0.5 \
  --quiet
```

---

## 7. 性能优化建议

### 7.1 小语料 (≤20 句)

```bash
--mode cycle
--cycles 10
--wake-repeats 100
--phi-mode chaos --chaos-n-folds 1 --chaos-alpha 3.14
--cp-rank 64
```

**理由**: 短句不需要长程依赖，L=1 α=π 最优

### 7.2 中等语料 (20-200 句)

```bash
--mode cycle
--cycles 10
--wake-repeats 10
--phi-mode chaos --chaos-n-folds 3 --chaos-alpha 2.0
--cp-rank 128
--prune-ratio 0.8
```

**理由**: 需要长程区分度，L=3 α=2 延长位置窗口

### 7.3 大语料 (>200 句)

```bash
--mode cycle
--cycles 20
--wake-repeats 5
--phi-mode random --d-phi 16
--cp-rank 256
--prune-ratio 0.7
--prune-max 500000
```

**理由**: 增加 D 和 d 提升容量，激进裁剪控制内存

---

## 8. 常见问题

### Q1: 训练很慢怎么办？

**A**:
1. 减少 `--wake-repeats` (默认 10)
2. 减少 `--sleep-max-epochs` (默认 1000)
3. 使用 MLX 后端 (Apple Silicon)
4. 增加 `--sleep-batch-size` (默认 5000)

### Q2: B Memory 爆炸 (>1M 条目)

**A**:
1. 启用裁剪: `--prune-ratio 0.7 --prune-max 200000`
2. 增加衰减: `--lambda-b 1e-3`
3. 减少 `--wake-repeats`

### Q3: A-only 准确率很低 (<10%)

**A**:
1. 增加 CP rank: `--cp-rank 128`
2. 检查 Sleep 是否收敛 (查看 loss)
3. 增加 γ: `--gamma-decay 0.95`
4. 增加训练轮次: `--cycles 20`

### Q4: 如何选择 φ 编码器？

**A**:
- **Random**: 默认，适合大多数场景
- **Chaos L=1 α=π**: 小语料 (≤20 句)
- **Chaos L=3 α=2**: 中大语料，需要长程依赖
- **Fourier**: 需要可解释性

### Q5: 如何调试模型？

**A**:
1. 查看 B 统计: `B entries`, `norm`
2. 查看 A 统计: `n_embeddings`, `E_src/E_tgt norm`
3. 使用 `--b-retention 0.5` 观察 A 的减负效果
4. 减少 `--cycles` 快速迭代

---

## 9. 参数速查表

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `--cycles` | 10 | 5-20 | Wake-Sleep 循环次数 |
| `--wake-repeats` | 10 | 5-100 | 每个 cycle 重复训练次数 |
| `--d-phi` | 8 | 8-32 | φ 维度 |
| `--cp-rank` | 64 | 64-256 | A 模块秩 |
| `--eta` | 0.5 | 0.3-0.8 | B 学习率 |
| `--lambda-b` | 1e-4 | 1e-5 - 1e-3 | B 衰减率 |
| `--gamma-decay` | 0.9 | 0.85-0.98 | Sleep 衰减因子 |
| `--sleep-lr` | 0.01 | 0.005-0.05 | Sleep 学习率 |
| `--prune-ratio` | 0.8 | 0.5-0.9 | 裁剪保留比例 |
| `--b-retention` | 0.0 | 0.0-1.0 | Sleep 后 B 保留比例 |

---

**版本**: v2.0
**最后更新**: 2026-03-07
