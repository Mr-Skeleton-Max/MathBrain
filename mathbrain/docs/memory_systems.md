# B/A 双记忆系统详解

## 1. 概述

MathBrain 采用 **双记忆系统** 架构，灵感来自神经科学的 **Complementary Learning Systems** 理论 (McClelland et al., 1995)。

### 1.1 核心思想

**问题**: 如何在快速学习新知识的同时避免灾难性遗忘？

**解决方案**: 分离快速学习和慢速巩固
- **B Memory** (海马体): 快速在线学习，存储最近经验
- **A Knowledge** (新皮层): 慢速离线巩固，提取泛化知识

### 1.2 系统对比

| 特性 | B Memory | A Knowledge |
|------|----------|-------------|
| **学习速度** | 快 (在线) | 慢 (离线) |
| **容量** | 有限 (~1M 条目) | 大 (K×d×D) |
| **泛化能力** | 弱 (记忆) | 强 (知识) |
| **更新方式** | 增量 (delta rule) | 批量 (梯度下降) |
| **生物对应** | 海马体 | 新皮层 |
| **数据结构** | Hash Table | 矩阵分解 |

---

## 2. B Memory (短期记忆)

### 2.1 设计目标

1. **快速适应**: 单次遍历即可学习新模式
2. **在线更新**: 无需存储历史数据
3. **自动衰减**: 旧知识逐渐遗忘
4. **负样本抑制**: 避免错误预测

### 2.2 数据结构

**实现**: Hash Table (v2)

```python
class BMemoryHashTableV2:
    def __init__(self, config):
        self._table = {}  # (src_slot, tgt_slot) → D 维向量
        self._step_counter = {}  # 记录最后更新步数
```

**键**: `(src_slot, tgt_slot)` 对
- src_slot: 源槽位 (来自 active_slots)
- tgt_slot: 目标槽位 (来自 target_slots)

**值**: D 维向量 (默认 D=8)
- 存储 φ 的加权累积
- 范数反映关联强度

### 2.3 更新规则 (Energy-based Delta Rule)

**公式**:

$$
\begin{aligned}
\text{gap} &= 1 - (\gamma \cdot a\_\text{strength} + b\_\text{strength}) \\
\Delta b &= \eta \cdot \text{gap} \cdot \phi
\end{aligned}
$$

**参数**:
- $\eta = 0.5$: 学习率
- $\gamma = 0.9$: A 的权重 (Wake 时)
- $a\_\text{strength}$: A 的预测强度 $= A \cdot \phi$
- $b\_\text{strength}$: B 的预测强度 $= B \cdot \phi$

**直觉**:
- gap = 1: A 和 B 都没学会 → 大幅更新
- gap = 0: A 或 B 已经学会 → 不更新
- gap < 0: 过度预测 → 负向修正

**代码**:
```python
def write_batch(self, active_slots, target_slots, phi, phi_hat):
    # 计算 A 强度
    a_strength = self._compute_a_strength_vectorized(
        active_slots, target_slots, phi_hat)

    # 计算 B 强度
    b_strength = self._compute_b_strength_vectorized(
        active_slots, target_slots, phi_hat)

    # 计算 gap
    gap = 1.0 - (self.gamma * a_strength + b_strength)

    # 更新 B
    for i, tgt in enumerate(target_slots):
        for src in active_slots:
            key = (int(src), int(tgt))
            delta = self.eta * gap[i] * phi[i]
            self._table[key] = self._table.get(key, 0) + delta
```

### 2.4 负样本抑制

**问题**: 只更新正样本 (target_slots) 会导致 B 对所有词都预测高分

**解决**: 抑制非目标词

```python
# 负样本: 所有词汇槽位 - 目标槽位
neg_slots = vocab_slots - target_slots

# 抑制强度
neg_gap = -NEG_SUPPRESS * b_strength_neg

# 更新
for neg_slot in neg_slots:
    for src in active_slots:
        key = (src, neg_slot)
        delta = eta * neg_gap * phi
        self._table[key] += delta
```

**参数**: `NEG_SUPPRESS = 0.1` (默认)

### 2.5 全局衰减

**目的**: 防止 B 无限增长，自动遗忘旧知识

**实现**: Lazy Decay
```python
def _apply_lazy_decay(self, key, current_step):
    last_step = self._step_counter.get(key, 0)
    steps_elapsed = current_step - last_step

    if steps_elapsed > 0:
        decay_factor = (1 - LAMBDA_B) ** steps_elapsed
        self._table[key] *= decay_factor
        self._step_counter[key] = current_step
```

**参数**: `LAMBDA_B = 1e-4` (默认)

**效果**: 每步衰减 0.01%，1000 步后衰减至 90%

### 2.6 预测

**公式**:
$$
\text{scores}[tgt] = \sum_{src \in \text{active}} B[src, tgt] \cdot \phi
$$

**代码**:
```python
def predict(self, active_slots, phi_hat):
    scores = np.zeros(self.K, dtype=np.float32)

    for src in active_slots:
        for tgt in range(self.K):
            key = (src, tgt)
            if key in self._table:
                b_vec = self._table[key]
                scores[tgt] += np.dot(b_vec, phi_hat)

    return scores
```

**优化**: 只遍历 B 中存在的键，跳过零条目

### 2.7 裁剪 (Pruning)

**目的**: 控制 B 大小，删除弱关联

**策略**: 按范数排序，保留 Top-K

```python
def prune_by_norm(self, keep_ratio=0.8, max_entries=200000):
    # 计算所有条目的范数
    norms = [(key, np.linalg.norm(vec)) for key, vec in self._table.items()]

    # 排序
    norms.sort(key=lambda x: x[1], reverse=True)

    # 保留 Top-K
    keep_count = min(int(len(norms) * keep_ratio), max_entries)
    keep_keys = set(key for key, _ in norms[:keep_count])

    # 删除
    self._table = {k: v for k, v in self._table.items() if k in keep_keys}
```

**触发时机**: 每个 Wake-Sleep 循环的 Sleep 前

---

## 3. A Knowledge (长期知识)

### 3.1 设计目标

1. **泛化能力**: 从 B 的具体经验中提取模式
2. **参数共享**: 通过矩阵分解减少参数量
3. **可扩展性**: 支持大规模槽位 (K=65536)
4. **稳定学习**: 避免灾难性遗忘

### 3.2 架构演进

#### v1 (Hadamard 积, 已废弃)

$$
A[tgt, src] = E_Q[tgt] \odot E_V[src]
$$

**问题**:
- 强制 d = D (秩受限)
- D=8 时 d=8 (秩不够，A-only 5-7%)
- 正交约束压缩嵌入范数

#### v2 (Slice-wise 矩阵分解, 当前)

$$
A[tgt, src] = \sum_{r=1}^d E_{\text{tgt}}[tgt, r, :] \odot E_{\text{src}}[src, r, :] \quad \in \mathbb{R}^D
$$

**改进**:
- d 和 D 解耦 (d=64, D=8)
- 每个距离切片独立秩-d 分解
- AdamW weight decay 替代正交约束

**结果**: A-only 84-86% (v1: 5-7%)

### 3.3 数据结构

```python
class AKnowledgeV2:
    def __init__(self, D=8, d=64, K=65536):
        self.D = D  # φ 维度
        self.d = d  # CP rank
        self.K = K  # 槽位数

        # 嵌入矩阵 (稀疏存储)
        self._E_src = {}  # slot → (d, D) ndarray
        self._E_tgt = {}  # slot → (d, D) ndarray

        # 缓存 (密集存储, 用于快速预测)
        self._E_src_cache = None  # (n_slots, d, D)
        self._E_tgt_cache = None  # (n_slots, d, D)
```

**存储**:
- 稀疏: 只存储活跃槽位的嵌入
- 密集缓存: 预测时转换为 numpy 数组

### 3.4 预测

**公式**:
$$
\begin{aligned}
C_A &= \sum_{src \in \text{active}} E_{\text{src}}[src] \odot \phi \quad &\in \mathbb{R}^{d \times D} \\
\text{scores}[tgt] &= \sum_{r=1}^d \sum_{j=1}^D E_{\text{tgt}}[tgt, r, j] \cdot C_A[r, j] \quad &\in \mathbb{R}
\end{aligned}
$$

**代码**:
```python
def predict(self, active_slots, phi):
    # 上下文累积
    C_A = np.zeros((self.d, self.D), dtype=np.float32)
    for src in active_slots:
        if src in self._E_src:
            C_A += self._E_src[src] * phi  # (d, D) ⊙ (D,) → (d, D)

    # 预测
    scores = np.zeros(self.K, dtype=np.float32)
    for tgt in range(self.K):
        if tgt in self._E_tgt:
            scores[tgt] = np.sum(self._E_tgt[tgt] * C_A)  # Frobenius 内积

    return scores
```

**优化**: 向量化版本使用矩阵乘法

### 3.5 Sleep 更新

**目标**: 从 B 学习，同时保留旧知识

**训练样本**:
```python
samples = []
for (src, tgt), b_vec in B.items():
    a_old = A.compute_entry(src, tgt)  # (D,)
    target = gamma * a_old + b_vec     # (D,)
    samples.append((src, tgt, target))
```

**损失函数**:
$$
\mathcal{L} = \mathcal{L}_{\text{new}} + \lambda_{\text{rehearsal}} \cdot \mathcal{L}_{\text{rehearsal}}
$$

**新样本损失**:
$$
\mathcal{L}_{\text{new}} = \frac{1}{N_{\text{new}}} \sum_{i=1}^{N_{\text{new}}} \| A(src_i, tgt_i) - target_i \|^2
$$

**排练损失** (防遗忘):
$$
\mathcal{L}_{\text{rehearsal}} = \frac{1}{N_{\text{rehearsal}}} \sum_{j=1}^{N_{\text{rehearsal}}} \| A_{\text{new}}(src_j, tgt_j) - A_{\text{old}}(src_j, tgt_j) \|^2
$$

**优化器**: AdamW
```python
optimizer = optim.AdamW(
    [E_src, E_tgt],
    lr=SLEEP_LR,
    weight_decay=LAMBDA_WD  # 替代正交约束
)
```

**早停**:
- Patience: 200 epochs
- Relative tolerance: 0.0001
- Minimum epochs: 0 (允许快速收敛)

### 3.6 排练策略

**目的**: 防止 A 遗忘已学知识

**采样**:
```python
# 从 A 的现有条目中随机采样
rehearsal_samples = random.sample(
    list(A._E_src.keys()),
    n_rehearsal
)
```

**权重**: `λ_rehearsal = 0.1` (默认)

**效果**: 新知识和旧知识的平衡

---

## 4. B 和 A 的协同

### 4.1 预测融合

**公式**:
$$
\text{scores}_{\text{final}} = \text{scores}_A + \text{scores}_B
$$

**直觉**:
- A: 泛化知识 (稳定，覆盖广)
- B: 最近经验 (精确，覆盖窄)
- 融合: 结合长期知识和短期记忆

### 4.2 Wake 阶段的协同

**B 更新考虑 A**:
```python
gap = 1 - (gamma * a_strength + b_strength)
```

**含义**:
- A 已学会 → gap 小 → B 更新少
- A 没学会 → gap 大 → B 更新多
- **B 记录 A 的残差**

### 4.3 Sleep 阶段的协同

**目标设置**:
```python
target = gamma * a_old + b_vec
```

**含义**:
- 保留 90% 的 A 旧知识
- 加入 100% 的 B 新知识
- **渐进式知识迁移**

### 4.4 B Retention 参数

**作用**: 控制 Sleep 后 B 的保留比例

```python
b.decay(b_retention)  # 0=清空, 1=完全保留
```

**策略**:
- `b_retention=0` (默认): 完全清空，A 独立工作
- `b_retention=0.3`: 保留 30%，A+B 协同
- `b_retention=1.0`: 完全保留，观察 A 的减负效果

**实验**:
```bash
python train.py --corpus data.txt --mode cycle --b-retention 0.3
```

---

## 5. 性能分析

### 5.1 记忆容量

**B Memory**:
- 理论容量: K² = 65536² ≈ 43 亿条目
- 实际容量: ~1M 条目 (受内存限制)
- 有效容量: ~100K 条目 (裁剪后)

**A Knowledge**:
- 参数量: 2 × K × d × D = 2 × 65536 × 64 × 8 ≈ 67M 参数
- 内存占用: ~268 MB (float32)
- 稀疏存储: 只存储活跃槽位 (~1K slots → ~4 MB)

### 5.2 学习速度

**B Memory**:
- 单次更新: O(|active| × |target| × D)
- 典型: 10 active, 10 target, D=8 → 800 ops
- 速度: ~0.5 ms/句

**A Knowledge**:
- Sleep 训练: 10K 样本, 500 epochs
- MLX: ~30 秒
- PyTorch CPU: ~60 秒

### 5.3 泛化能力

**实验** (tinystories_10.txt, D=8):

| 配置 | top1 | top5 | 说明 |
|------|------|------|------|
| B-only | 87.0% | 99.9% | 纯记忆，无泛化 |
| A-only (10 cycles) | 84-86% | 98-99% | 泛化能力强 |
| A+B | 92-93% | 99.5% | 最佳组合 |

**观察**:
- B 在小语料上表现好 (记忆)
- A 需要多轮 Sleep 才能学会
- A+B 结合了两者优势

---

## 6. 调试与诊断

### 6.1 B Memory 诊断

```python
stats = model.get_stats()['b']
print(f"B entries: {stats['n_entries']}")
print(f"B norm: {stats['norm_mean']:.4f}")
```

**正常范围**:
- n_entries: 10K-500K
- norm_mean: 0.5-1.5

**异常情况**:
- n_entries > 1M: 需要裁剪
- norm_mean > 3.0: 可能过拟合

### 6.2 A Knowledge 诊断

```python
a_stats = model.a.stats()
print(f"A slots: {a_stats['n_embeddings']}")
print(f"E_src norm: {a_stats['E_src_norm_mean']:.4f}")
print(f"E_tgt norm: {a_stats['E_tgt_norm_mean']:.4f}")
```

**正常范围**:
- n_embeddings: 100-5000 (取决于词汇表)
- E_src/E_tgt norm: 0.3-0.8

**异常情况**:
- norm > 2.0: 可能梯度爆炸
- norm < 0.1: 可能欠拟合

### 6.3 协同诊断

**检查 B 是否记录残差**:
```python
# Wake 后
b_strength = B.predict(active_slots, phi_hat)
a_strength = A.predict(active_slots, phi)

# B 应该补充 A 的不足
correlation = np.corrcoef(b_strength, a_strength)[0, 1]
print(f"B-A correlation: {correlation:.3f}")
```

**期望**: correlation < 0.5 (B 和 A 互补)

---

## 7. 未来改进

### 7.1 B Memory

1. **自适应学习率**: 根据 gap 动态调整 η
2. **重要性采样**: 优先保留高价值条目
3. **分层存储**: 热数据 (内存) + 冷数据 (磁盘)

### 7.2 A Knowledge

1. **动态秩**: 根据任务复杂度调整 d
2. **多头注意力**: 不同头学习不同模式
3. **元学习**: 快速适应新领域

### 7.3 协同机制

1. **自适应 γ**: 根据 A 的置信度动态调整
2. **选择性巩固**: 只迁移 B 中的高质量知识
3. **主动遗忘**: 删除 A 中的低频知识

---

**版本**: v2.0
**最后更新**: 2026-03-07
