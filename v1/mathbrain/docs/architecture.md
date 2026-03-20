# MathBrain 整体架构

## 1. 系统概览

MathBrain 是一个在线序列预测模型，采用 **双记忆系统 + Wake-Sleep 学习** 架构。

### 1.1 核心组件

```
输入序列: w₁, w₂, w₃, ...
    ↓
┌─────────────────────────────────────────────────┐
│  Retina: Hash 编码 (N-gram → slot IDs)         │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  Q State: EMA 上下文 (4 时间尺度)               │
│  Q_n(t+1) = ρ_n·Q_n(t) + (1-ρ_n)·x(t)          │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  φ Encoder: 位置编码                            │
│  φ = encode(Q)  [Random / Fourier / Chaos]     │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  双记忆系统                                      │
│  ┌─────────────────┐  ┌─────────────────┐      │
│  │  B Memory       │  │  A Knowledge    │      │
│  │  (短期/快速)    │  │  (长期/慢速)    │      │
│  │  Hash Table     │  │  Slice-wise CP  │      │
│  └─────────────────┘  └─────────────────┘      │
│         ↓                      ↓                 │
│    scores_B(φ)          scores_A(φ)             │
│         └──────────┬──────────┘                 │
│                    ↓                             │
│            scores = A + B                        │
└─────────────────────────────────────────────────┘
    ↓
预测: argmax(scores)
```

### 1.2 数据流

**前向预测** (Inference):
1. 输入词序列 → Retina 编码 → slot IDs
2. Q State 更新 → 获取活跃 slots 和 Q 值
3. φ Encoder → 位置特征 φ
4. A 预测: scores_A = A(active_slots, φ)
5. B 预测: scores_B = B(active_slots, φ)
6. 合并: scores = scores_A + scores_B
7. 解码: slot IDs → 词汇

**Wake 训练** (Online Learning):
1. 前向预测获取 φ, active_slots
2. 目标词 → target_slots
3. B 更新: B.write(active_slots, target_slots, φ)
   - Energy-based delta rule
   - gap = 1 - (γ·a_strength + b_strength)
   - Δb = η · gap · φ

**Sleep 巩固** (Offline Consolidation):
1. 收集 B 条目 → 训练样本
2. 目标: target = γ·A_old + B
3. 优化 A: minimize ||A_new - target||²
4. 排练: 防止 A 遗忘旧知识
5. B 衰减/清空

---

## 2. 组件详解

### 2.1 Retina (词汇编码)

**文件**: `retina.py`

**功能**: 将词转换为固定大小的 slot ID 集合

**实现**: Hash Retina V2
- N-gram 哈希: 字符级 3-gram
- 多尺度: 支持 (1,3,5) 或单尺度 (3)
- 槽位数: K=65536 (可配置)

**示例**:
```python
retina = HashRetinaV2(config)
slots = retina.encode("dog")  # {12345, 23456, 34567, ...}
```

**优点**:
- 固定维度 (不随词汇表增长)
- 自动处理 OOV
- 字符级泛化

### 2.2 Q State (EMA 上下文)

**文件**: `q_state.py`

**功能**: 维护多时间尺度的上下文状态

**公式**:
$$Q_n(t+1) = \rho_n \cdot Q_n(t) + (1-\rho_n) \cdot x(t)$$

**默认配置**:
- N=4 个时间尺度
- ρ = (0.3, 0.6, 0.85, 0.95)
- 半衰期: 0.6, 1.4, 4.3, 13.5 步

**特性**:
- 快尺度: 近端精细区分
- 慢尺度: 远端粗略区分
- 自然的注意力衰减

**API**:
```python
q = QState(config)
q.update(slots)  # 更新状态
active_slots, Q_vals = q.get_active()  # 获取活跃槽位和 Q 值
```

### 2.3 φ Encoder (位置编码)

**文件**: `phi_encoder.py`, `phi_encoder_chaos.py`

**功能**: 将 Q 状态投影到 D 维特征空间

**三种模式**:

#### 2.3.1 Random (默认)
```python
φ = cos(P · (α_scale ⊙ Q))
```
- P: 随机高斯矩阵 (D, N)
- α_scale: 反向缩放权重 (快尺度高频)
- 输出: D=8 维

#### 2.3.2 Fourier
```python
φ = [cos(α_f · w ⊙ Q), sin(α_f · w ⊙ Q)]  for f=1..F
```
- F 个几何间距频率
- 输出: D = F × N × 2

#### 2.3.3 Cosine Chaos (新)
```python
M_0 = 0
M_t = cos(α · (W_shift · M_{t-1} + E))  for t=1..L
φ = M_L
```
- 递归折叠 L 次
- α=2 为临界点 (Lyapunov λ=0)
- 延长有效位置窗口 2x

**选择指南**:
- **Random**: 默认，快速，适合大多数场景
- **Fourier**: 确定性，可解释性强
- **Chaos**: 长序列，需要远程依赖

### 2.4 B Memory (短期记忆)

**文件**: `b_memory_hashtable_v2.py`

**功能**: 快速在线学习，存储 (src, tgt) → vector 映射

**实现**: Hash Table
- 键: (src_slot, tgt_slot) 对
- 值: D 维向量
- 容量: 百万级条目

**更新规则** (Energy-based):
```python
gap = 1 - (γ·a_strength + b_strength)
Δb = η · gap · φ
```

**特性**:
- 在线增量学习
- 自动衰减 (λ_B)
- 负样本抑制

**API**:
```python
b = BMemoryV2Wrapper(config)
b.write_batch(active_slots, target_slots, phi, phi_hat)
scores = b.predict(active_slots, phi_hat)
```

### 2.5 A Knowledge (长期知识)

**文件**: `a_knowledge_v2.py`

**功能**: 慢速离线巩固，泛化能力强

**实现**: Slice-wise 矩阵分解
```python
A[tgt, src] = Σ_r E_tgt[tgt, r, :] * E_src[src, r, :]  → (D,)
```

**参数**:
- E_src[slot]: (d, D) 矩阵
- E_tgt[slot]: (d, D) 矩阵
- d=64 (CP rank, 可配置)
- D=8 (φ 维度)

**优点**:
- d 和 D 解耦 (v1 强制 d=D)
- AdamW weight decay (v1 正交约束)
- 更强的表达能力

**API**:
```python
a = AKnowledgeV2(D=8, d=64, K=65536)
scores = a.predict(active_slots, phi)
a.update_from_sleep(E_src_dict, E_tgt_dict)  # Sleep 后更新
```

---

## 3. Wake-Sleep 训练

### 3.1 Wake 阶段 (在线学习)

**目标**: 快速适应新数据

**流程**:
```python
for sentence in corpus:
    for i in range(len(words) - 1):
        context = words[:i+1]
        target = words[i+1]

        # 前向
        active_slots, Q = q.get_active()
        phi = phi_encoder.encode(Q)

        # B 更新
        target_slots = retina.encode(target)
        b.write_batch(active_slots, target_slots, phi, phi_hat)
```

**特点**:
- 单次遍历 (one-shot)
- 增量更新
- 无需梯度

### 3.2 Sleep 阶段 (离线巩固)

**目标**: 将 B 知识迁移到 A，防止遗忘

**流程**:
```python
# 1. 收集训练样本
samples = []
for (src, tgt), b_vec in B.items():
    a_old = A.compute_entry(src, tgt)
    target = γ * a_old + b_vec
    samples.append((src, tgt, target))

# 2. 优化 A
for epoch in range(max_epochs):
    # 新样本
    loss_new = ||A_new(src, tgt) - target||²

    # 排练 (防遗忘)
    loss_rehearsal = ||A_new(src_old, tgt_old) - A_old(src_old, tgt_old)||²

    # 总损失
    loss = loss_new + λ_rehearsal * loss_rehearsal

    # AdamW 更新
    optimizer.step()

# 3. B 衰减/清空
B.decay(b_retention)  # 默认 b_retention=0 (清空)
```

**关键参数**:
- γ=0.9: 衰减因子 (保留 90% 旧知识)
- λ_rehearsal: 排练权重
- b_retention=0: Sleep 后清空 B

**后端**:
- **MLX** (默认): Apple Silicon GPU 加速
- **PyTorch**: 通用 CPU/CUDA

---

## 4. 配置系统

**文件**: `config.py`

**核心参数**:

```python
@dataclass
class MathBrainConfig:
    # Hash Retina
    K: int = 65536              # 槽位数
    NGRAM_SIZE: int = 3         # N-gram 大小

    # EMA
    N: int = 4                  # 时间尺度数
    RHO: tuple = (0.3, 0.6, 0.85, 0.95)

    # φ 编码
    PHI_MODE: str = "random"    # "random" | "fourier" | "chaos"
    D_PHI: int = 8              # φ 维度
    PHI_SIGMA: float = 0.5      # 随机投影标准差

    # Cosine Chaos
    CHAOS_N_FOLDS: int = 3      # 折叠次数
    CHAOS_ALPHA: float = 2.0    # 混沌参数

    # B Memory
    ETA: float = 0.5            # 学习率
    LAMBDA_B: float = 1e-4      # 全局衰减

    # A Knowledge
    CP_RANK: int = 64           # Slice-wise 秩
    LAMBDA_WD: float = 1e-4     # AdamW weight decay

    # Sleep
    GAMMA_DECAY: float = 0.9    # 衰减因子
    SLEEP_LR: float = 0.01      # 学习率
    SLEEP_MAX_EPOCHS: int = 1000
    SLEEP_PATIENCE: int = 200   # 早停
```

**派生参数** (自动计算):
```python
INVERSE_WEIGHT = (1-ρ) / mean(1-ρ)  # 反向缩放
d = D_PHI  # φ 维度 (random/chaos 模式)
```

---

## 5. 模型保存/加载

**保存**:
```bash
python train.py --corpus data.txt --mode cycle --save model.pkl
```

**加载**:
```bash
python train.py --load model.pkl --corpus test.txt --mode cycle
```

**包含内容**:
- B Memory (所有条目和权重)
- A Knowledge (E_src, E_tgt 嵌入)
- 词汇表 (vocab, word_to_slots)
- Q State (当前上下文)
- φ Encoder (投影矩阵)

**注意**: 使用 `pickle` 序列化，模型文件较大 (取决于 B 条目数)

---

## 6. 性能特征

### 6.1 计算复杂度

**前向预测**:
- Retina: O(|word| × ngram_size)
- Q State: O(N × |active_slots|)
- φ Encoder: O(N × D)
- B Predict: O(|active_slots| × |vocab_slots| × D)
- A Predict: O(|active_slots| × |vocab_slots| × d × D)

**瓶颈**: B/A 预测 (与词汇表大小成正比)

**优化**:
- 预处理: 缓存 φ, active_slots
- 批量训练: 向量化操作
- 哈希表: O(1) B 查找

### 6.2 内存占用

**B Memory**:
- 每条目: ~40 bytes (key + D×4 bytes)
- 100K 条目: ~4 MB
- 1M 条目: ~40 MB

**A Knowledge**:
- E_src + E_tgt: 2 × K × d × D × 4 bytes
- K=65536, d=64, D=8: ~512 MB

**总计**: ~500-600 MB (典型配置)

### 6.3 训练速度

**Wake** (tinystories_10.txt, 10 句):
- 预处理: ~50 ms
- 100 epochs: ~5 秒
- ~0.5 ms/句/epoch

**Sleep**:
- 10K 样本, 500 epochs: ~30 秒 (MLX)
- ~60 秒 (PyTorch CPU)

---

## 7. 扩展性

### 7.1 支持的任务

- **序列预测**: 下一个词预测
- **语言建模**: 困惑度评估
- **记忆测试**: 完形填空

### 7.2 可扩展点

1. **新的 φ 编码器**: 继承 `PhiEncoder` 基类
2. **新的 B 更新规则**: 修改 `BMemory.write_batch()`
3. **新的 Sleep 策略**: 实现 `consolidate_*()` 函数
4. **多模态输入**: 扩展 Retina 支持图像/音频

### 7.3 已知限制

- **词汇表大小**: 预测复杂度 O(V)，大词汇表慢
- **长序列**: EMA 衰减限制有效窗口 (~30 步)
- **冷启动**: A 需要多轮 Sleep 才能学会

---

## 8. 调试与诊断

### 8.1 日志输出

**Wake**:
```
Wake ep=1: B entries=247371, norm=1.0845
```

**Sleep**:
```
Sleep: loss=0.123456, epochs=500, n_new=10000, n_rehearsal=5000
[A诊断] Sleep后: 1234 slots, E_src norm=0.5678, E_tgt norm=0.5432
```

### 8.2 常见问题

**Q: A-only 准确率很低 (<10%)**
- 检查: CP_RANK 是否太小 (推荐 ≥64)
- 检查: Sleep 是否收敛 (loss 下降)
- 检查: γ 是否太小 (推荐 0.9-0.98)

**Q: B 条目数爆炸 (>1M)**
- 启用裁剪: `--prune-ratio 0.8`
- 增加衰减: `--lambda-b 1e-3`

**Q: Sleep 很慢**
- 使用 MLX 后端 (Apple Silicon)
- 减少 max_epochs: `--sleep-max-epochs 500`
- 增加 batch_size: `--sleep-batch-size 10000`

---

## 9. 与其他模型对比

| 特性 | MathBrain | Transformer | RNN/LSTM |
|------|-----------|-------------|----------|
| 在线学习 | ✓ (Wake) | ✗ | ✗ |
| 增量更新 | ✓ | ✗ | ✓ |
| 长期记忆 | ✓ (A) | ✗ | ✗ |
| 位置编码 | EMA + φ | Sinusoidal/RoPE | 隐式 |
| 计算复杂度 | O(V) | O(L²) | O(L) |
| 可解释性 | 高 | 低 | 中 |

**优势**:
- 真正的在线学习 (无需重训练)
- 双记忆系统 (快速适应 + 长期保留)
- 神经科学可解释性

**劣势**:
- 预测复杂度 O(V) (大词汇表慢)
- 需要 Wake-Sleep 循环 (训练复杂)
- 有效窗口受 EMA 限制

---

**版本**: v2.0
**最后更新**: 2026-03-07
