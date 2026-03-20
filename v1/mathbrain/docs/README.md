# MathBrain 架构文档

本目录包含 MathBrain 神经符号序列预测模型的完整架构文档。

## 文档索引

### 数学规范
- **[mathematical_specification.md](mathematical_specification.md)** - 完整数学定义与公式推导

### 核心架构
- **[architecture.md](architecture.md)** - 整体架构与数据流
- **[memory_systems.md](memory_systems.md)** - B/A 双记忆系统详解

### 组件详解
- **[phi_encoding.md](phi_encoding.md)** - 位置编码器 (Random, Fourier, Cosine Chaos)

### 训练与评估
- **[cli_usage.md](cli_usage.md)** - train.py 命令行使用指南

## 快速开始

### 基本概念

**MathBrain** 是一个受神经科学启发的在线序列预测模型，核心特点：

1. **双记忆系统**
   - **B Memory** (短期记忆): 快速在线学习，类似海马体
   - **A Knowledge** (长期知识): 慢速离线巩固，类似新皮层

2. **Wake-Sleep 学习**
   - **Wake**: 在线增量学习到 B
   - **Sleep**: 离线巩固 B → A，伪排练防遗忘

3. **位置编码**
   - **EMA 上下文**: 4 个时间尺度的指数移动平均
   - **φ 编码器**: 将 EMA 状态投影到特征空间
   - **Cosine Chaos**: 基于混沌边缘理论的递归编码

### 代码结构

```
mathbrain/
├── model.py                    # MathBrain 主模型
├── config.py                   # 配置管理
│
├── retina.py                   # Hash Retina 词汇编码
├── q_state.py                  # EMA 上下文状态
├── phi_encoder.py              # 位置编码器 (Random, Fourier)
├── phi_encoder_chaos.py        # Cosine Chaos 编码器
│
├── b_memory_hashtable_v2.py    # B Memory (哈希表实现)
├── a_knowledge_v2.py           # A Knowledge (slice-wise 分解)
│
├── sleep_v2_mlx.py             # Sleep 巩固 (MLX 后端)
├── sleep_v2.py                 # Sleep 巩固 (PyTorch 后端)
│
└── docs/                       # 本文档目录
```

### 使用示例

```bash
# 基本训练
python train.py --corpus data.txt --mode cycle --cycles 10

# 使用 Cosine Chaos 编码器
python train.py --corpus data.txt --mode cycle \
                --phi-mode chaos --chaos-n-folds 3 --chaos-alpha 2.0

# 保存/加载模型
python train.py --corpus data.txt --mode cycle --save model.pkl
python train.py --load model.pkl --corpus test.txt --mode cycle
```

## 版本历史

### v2 (当前版本, 2026-03)

**核心改进**:
- **Slice-wise 矩阵分解**: A 模块从 Hadamard 积改为独立秩分解，d 和 D 解耦
- **AdamW 替代正交约束**: 使用 weight decay 替代显式正交约束
- **Cosine Chaos 编码器**: 基于 Lyapunov 分析的递归位置编码
- **MLX 后端**: 支持 Apple Silicon GPU 加速

**性能**:
- A-only: 84-86% (v1: 5-7%)
- A+B: 92-93% (tinystories_10.txt, D=8)

### v1 (基线版本)

- Hadamard 积: A[tgt,src] = E_Q[tgt] ⊙ E_V[src]
- 正交约束: 显式 Gram 矩阵惩罚
- 随机投影编码器

## 理论基础

### 神经科学启发

1. **Complementary Learning Systems** (McClelland et al., 1995)
   - 快速学习系统 (海马体) ↔ B Memory
   - 慢速学习系统 (新皮层) ↔ A Knowledge

2. **TILT Model** (Howard et al., 2014)
   - Laplace Neural Manifold ↔ EMA 上下文
   - 时间尺度分离 ↔ 多尺度 ρ

3. **Edge of Chaos** (Bertschinger & Natschläger, 2004)
   - 混沌边缘计算 ↔ Cosine Chaos 编码器
   - Lyapunov 指数 λ=0 ↔ α*=2

### 数学工具

- **CP 分解**: CANDECOMP/PARAFAC 张量分解
- **Vandermonde 矩阵**: 逆 Laplace 变换
- **随机投影**: Johnson-Lindenstrauss 引理
- **动力系统**: Lyapunov 指数，混沌理论

## 参考文献

1. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419.

2. Howard, M. W., et al. (2014). A distributed representation of temporal context. *Journal of Mathematical Psychology*, 69, 269-299.

3. Bertschinger, N., & Natschläger, T. (2004). Real-time computation at the edge of chaos in recurrent neural networks. *Neural Computation*, 16(7), 1413-1436.

4. Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. *SIAM Review*, 51(3), 455-500.

## 贡献者

MathBrain 研究组

## 许可证

研究用途
