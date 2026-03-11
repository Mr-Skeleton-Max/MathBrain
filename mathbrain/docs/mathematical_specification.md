# MathBrain v2: 数学规范

> 本文档提供 MathBrain v2 架构的完整数学定义。实现细节见 architecture.md，使用指南见 cli_usage.md。

---

## 0. 符号总表

### 0.1 维度与索引

| 符号 | 含义 | 典型值 |
|------|------|--------|
| $V$ | 词表大小 | $10^3$–$10^5$ |
| $K$ | 槽位空间大小 (Hash Retina) | 65536 |
| $n_{\text{gram}}$ | N-gram 大小 | 3 (trigram) |
| $N$ | EMA 时间尺度数 | 4 |
| $D$ | φ 特征维度 | 8 |
| $d$ | A 模块 CP 秩 | 64 |
| $i, j$ | 槽位索引 ($0 \leq i < K$) | |
| $k$ | 时间尺度索引 ($0 \leq k < N$) | |
| $r$ | CP 分量索引 ($0 \leq r < d$) | |

### 0.2 核心数据结构

| 符号 | 形状 | 含义 | 持久性 |
|------|------|------|--------|
| $Q$ | $K \times N$ (稀疏) | EMA 上下文状态 | 会话内 |
| $\varphi$ | $K \times D$ (稀疏) | 位置特征向量 | 从 Q 实时计算 |
| $B$ | $(K \times K) \to \mathbb{R}^D$ (Hash Table) | 短期记忆 | 会话内 (Sleep 后衰减) |
| $E_{\text{src}}$ | $K \to \mathbb{R}^{d \times D}$ (稀疏字典) | A 源嵌入 | 长期 |
| $E_{\text{tgt}}$ | $K \to \mathbb{R}^{d \times D}$ (稀疏字典) | A 目标嵌入 | 长期 |
| $P$ | $D \times N$ | φ 随机投影矩阵 | 固定 (seed=42) |

**注**: 所有组件在槽位空间 $K$ 上运行，词到槽位映射由 Hash Retina 完成。

### 0.3 标量参数

| 符号 | 含义 | 默认值 |
|------|------|--------|
| $\rho_k$ | EMA 保留率 (第 $k$ 尺度) | [0.3, 0.6, 0.85, 0.95] |
| $\eta$ | B 学习率 | 0.5 |
| $\lambda_B$ | B 全局衰减率 | $1 \times 10^{-4}$ |
| $\gamma$ | Sleep 衰减因子 | 0.9 |
| $\theta_Q$ | Q 活跃阈值 | 0.005 |
| $\lambda_{\text{wd}}$ | AdamW weight decay | $1 \times 10^{-4}$ |

---

## 1. Hash Retina: 词汇编码

### 1.1 定义

Hash Retina 将词 $w$ 映射到稀疏槽位集合 $\mathcal{S}(w) \subset [K]$。

**编码流程**:

1. 添加边界符: $w \to \text{^}w\text{\$}$
2. 提取 N-gram: $\{\text{ng}_1, \text{ng}_2, \ldots\}$
3. 哈希到槽位: $\text{slot}_i = \text{hash}(\text{ng}_i) \mod K$
4. 统计频次: $c_j = |\{i : \text{slot}_i = j\}|$

**输出**: 字典 $\{j : c_j\}$，表示槽位 $j$ 的激活强度。

### 1.2 数学性质

**Jaccard 相似度** (形态相似词):

$$
J(w_1, w_2) = \frac{|\mathcal{S}(w_1) \cap \mathcal{S}(w_2)|}{|\mathcal{S}(w_1) \cup \mathcal{S}(w_2)|}
$$

**示例**:
- $J(\text{apple}, \text{apples}) \approx 0.67$
- $J(\text{cat}, \text{dog}) \approx 0$ (K=65536 时)

**泛化**: 学习 "cat" 的知识可部分迁移到 "cats"。

---

## 2. Q State: EMA 上下文

### 2.1 定义

Q 维护 $N$ 个时间尺度的指数移动平均，编码上下文的时间结构。

**更新规则**:

$$
Q_{i,k}(t+1) = \rho_k \cdot Q_{i,k}(t) + (1-\rho_k) \cdot x_i(t)
$$

其中:
- $x_i(t)$: 时刻 $t$ 槽位 $i$ 的激活 (来自 Hash Retina)
- $\rho_k$: 第 $k$ 尺度的保留率

**初始化**: $Q_{i,k}(0) = 0$

### 2.2 时间尺度

| 尺度 $k$ | $\rho_k$ | 半衰期 $\tau_{1/2}$ | 角色 |
|----------|----------|---------------------|------|
| 0 | 0.30 | 0.6 步 | 近端精细区分 |
| 1 | 0.60 | 1.4 步 | 短程过渡 |
| 2 | 0.85 | 4.3 步 | 中程区分 |
| 3 | 0.95 | 13.5 步 | 远端粗略区分 |

**半衰期**: $\tau_{1/2} = \frac{\ln 0.5}{\ln \rho_k}$

### 2.3 活跃槽位

**定义**: 槽位 $i$ 在时刻 $t$ 活跃当且仅当

$$
\max_k Q_{i,k}(t) > \theta_Q
$$

**输出**: 活跃槽位集合 $\mathcal{A}(t)$ 及其 Q 值矩阵 $Q_{\mathcal{A}} \in \mathbb{R}^{|\mathcal{A}| \times N}$

---

## 3. φ Encoder: 位置编码

### 3.1 Random 模式 (默认)

**公式**:

$$
\varphi_i = \cos(P \cdot (\alpha_{\text{scale}} \odot Q_i))
$$

其中:
- $P \in \mathbb{R}^{D \times N}$: 随机高斯矩阵, $P_{m,k} \sim \mathcal{N}(0, \sigma^2)$, seed=42
- $\alpha_{\text{scale}} \in \mathbb{R}^N$: 反向缩放权重

$$
\alpha_{\text{scale},k} = \frac{1-\rho_k}{\text{mean}(1-\rho)} \cdot \alpha_{\text{base}}
$$

$$
\alpha_{\text{base}} = \frac{\pi}{2\sigma\sqrt{D}} \cdot 63.5
$$

**直觉**: 快尺度 (小 $\rho$) 获得高频，慢尺度 (大 $\rho$) 获得低频。

### 3.2 Cosine Chaos 模式

**递归映射**:

$$
\begin{aligned}
E_i &= P \cdot (\alpha_{\text{scale}} \odot Q_i) \quad &\in \mathbb{R}^D \\
M_0 &= 0 \quad &\in \mathbb{R}^D \\
M_t &= \cos(\alpha \cdot (W_{\text{shift}} \cdot M_{t-1} + E_i)) \quad &t=1,\ldots,L \\
\varphi_i &= M_L
\end{aligned}
$$

其中:
- $W_{\text{shift}}$: 循环位移矩阵 (正交)
- $\alpha$: 混沌参数 (推荐 $\alpha=2$, Lyapunov 临界点)
- $L$: 折叠次数 (推荐 $L=1$ 或 $L=3$)

**Lyapunov ��数**:

$$
\lambda = \log\frac{\alpha}{2}
$$

- $\alpha < 2$: 有序区 ($\lambda < 0$)
- $\alpha = 2$: 临界区 ($\lambda = 0$, 最大计算能力)
- $\alpha > 2$: 混沌区 ($\lambda > 0$)

### 3.3 归一化

**定义**:

$$
\hat{\varphi}_i = \frac{\varphi_i}{\|\varphi_i\| + \epsilon}
$$

其中 $\epsilon = 10^{-8}$ 防止除零。

---

## 4. B Memory: 短期记忆

### 4.1 数据结构

B 是一个 Hash Table:

$$
B: (i_{\text{src}}, j_{\text{tgt}}) \to \mathbb{R}^D
$$

存储源槽位 $i$ 到目标槽位 $j$ 的关联向量。

### 4.2 预测

**公式**:

$$
l_B[j] = \sum_{i \in \mathcal{A}} B[i, j] \cdot \hat{\varphi}_i
$$

其中 $\mathcal{A}$ 是活跃槽位集合。

### 4.3 更新 (Energy-based Delta Rule)

**计算强度**:

$$
\begin{aligned}
s_A[i, j] &= A[i, j] \cdot \hat{\varphi}_i \quad &\text{(A 的预测强度)} \\
s_B[i, j] &= B[i, j] \cdot \hat{\varphi}_i \quad &\text{(B 的预测强度)}
\end{aligned}
$$

**能量缺口**:

$$
g[j] = 1 - (\gamma \cdot s_A[i, j] + s_B[i, j])
$$

**更新规则**:

$$
\Delta B[i, j] = \eta \cdot g[j] \cdot \varphi_i
$$

**直觉**:
- $g > 0$: A 和 B 都没学会 → 增强连接
- $g \approx 0$: 已经学会 → 不更新
- $g < 0$: 过度预测 → 减弱连接

### 4.4 全局衰减

**Lazy Decay**:

$$
B[i, j](t) = B[i, j](t_{\text{last}}) \cdot (1-\lambda_B)^{t - t_{\text{last}}}
$$

只在访问时应用衰减，避免遍历所有条目。

### 4.5 负样本抑制

对于非目标槽位 $j' \notin \mathcal{T}$ (目标集合):

$$
\Delta B[i, j'] = -\eta \cdot \lambda_{\text{neg}} \cdot s_B[i, j'] \cdot \varphi_i
$$

其中 $\lambda_{\text{neg}} = 0.1$ (默认)。

---

## 5. A Knowledge: 长期知识

### 5.1 Slice-wise 矩阵分解 (v2)

**定义**:

$$
A[i, j] = \sum_{r=1}^d E_{\text{tgt}}[j, r, :] \odot E_{\text{src}}[i, r, :] \quad \in \mathbb{R}^D
$$

其中:
- $E_{\text{src}}[i] \in \mathbb{R}^{d \times D}$: 源槽位 $i$ 的嵌入矩阵
- $E_{\text{tgt}}[j] \in \mathbb{R}^{d \times D}$: 目标槽位 $j$ 的嵌入矩阵
- $\odot$: 逐元素乘积

**参数量**: $2 \times K_{\text{active}} \times d \times D$ (稀疏存储)

### 5.2 预测

**上下文累积**:

$$
C_A = \sum_{i \in \mathcal{A}} E_{\text{src}}[i] \odot \varphi_i \quad \in \mathbb{R}^{d \times D}
$$

**预测分数**:

$$
l_A[j] = \sum_{r=1}^d \sum_{m=1}^D E_{\text{tgt}}[j, r, m] \cdot C_A[r, m]
$$

等价于 Frobenius 内积:

$$
l_A[j] = \langle E_{\text{tgt}}[j], C_A \rangle_F
$$

### 5.3 Sleep 巩固

**目标设置**:

对于 B 中的每个条目 $(i, j, b_{ij})$:

$$
\text{target}_{ij} = \gamma \cdot A_{\text{old}}[i, j] + b_{ij}
$$

**损失函数**:

$$
\mathcal{L} = \mathcal{L}_{\text{new}} + \lambda_{\text{rehearsal}} \cdot \mathcal{L}_{\text{rehearsal}}
$$

**新样本损失**:

$$
\mathcal{L}_{\text{new}} = \frac{1}{N_{\text{new}}} \sum_{(i,j) \in \mathcal{B}} \| A[i, j] - \text{target}_{ij} \|^2
$$

**排练损失** (防遗忘):

$$
\mathcal{L}_{\text{rehearsal}} = \frac{1}{N_{\text{rehearsal}}} \sum_{(i,j) \in \mathcal{R}} \| A_{\text{new}}[i, j] - A_{\text{old}}[i, j] \|^2
$$

**优化器**: AdamW

$$
\theta_{t+1} = \theta_t - \alpha \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} - \lambda_{\text{wd}} \cdot \theta_t
$$

其中 $\theta = \{E_{\text{src}}, E_{\text{tgt}}\}$。

---

## 6. 预测与解码

### 6.1 槽位级预测

**合并分数**:

$$
l[j] = l_A[j] + l_B[j]
$$

**Softmax**:

$$
p[j] = \frac{\exp(l[j] / \tau)}{\sum_{j'} \exp(l[j'] / \tau)}
$$

其中 $\tau$ 是温度参数 (默认 1.0)。

### 6.2 词级解码

**槽位到词映射**: 对于词 $w$，其槽位集合为 $\mathcal{S}(w)$。

**词级分数**:

$$
\text{Score}_w = \frac{1}{|\mathcal{S}(w)|} \sum_{j \in \mathcal{S}(w)} l[j]
$$

**Top-K 预测**: 返回分数最高的 $K$ 个词。

---

## 7. Wake-Sleep 训练

### 7.1 Wake 阶段 (在线学习)

**输入**: 词序列 $w_1, w_2, \ldots, w_T$

**流程**:

```
for t = 1 to T-1:
    # 1. 编码
    slots_src = HashRetina(w_1, ..., w_t)
    Q.update(slots_src)
    A = Q.get_active()
    φ = PhiEncoder.encode(Q[A])

    # 2. 目标
    slots_tgt = HashRetina(w_{t+1})

    # 3. B 更新
    for i in A:
        for j in slots_tgt:
            s_A = A[i,j] · φ_i
            s_B = B[i,j] · φ_i
            g = 1 - (γ·s_A + s_B)
            B[i,j] += η · g · φ_i
```

### 7.2 Sleep 阶段 (离线巩固)

**输入**: B Memory 条目集合 $\mathcal{B}$

**流程**:

```
# 1. 构建训练样本
samples = []
for (i, j, b_ij) in B:
    a_old = A[i, j]
    target = γ · a_old + b_ij
    samples.append((i, j, target))

# 2. 优化 A
for epoch in range(max_epochs):
    # 新样本
    loss_new = Σ ||A[i,j] - target_ij||²

    # 排练
    loss_rehearsal = Σ ||A_new[i,j] - A_old[i,j]||²

    # 总损失
    loss = loss_new + λ_rehearsal · loss_rehearsal

    # AdamW 更新
    optimizer.step()

# 3. B 衰减
B.decay(b_retention)  # 默认 b_retention=0 (清空)
```

---

## 8. 理论性质

### 8.1 Complementary Learning Systems

**快速学习系统** (B Memory):
- 单次遍历学习
- 存储具体经验
- 容量有限

**慢速学习系统** (A Knowledge):
- 多轮迭代学习
- 提取泛化知识
- 容量大

**协同**: B 记录 A 的残差，Sleep 时迁移到 A。

### 8.2 Laplace Neural Manifold (Q State)

**时间编码**: EMA 状态 $Q$ 近似 Laplace 变换:

$$
Q_k(t) \approx \int_0^t e^{-s/\tau_k} x(t-s) ds
$$

其中 $\tau_k = -1/\ln\rho_k$ 是时间常数。

**多尺度**: 不同 $\rho_k$ 捕捉不同时间尺度的依赖。

### 8.3 Edge of Chaos (Cosine Chaos)

**Lyapunov 指数**:

$$
\lambda = \lim_{T\to\infty} \frac{1}{T} \sum_{t=1}^T \log \left| \frac{\partial M_t}{\partial M_{t-1}} \right|
$$

对于 Cosine Chaos:

$$
\lambda = \log\frac{\alpha}{2}
$$

**临界点**: $\alpha^* = 2$ 使 $\lambda = 0$，系统在混沌边缘，具有最大计算能力。

### 8.4 CP 分解

**张量分解**: A 可视为 3-阶张量 $\mathcal{A} \in \mathbb{R}^{K \times K \times D}$ 的 CP 分解:

$$
\mathcal{A}_{i,j,m} = \sum_{r=1}^d E_{\text{src}}[i, r, m] \cdot E_{\text{tgt}}[j, r, m]
$$

**秩**: $d$ 是 CP 秩，控制模型容量。

---

## 9. 复杂度分析

### 9.1 时间复杂度

**前向预测**:
- Hash Retina: $O(|w| \cdot n_{\text{gram}})$
- Q Update: $O(N \cdot |\mathcal{S}(w)|)$
- φ Encode: $O(N \cdot D \cdot |\mathcal{A}|)$
- B Predict: $O(|\mathcal{A}| \cdot V_{\text{active}} \cdot D)$
- A Predict: $O(|\mathcal{A}| \cdot V_{\text{active}} \cdot d \cdot D)$

**瓶颈**: A/B 预测，与活跃词汇量成正比。

**Wake 训练**:
- B Update: $O(|\mathcal{A}| \cdot |\mathcal{T}| \cdot D)$

**Sleep 训练**:
- 每个 epoch: $O(|\mathcal{B}| \cdot d \cdot D)$
- 总计: $O(E \cdot |\mathcal{B}| \cdot d \cdot D)$，其中 $E$ 是 epoch 数

### 9.2 空间复杂度

**B Memory**:
- 每条目: $\sim 40$ bytes (key + D×4 bytes)
- 100K 条目: $\sim 4$ MB

**A Knowledge**:
- 参数量: $2 \times K_{\text{active}} \times d \times D \times 4$ bytes
- 稀疏存储: $\sim 1K$ slots → $\sim 4$ MB

**总计**: $\sim 10-100$ MB (典型配置)

---

## 10. 参数推荐

### 10.1 基础配置

| 参数 | 小语料 (≤20句) | 中等语料 (20-200句) | 大语料 (>200句) |
|------|----------------|---------------------|-----------------|
| D | 8 | 8 | 16 |
| d | 64 | 128 | 256 |
| φ 模式 | chaos (L=1, α=π) | chaos (L=3, α=2) | random |
| cycles | 10 | 10 | 20 |
| wake_repeats | 100 | 10 | 5 |

### 10.2 学习率

| 参数 | 推荐值 | 范围 |
|------|--------|------|
| η (B 学习率) | 0.5 | 0.3-0.8 |
| sleep_lr (A 学习率) | 0.01 | 0.005-0.05 |
| λ_B (B 衰减) | 1e-4 | 1e-5 - 1e-3 |
| λ_wd (weight decay) | 1e-4 | 1e-5 - 1e-3 |

### 10.3 Sleep 参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| γ (衰减因子) | 0.9 | 保留 90% 旧知识 |
| max_epochs | 1000 | 最大训练轮次 |
| patience | 200 | 早停 patience |
| b_retention | 0.0 | Sleep 后清空 B |

---

## 11. 与 v1 的对比

| 特性 | v1 (Hadamard) | v2 (Slice-wise) |
|------|---------------|-----------------|
| **A 结构** | $E_Q[j] \odot E_V[i]$ | $\sum_r E_{\text{tgt}}[j,r,:] \odot E_{\text{src}}[i,r,:]$ |
| **d 和 D** | 强制 d=D | 解耦 (d=64, D=8) |
| **正则化** | 正交约束 | AdamW weight decay |
| **A-only 准确率** | 5-7% | 84-86% |
| **φ 编码器** | Random | Random / Fourier / Chaos |

**改进**: v2 通过解耦 d 和 D，大幅提升 A 的表达能力。

---

## 12. 参考文献

1. **McClelland, J. L., et al. (1995)**. Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419.

2. **Howard, M. W., et al. (2014)**. A distributed representation of temporal context. *Journal of Mathematical Psychology*, 69, 269-299.

3. **Bertschinger, N., & Natschläger, T. (2004)**. Real-time computation at the edge of chaos in recurrent neural networks. *Neural Computation*, 16(7), 1413-1436.

4. **Kolda, T. G., & Bader, B. W. (2009)**. Tensor decomposations and applications. *SIAM Review*, 51(3), 455-500.

5. **Kingma, D. P., & Ba, J. (2015)**. Adam: A method for stochastic optimization. *ICLR*.

---

**版本**: v2.0
**最后更新**: 2026-03-07
**作者**: MathBrain 研究组
