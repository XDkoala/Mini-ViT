# 阶段 2：逐个实现 MiniViT 零件

本阶段逐个实现并验证 Patch Embedding、Multi-Head Self-Attention、MLP 和 Transformer Encoder Block。这里只建立可靠的模型零件，不组装完整 MiniViT，也不进入训练循环。

开始日期：2026-09-10  
完成日期：2026-09-17  
当前状态：已完成  
当前位置：阶段 2 的四个模型零件均已实现并通过验收

## 阶段概览

### 阶段目标

每个模型零件都形成独立、可检查的输入输出闭环：

```text
理解职责与原理
  → 推导 shape
  → 实现 forward
  → 定义对外契约
  → 编写自动化测试
  → 验收数值和梯度
```

### 阶段边界

本阶段修改：

- `src/models/minivit.py`：实现四个模型零件。
- `tests/test_model.py`：每完成一个零件就补充对应测试。
- `notes/STAGE_2_NOTES.md`：记录原理、机制、契约与验收结论。

本阶段不做：

- 完整 `MiniViT` 组装。
- CLS token、位置编码和分类头。
- loss、优化器、训练循环和准确率评估。

这些内容留到阶段 3 及之后，避免模型零件错误与训练流程错误混在一起。

## 进度总览

- [x] 公共基础：`nn.Module`、注册机制与梯度更新
- [x] Patch Embedding：原理与实现
- [x] Patch Embedding：自动化契约测试
- [x] Self-Attention：Q/K/V、缩放点积、softmax 与 Value 聚合
- [x] Multi-Head Self-Attention：分头、合并与输出投影
- [x] Multi-Head Self-Attention：实现与测试
- [x] MLP：原理、实现与测试
- [x] Transformer Encoder Block：原理、实现与测试
- [x] `tests/test_model.py` 中阶段 2 的全部测试通过
- [x] 阶段 2 总验收

## 符号与 shape 速查

### 固定符号

| 符号 | 含义 | 本项目默认值 |
|---|---|---:|
| `B` | batch size | 测试常用 2 |
| `C` | 图片通道数 | 3 |
| `H, W` | 图片高、宽 | 32, 32 |
| `P` | patch 边长 | 4 |
| `N` | token 数 | Patch 后 64；加入 CLS 后 65 |
| `D` | token embedding 维度 | 192 |
| `A` | attention head 数 | 6 |
| `Dh` | 每个 head 的维度 | `D/A = 32` |
| `M` | MLP 隐藏维度 | 768 |

类通过构造参数表达尺寸，不把默认数值散落硬编码在 `forward()` 中。

### 四个零件的 shape

| 零件 | 输入 | 输出 |
|---|---|---|
| Patch Embedding | `[B,C,H,W]` | `[B,N,D]` |
| Multi-Head Self-Attention | `[B,N,D]` | `[B,N,D]`；可选权重 `[B,A,N,N]` |
| MLP | `[B,N,D]` | `[B,N,D]` |
| Encoder Block | `[B,N,D]` | `[B,N,D]` |

本节只用于快速查阅；每个 shape 的来源在对应零件章节解释。

## 1. 公共基础

### 1.1 `nn.Module`、`__init__()` 与 `forward()`

`__init__()` 在创建对象时执行，负责保存配置并组装子层；`forward()` 在每次输入数据时执行，负责定义计算。正常使用 `module(x)`，由 `nn.Module.__call__()` 先处理 hooks 等框架逻辑，再调用 `forward(x)`，因此不应直接调用 `module.forward(x)`。

`super().__init__()` 必须在注册子层之前执行，用来建立 PyTorch 的模块、参数和 buffer 管理结构。

### 1.2 注册机制

执行：

```python
self.projection = nn.Conv2d(...)
```

时，右侧先创建卷积对象；随后 `nn.Module.__setattr__()` 根据对象的运行时类型，将它登记为当前模块的子模块。可以用三本登记册理解：

| 对象 | 推荐写法 | 内部登记 | 是否训练 | 随 `.to(device)` 移动 | 进入 `state_dict()` |
|---|---|---|---:|---:|---:|
| 子模块 | `self.layer = nn.Linear(...)` | `_modules` | 取决于其参数 | 是 | 是 |
| 可训练 Tensor | `self.x = nn.Parameter(...)` | `_parameters` | 是 | 是 | 是 |
| 非训练状态 Tensor | `register_buffer(...)` | `_buffers` | 否 | 是 | 默认是 |
| 普通值或普通 Tensor | 普通属性赋值 | 普通属性 | 否 | 普通 Tensor 不自动移动 | 否 |

父模块沿注册关系递归完成 `parameters()`、`state_dict()`、`.to(device)`、`train()` 和 `eval()`。多个子层应使用 `nn.ModuleList`、`nn.ModuleDict` 或 `nn.Sequential`；放在普通 Python 列表中的模块不会被递归注册。

### 1.3 梯度计算与参数更新

一次训练迭代的核心顺序：

```python
optimizer.zero_grad()
prediction = model(inputs)
loss = criterion(prediction, targets)
loss.backward()
optimizer.step()
```

`backward()` 沿动态计算图应用链式法则，把梯度累积到参与 loss 计算的叶子参数的 `.grad` 中；它本身不更新参数。`optimizer.step()` 才根据梯度执行更新，例如 SGD 的基本形式为：

```text
新参数 = 旧参数 - 学习率 × 梯度
```

PyTorch 默认累积梯度，所以普通训练迭代需要先清零。注册与梯度也不能混为一谈：注册使 `model.parameters()` 和优化器能够找到参数；参数还必须参与当前 loss 的计算，才能在 `backward()` 后得到梯度。

### 1.4 模块契约与自动化测试

契约是模块对调用者承诺的可观察行为，包括合法输入、输出 shape、数值性质、梯度通路和非法输入的异常。测试把自然语言承诺变成可执行断言，用于防止以后重构造成回归。

测试通常分为：

```text
Arrange：准备模块和输入
Act：执行被测行为
Assert：验证输出或异常
```

测试重点检查公开行为，不依赖无关的局部变量名等实现细节。随机初始化使具体输出数值不固定，因此通常检查 shape、有限性和数学性质；需要比较精确数值时，应固定输入、权重或随机种子。

## 2. Patch Embedding

### 2.1 职责

将二维图片切成互不重叠的 patch，并用共享的可训练投影把每个 patch 变成一个 token：

```text
[B,C,H,W] → [B,N,D]
```

其中：

```text
N = (H/P) × (W/P)
```

### 2.2 图片如何变成 patch

一个 `P×P` patch 含有 `C×P×P` 个数。默认配置中：

```text
单个 patch：[3,4,4] → 展平后 48 个数
patch 网格：32/4 × 32/4 = 8×8
patch 总数：N = 64
```

Patch 是原图片中的空间区域；token 是 patch 经可训练投影得到的 `D` 维特征向量，两者不能混为一谈。

### 2.3 Conv2d 为什么等价于 patch 投影

```python
nn.Conv2d(
    in_channels=C,
    out_channels=D,
    kernel_size=P,
    stride=P,
)
```

`kernel_size=P` 使卷积每次覆盖一个 patch，`stride=P` 使相邻 patch 不重叠，`out_channels=D` 使每个位置产生 `D` 个特征。

卷积权重 shape 为 `[D,C,P,P]`，展平后等价于 Linear 权重 `[D,C×P×P]`。同一组卷积核扫描所有位置，相当于所有 patch 共用同一个线性投影。默认配置与 `Linear(48,192)` 都有：

```text
权重：192×3×4×4 = 9,216
偏置：192
合计：9,408
```

### 2.4 实现机制与 shape

```text
[B,C,H,W]
  ── Conv2d(kernel=P, stride=P) ──>
[B,D,H/P,W/P]
  ── flatten(start_dim=2) ──>
[B,D,N]
  ── transpose(1,2) ──>
[B,N,D]
```

`flatten(start_dim=2)` 保留 batch 和特征轴，只将二维 patch 网格按从左到右、从上到下的顺序展平：

```text
n = row × grid_width + column
```

`transpose(1,2)` 只交换 token 轴与特征轴：

```text
tokens[b,n,d] == patch_features[b,d,n]
```

它不会混合或重新计算数值。`[B,N,D]` 把特征维放在最后，使后续 `nn.Linear` 能独立处理每个 token。

### 2.5 输入输出契约

构造参数：

- `in_channels > 0`
- `embed_dim > 0`
- `patch_size > 0`

输入：

- 必须为四维 `[B,C,H,W]`。
- `C` 必须等于构造时的 `in_channels`。
- `H,W` 必须不小于 `P`。
- `H,W` 必须能被 `P` 整除，防止卷积静默丢弃边缘。

输出：

- shape 必须为 `[B,(H/P)×(W/P),D]`。
- 对正常有限输入，输出应保持有限。
- 从输出构造 loss 并反向传播后，投影层参数应得到非空且有限的梯度。

### 2.6 自动化测试计划

- [x] 默认配置输出 `[2,64,192]` 且数值有限。
- [x] 自定义配置验证通用 shape 公式。
- [x] 非法构造参数抛出 `ValueError`。
- [x] 非四维、错误通道、尺寸过小和不可整除输入均被拒绝。
- [x] 投影层参数在反向传播后具有非空且有限的梯度。

### 2.7 当前验收结果

2026-09-14 验收通过：

- 默认输入 `[2,3,32,32]` 输出 `[2,64,192]`。
- 自定义输入 `[2,1,6,10]` 配合 `D=8,P=2` 输出 `[2,15,8]`。
- 输出与 `projection → flatten → transpose` 手工管线完全一致，且数值有限。
- 四类非法输入均被拒绝。
- 卷积权重与偏置均获得非空且有限的梯度。

自动化测试已写入 `tests/test_model.py`。2026-09-14 独立运行结果：`13 passed`，覆盖默认与自定义 shape、六组非法构造参数、四类非法图片输入，以及投影权重和偏置的有限梯度。

## 3. Multi-Head Self-Attention

### 3.1 核心职责与 Q、K、V

Self-Attention 让每个 token 根据当前输入内容，动态选择并聚合序列中所有 token 的信息。输入 patch token 起初主要描述局部区域；Attention 输出则包含与该位置相关的全局上下文。

同一个输入 token 通过三套独立的可学习投影获得三种角色：

```text
Query：当前 token 想寻找什么
Key：当前 token 用什么特征接受匹配
Value：当前 token 被选中后实际提供什么内容
```

标准 Self-Attention 中，Q、K、V 都来自同一输入序列 `X`，这就是 “Self” 的含义；并不是每个 token 只关注自己。若 Query 来自一个序列，而 Key、Value 来自另一个序列，则属于 Cross-Attention。

```text
Q = XWQ + bQ
K = XWK + bK
V = XWV + bV
```

Q 与 K 决定“选谁”，V 决定“取回什么”。每个输入 token 对应一个 Query，因此输入和输出 token 数均为 `N`。

### 3.2 为什么 Q 和 K 分开

如果令 `Q=K`，可以使用 `k_i·k_j` 衡量 token 相似性，这在数学上完全可行。但原始分数满足：

```text
k_i·k_j = k_j·k_i
```

所以 `KKᵀ` 必然对称。逐行 softmax 的分母不同，因此最终权重不一定对称，但 softmax 前的两两兼容度仍受到对称约束。

标准形式使用独立投影：

```text
score(i,j) = q_i·k_j
score(j,i) = q_j·k_i
```

两者可以不同，因此能表达“i 很需要 j，但 j 不同等需要 i”的有向关系。这里建模的是需求与供给的兼容度，而不只是相似度。

从矩阵上看：

```text
Q、K 独立：S = X WQ WKᵀ Xᵀ = XAXᵀ，A 不必对称
Q、K 共享：S = X W Wᵀ Xᵀ，WWᵀ 必然对称且约束更强
```

`Q=K` 或 `Q=K=V` 不是错误设计，而是参数共享形成的更强归纳偏置。共享全部三者时，Attention 变为 `softmax(EEᵀ/√d)E`，相当于在学习出的特征空间中按相似度平滑信息。多头可以提供多个不同的共享相似度空间，可能部分弥补能力损失，但每个 head 的原始匹配矩阵仍受对称约束，不能完整恢复独立 Q、K 的有向兼容度。这一想法可在标准模型完成后作为消融实验比较参数量与准确率。

### 3.3 单头 Scaled Dot-Product Attention

单头核心公式：

```text
Attention(Q,K,V) = softmax(QKᵀ / √d_k)V
```

带 batch 的 shape 路线：

```text
Q                     [B,N,d_k]
K                     [B,N,d_k]
K.transpose(-2,-1)    [B,d_k,N]
Q @ Kᵀ                [B,N,N]
/ √d_k                [B,N,N]
softmax(dim=-1)       [B,N,N]
attention @ V         [B,N,d_v]
```

`@` 对最后两个维度执行矩阵乘法，把前面的维度视为批量维。因此每个样本独立计算，不同图片之间不会互相注意。

### 3.4 `QKᵀ` 与分数矩阵

`QKᵀ` 一次计算所有 `q_i·k_j`。输出 `[B,N,N]` 中：

```text
scores[b,i,j] = 第 b 个样本中，Query i 对 Key j 的原始匹配分数
```

矩阵的行对应 Query，列对应 Key。第 `i` 行描述 token `i` 如何评价所有信息来源。使用 `key.transpose(-2,-1)` 只交换最后两个维度，因此单头和后续多头都可使用同一写法。

原始分数不是概率：它可以为负，也不要求位于 `[0,1]` 或每行和为 1。

### 3.5 为什么除以 `√d_k`

点积是 `d_k` 个乘积之和。在各分量近似独立、均值为 0、方差为 1 的简化假设下：

```text
Var(q·k) ≈ d_k
Std(q·k) ≈ √d_k
```

所以除以 `√d_k` 后，送入 softmax 的分数尺度约保持稳定。若不缩放，高维点积容易变大，使 softmax 过早接近 one-hot，`p(1-p)` 一类梯度因素随之接近 0；若除以 `d_k`，分数又可能随维度增大而过小、趋向均匀分布。

缩放等价写法：

```python
scaled_scores = scores / math.sqrt(key_dimension)
scale = key_dimension ** -0.5
scaled_scores = scores * scale
```

除以同一个正数不会改变同一行分数的大小顺序，只调节 softmax 的尖锐程度。多头实现应使用每个 head 的 `√Dh`，不是完整 `√D` 或 token 数 `√N`。

### 3.6 Softmax 与权重方向

```text
weight(i,j) = exp(score(i,j)) / Σ_m exp(score(i,m))
```

softmax 在最后一维，也就是 Key 维执行：

```python
attention_weights = torch.softmax(scaled_scores, dim=-1)
```

于是每个 Query 对所有 Key 的权重均为正且和为 1。若误用 `dim=0`，归一化的将是不同 Query 对同一个 Key 的列方向，不符合标准 Attention 的语义。

Softmax 只取决于一行内的相对差异，满足 `softmax(z)=softmax(z-c)`。稳定实现会先减去该行最大值，避免 `exp()` 因过大的正数溢出；PyTorch 已在内部完成这一处理。

### 3.7 聚合 Value 与上下文化表示

```text
Z = attention_weights @ V
z_i = Σ_j weight(i,j)v_j
```

因为权重为正且每行和为 1，每个 `z_i` 是所有 Value 的加权平均（凸组合）。输出位置 `i` 仍对应 Query `i`，但内容已融合它所关注的其他 token，因此成为上下文化表示。

Q、K 虽不直接进入最终加权和，仍会获得梯度，因为权重由 `softmax(QKᵀ/√d_k)` 产生；V 则通过加权求和直接连接到输出。反向传播因此同时训练“怎样寻找、怎样被匹配、传递什么”。

### 3.8 Linear 如何生成 Q、K、V

`nn.Linear(in_features=D, out_features=M)` 只变换输入的最后一维，保留所有前导维度：

```text
[B,N,D] → [B,N,M]
Y[b,n,o] = Σ_d X[b,n,d]W[o,d] + bias[o]
```

因此，同一个 Linear 独立处理每个 token，不直接混合 token；所有 batch 和 token 共用同一套权重。真正的 token 间交互发生在 `QKᵀ` 与 `attention @ V`。

教学时可用三个独立的 `Linear(D,D)`：

```python
query = query_projection(x)
key = key_projection(x)
value = value_projection(x)
```

三个层结构相同，但各自拥有独立参数。完整的单头计算为：

```python
scores = (query @ key.transpose(-2, -1)) * scale
attention_weights = torch.softmax(scores, dim=-1)
output = attention_weights @ value
```

若 loss 由 `output` 构造，Q、K、V 三条路径都会连接到 loss；`backward()` 后相应参数应得到梯度。重复反向传播前要先清空旧梯度，因为 PyTorch 默认累积 `.grad`。

### 3.9 融合 QKV 投影

正式实现使用一个：

```python
self.qkv_projection = nn.Linear(D, 3 * D)
```

一次生成：

```text
x [B,N,D] → qkv [B,N,3D]
```

再沿最后一维平均切分：

```python
query, key, value = qkv.chunk(3, dim=-1)
```

得到三个 `[B,N,D]` 张量。融合投影只是把三次矩阵乘法合并为一次较大的矩阵乘法，减少算子调用和重复读取输入，并不让 Q、K、V 共享参数。

PyTorch 的 Linear 权重按 `[out_features,in_features]` 存储，所以 `qkv_projection.weight` 为 `[3D,D]`：

```text
前 D 行       → WQ
中间 D 行     → WK
最后 D 行     → WV
```

三块参数仍能独立学习。融合前后参数量也相同：

```text
三个 Linear(D,D)：3(D²+D)
一个 Linear(D,3D)：3D²+3D
```

### 3.10 多头的本质：多套独立的注意力分布

设注意力头数为 `A`，每头维度为 `Dh`，标准设计满足：

```text
D = A × Dh
Dh = D / A
```

本项目中：

```text
D = 192
A = 6
Dh = 32
```

必须检查 `D % A == 0`，否则不能平均拆分。

多头不是把原始 token 的 192 维直接切成六段。顺序是先进行可学习投影，再拆分投影结果。第 `h` 个头的投影可写为：

```text
WQ_h：[192,32]
q_i_h = x_i @ WQ_h
```

因此，每个头生成的 32 维表示都可以综合原始 token 的全部 192 维信息，而不是只能读取某段固定的原始特征。更准确的理解是：

```text
完整 token
  → 各头从全部 D 维中提取自己的 Dh 维摘要
  → 各头在自己的表示空间中学习关系
```

也可以让每个头都保留完整的 D 维，但此时内部宽度会变为 `A×D`，QKV、输出投影和注意力计算量都会大约扩大 A 倍。标准设计令总内部宽度保持为 D，是模型容量与计算预算之间的折中。

决定单头或多头的不是“代码中用了几个大矩阵”，而是有几套独立的注意力分布：

```text
单头：softmax(S1 + S2 + ... + SA) → 一张注意力图
多头：softmax(S1), ..., softmax(SA) → A 张注意力图
```

由于 softmax 是非线性的，先相加再做一次 softmax，不等于分别做多次 softmax。融合 QKV 投影和批量矩阵乘法只是并行计算；只要各头分别保留分数、分别 softmax、分别聚合 Value，数学上仍然是多个头。

### 3.11 拆头与并行注意力

Q、K、V 最初都是 `[B,N,D]`。拆头分两步：

```text
[B,N,D]
  → reshape
[B,N,A,Dh]
  → transpose(1,2)
[B,A,N,Dh]
```

拆的是每个 token 的特征维，不是 token 维或 batch 维；每个头仍然看到全部 N 个 token。把 head 维移到 token 维前，是为了让 `B` 和 `A` 都作为矩阵乘法的批量维：

```python
scores = query @ key.transpose(-2, -1)
```

```text
[B,A,N,Dh] @ [B,A,Dh,N] → [B,A,N,N]
```

随后：

```python
scores = scores * (Dh ** -0.5)
attention_weights = torch.softmax(scores, dim=-1)
head_outputs = attention_weights @ value
```

shape 路线：

```text
scores              [B,A,N,N]
attention_weights   [B,A,N,N]
value               [B,A,N,Dh]
head_outputs        [B,A,N,Dh]
```

`attention_weights[b,a,i,:]` 是第 b 个样本、第 a 个头、第 i 个 Query 对所有 Key 的独立概率分布，最后一维之和为 1。缩放必须使用 `√Dh`，因为每个头的点积由 Dh 项组成。

### 3.12 合并多头与输出投影

各头完成 Value 聚合后，要让同一个 token 的各头结果相邻：

```text
[B,A,N,Dh]
  → transpose(1,2)
[B,N,A,Dh]
  → reshape
[B,N,D]
```

不能跳过 transpose 直接 reshape，否则会按照错误的轴顺序组合数据。由于 transpose 后张量通常不连续，这里使用 `reshape()`；另一种等价写法是 `contiguous().view(...)`。

合并相当于沿特征维拼接：

```text
[head 0 的 Dh 维 | head 1 的 Dh 维 | ...] → D 维
```

拼接只把结果放在一起，还没有让不同头真正混合。因此还需：

```python
self.output_projection = nn.Linear(D, D)
output = self.output_projection(combined_heads)
```

输出投影的每个输出特征都能读取完整 D 维，从而学习如何组合、保留或削弱不同头提供的信息。输出仍为 `[B,N,D]`，也为后续残差连接 `x + attention(x)` 保持 shape 契约。

### 3.13 完整 shape 路线

```text
x                              [B,N,D]
QKV Linear                     [B,N,3D]
拆出 Q、K、V                   各 [B,N,D]
拆成 A 个 head                 各 [B,A,N,Dh]
Q @ Kᵀ / √Dh                   [B,A,N,N]
softmax(dim=-1)                [B,A,N,N]
attention_weights @ V          [B,A,N,Dh]
transpose + reshape            [B,N,D]
output projection              [B,N,D]
```

一句话概括完整机制：

> 每个头先从完整 token 中提取自己的低维 Q、K、V，再独立生成一张注意力图并聚合信息；所有头的结果最后拼接并通过输出投影融合。

### 3.14 实现契约与验收

正式模块需要满足：

- 构造参数 `embed_dim > 0`、`num_heads > 0`，且 `embed_dim % num_heads == 0`。
- 输入必须为 `[B,N,D]`，最后一维必须等于构造时的 `embed_dim`。
- 输入 `[2,65,192]` 时，输出为 `[2,65,192]`。
- 可选 attention 权重为 `[2,6,65,65]`。
- 每个头的权重在最后一维之和接近 1。
- 输出不含 NaN 或 Inf。
- QKV 与输出投影参数均具有非空且有限的梯度。

2026-09-15 验收通过：

- 默认输入 `[2,65,192]` 输出 `[2,65,192]`，可选权重为 `[2,6,65,65]`。
- 自定义 `D=12,A=3,N=7` 输出 `[4,7,12]`，证明默认尺寸未被写死。
- 每个头、每个 Query 的权重在 Key 维求和接近 1，且输出与权重均为有限值。
- 非法维度、不可整除的 head 配置、非法 dropout 和错误输入 shape 均被拒绝。
- QKV 与输出投影的所有参数均获得非空且有限的梯度。
- `tests/test_model.py` 当前整体结果：`26 passed`。

下一步进入 MLP 的原理、实现与测试。

## 4. MLP

### 4.1 职责：交流之后独立加工

Attention 主要沿 token 方向交换信息，让每个 token 从其他 token 获取上下文；MLP 则对每个上下文化 token 的特征独立加工：

```text
Attention：token 与 token 之间交流
MLP：每个 token 对已有信息进行非线性变换
```

`nn.Linear` 只处理最后一维，因此同一个 MLP 会并行处理所有 token，并让它们共享参数，但不会在 MLP 内混合不同 token：

```text
output[b,n] = mlp(x[b,n])
```

二者在 Encoder 中反复形成“交流 → 加工 → 再交流 → 再加工”。

### 4.2 扩维、激活与缩维

本项目默认结构：

```text
[B,N,192]
  → Linear(192,768)
  → GELU
  → Dropout
  → Linear(768,192)
  → Dropout
[B,N,192]
```

一般 shape 为：

```text
[B,N,D] → [B,N,M] → [B,N,D]
```

第一层把每个 token 展开到更宽的内部计算空间，第二层将加工结果融合回模型维度。这里 `M` 不必被 `D` 整除；默认 `M=4D` 是架构选择，不是 reshape 约束。

若两层 Linear 中间没有激活函数：

```text
y = (xW1 + b1)W2 + b2 = xW' + b'
```

两层仍可折叠成一个线性层，扩维不会带来真正的非线性表达能力。加入 GELU 后，这种合并不再成立。

### 4.3 GELU

```text
GELU(x) = x Φ(x)
```

`Φ(x)` 是标准正态分布的累积分布函数。GELU 对明显正值大部分放行，对接近零的值平滑调节，对负值明显抑制但不像 ReLU 那样全部截断为零。它逐元素计算、不改变 shape，也不混合 token 或特征；特征混合由前后的 Linear 完成。

GELU 是 Transformer 中常见的平滑非线性选择，但并不意味着它在所有任务上都必然优于其他激活函数。

### 4.4 两处 Dropout

第一处 Dropout 位于 GELU 后，正则化扩展后的隐藏特征；第二处位于第二个 Linear 后，正则化 MLP 即将提供给残差路径的输出。

训练时，PyTorch Dropout 可写为：

```text
mask ~ Bernoulli(1-p)
y = mask × x / (1-p)
```

随机置零减少模型对固定特征组合的依赖；保留元素除以 `1-p`，使输出的数学期望保持不变。每次调用通常生成新 mask，即使复用同一个 Dropout 模块也不会固定丢弃位置。本项目注册 `dropout1` 和 `dropout2` 两个模块，使位置更明确并便于以后分别配置。

`model.train()` 启用随机丢弃，`model.eval()` 自动关闭。训练时不应在每次 forward 前重设同一随机种子，否则会反复丢弃固定位置，削弱 Dropout 的意义。

### 4.5 实现契约与验收

模块契约：

- `embed_dim > 0`、`hidden_dim > 0`，`dropout` 位于 `[0,1)`。
- 输入必须为 `[B,N,D]`，最后一维与 `embed_dim` 一致。
- 输出保持 `[B,N,D]`，以便后续执行残差连接。
- 输出为有限值，`fc1` 与 `fc2` 的参数均位于梯度通路。
- 两个 Dropout 均注册为子模块，并随父模块切换训练状态。

2026-09-15 验收通过：

- 默认输入 `[2,65,192]` 输出 `[2,65,192]`。
- 自定义 `D=8,M=13,N=5` 输出 `[3,5,8]`，证明默认尺寸和整数扩展比例未被写死。
- 非法构造参数和错误 token shape 均被拒绝。
- 两个 Linear 的权重与偏置均获得非空且有限的梯度。
- 两个 Dropout 均正确跟随 `train()/eval()`。
- `tests/test_model.py` 当前整体结果：`38 passed`。

## 5. Transformer Encoder Block

### 5.1 职责与结构

使用 Pre-Norm 组合 Attention、MLP 和两条残差路径：

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

Attention 负责 token 之间的信息交互；MLP 负责单个 token 内部的非线性变换；残差连接保留原信息，并为反向传播提供直接梯度通路。

执行顺序不能交换：第二个 `norm2` 和 MLP 接收的是 Attention 残差更新后的 `x`，而不是 Block 最初的输入。

### 5.2 残差连接

残差结构：

```text
y = x + F(x)
```

让分支只需学习“在原表示上补充或修正什么”。如果当前层不需要改变输入，只需令 `F(x)≈0`，便可近似恒等映射；没有残差时，复杂分支必须自行学出 `F(x)=x`。

梯度为：

```text
∂y/∂x = I + ∂F/∂x
```

恒等项 `I` 为信息和梯度提供不经过复杂分支的直接路径，使深层网络通常更容易优化，但不保证彻底消除梯度消失或爆炸。

逐元素相加要求 `x` 和分支输出同为 `[B,N,D]`，这也是 Attention 和 MLP 都保持输入输出 shape 的原因。实现使用：

```python
x = x + branch_output
```

避免不必要的 `x += branch_output` 原地修改，以免覆盖 autograd 反向传播需要的旧值。

### 5.3 LayerNorm

`nn.LayerNorm(D)` 对 `[B,N,D]` 中每个 token 的最后 D 个特征独立计算：

```text
μ = mean(x)
σ² = mean((x-μ)²)
x_hat = (x-μ) / sqrt(σ²+ε)
y = γ × x_hat + β
```

`ε` 防止零方差导致除零；`γ` 和 `β` 是 shape 为 `[D]` 的可训练缩放、平移参数，并由所有 batch 和 token 共享。标准化中间结果接近零均值、单位方差，但经过训练后的 `γ`、`β` 后，最终输出不必继续满足这一性质。

LayerNorm 不混合不同 token，也不依赖 batch 中的其他样本。相比 BatchNorm，它：

- batch size 为 1 时仍可正常工作；
- 不关心 token 数 N 是否变化；
- 不维护训练与评估两套运行统计量；
- 正好作用于单个 token 的完整特征表示。

Encoder 使用两个独立 LayerNorm：`norm1` 为 Attention 准备输入，`norm2` 为 MLP 准备输入；二者结构相同但服务位置不同，因此不共享 `γ`、`β`。

### 5.4 Pre-Norm 与 Post-Norm

```text
Post-Norm：y = LN(x + F(x))
Pre-Norm： y = x + F(LN(x))
```

Post-Norm 的梯度形式为：

```text
∂y/∂x = J_LN (I + J_F)
```

梯度最终必须经过 LayerNorm。Pre-Norm 为：

```text
∂y/∂x = I + J_F J_LN
```

残差主路径上的恒等项不经过 LayerNorm，同时复杂分支始终接收尺度相对稳定的输入，因此 Pre-Norm 通常更适合稳定训练较深的 Transformer。

Pre-Norm 不会立即归一化残差相加后的输出，完整模型通常会在所有 Encoder Blocks 后再使用一个最终 LayerNorm。Pre-Norm 与 Post-Norm 都是有效架构，前者改善优化稳定性，但并不保证在所有任务上必然更优。

### 5.5 展开的数据流

```python
normalized = norm1(x)
attention_output = attention(normalized)
x = x + attention_output

normalized = norm2(x)
mlp_output = mlp(normalized)
x = x + mlp_output
```

项目接口还需支持可选返回 Attention 权重：

```text
return_attention=False → output [B,N,D]
return_attention=True  → (output [B,N,D], weights [B,A,N,N])
```

### 5.6 验收目标

- 输入输出均为 `[2,65,192]`。
- 可选 Attention 权重为 `[2,6,65,65]`，且 Key 维求和接近 1。
- 前向结果全部有限。
- Attention 与 MLP 分支都参与输出。
- 两个 LayerNorm 使用独立参数，并正确注册。
- 反向传播后两条分支的关键参数均具有非空且有限的梯度。

2026-09-17 验收通过：

- 默认输入 `[2,65,192]` 输出 `[2,65,192]`，可选权重为 `[2,6,65,65]`。
- 自定义 `D=8,A=2,M=13,N=5` 输出 `[3,5,8]`。
- 模块输出与手工展开的两条串行 Pre-Norm 残差路径完全一致，确认 MLP 读取 Attention 更新后的表示。
- `norm1` 和 `norm2` 的模块、权重与偏置均为独立对象。
- Attention 权重满足概率性质，所有输出均为有限值。
- LayerNorm、Attention 和 MLP 的全部参数均获得非空且有限的梯度。
- 非法构造参数和错误 token shape 均被拒绝。
- `tests/test_model.py`：`53 passed`；项目全部测试：`64 passed`。

## 6. 阶段总验收

- [x] 四个零件的输入输出 shape 均符合契约。
- [x] 所有规定的错误输入都给出清晰异常。
- [x] Attention 权重满足概率性质。
- [x] 四个零件的前向结果均不含 NaN 或 Inf。
- [x] Attention、MLP 和 Encoder Block 的关键参数均有有限梯度。
- [x] `tests/test_model.py` 中阶段 2 的全部测试通过。

阶段 2 已完成。上述结果说明模型零件在结构、数值和梯度上可运行；完整 MiniViT 将在阶段 3 组装和验收。
