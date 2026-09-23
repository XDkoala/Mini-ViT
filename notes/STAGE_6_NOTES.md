# 阶段 6：CNN 对照基线

本阶段实现一个结构清晰的 `SimpleCNN`，并让它与 MiniViT 复用同一套 CIFAR-10 数据协议、训练引擎、指标和模型保存机制，从而获得可解释的对照基线。

开始日期：2026-09-22

完成日期：2026-09-22

当前状态：已完成
当前位置：CNN 100 epochs 正式基线已完成并归档

## 阶段概览

### 阶段目标

```text
CIFAR-10 images
  → convolution blocks
  → spatial feature aggregation
  → classification head
  → 10-class logits
  → shared training engine
  → CNN baseline result
```

### 公平比较原则

CNN 与 MiniViT 复用：

- 相同的训练集、验证集和测试集划分；
- 相同的数据增强与归一化；
- 相同的 batch size 和训练 epoch 数；
- 相同的指标、checkpoint 与最佳模型选择规则；
- 第一版统一使用 AdamW，减少额外实验变量。

模型架构和参数量可以不同，但必须分别统计并在结论中说明，不能把“参数量不同的比较”误称为严格单变量实验。

### 阶段边界

本阶段主要修改：

- `src/models/cnn.py`：实现 `SimpleCNN`；
- `configs/cnn.yaml`：定义 CNN 实验配置；
- `scripts/train.py`：按配置选择 MiniViT 或 CNN；
- `tests/test_cnn.py`：由 AI 维护 CNN 的结构、shape 与梯度测试；
- `notes/STAGE_6_NOTES.md`：记录原理、实现、验收和实验结果。

本阶段不修改共享数据划分和指标定义，也不开始阶段 7 的消融实验。

## 进度总览

- [x] 明确阶段目标、边界与公平比较原则
- [x] 快速复习卷积的局部连接、权重共享与归纳偏置
- [x] 设计 `SimpleCNN` 的结构与 shape 流程
- [x] 实现并注册卷积模块
- [x] 实现前向传播与分类头
- [x] 补充并通过 CNN 自动化测试
- [x] 让训练入口按配置创建不同模型
- [x] 配置并通过 CNN smoke test
- [x] 通过 CNN tiny overfit
- [x] 完成 CNN 正式基线训练
- [x] 汇总 MiniViT 与 CNN 的参数量及验证结果
- [x] 阶段 6 总验收

## 1. CNN 如何理解一张图片

### 1.1 图片、通道与特征图

CIFAR-10 图片的 shape 是 `[B, 3, 32, 32]`：

- `B`：一个 batch 中的图片数量；
- `3`：RGB 三个输入通道；
- `32, 32`：图片的高度和宽度。

进入网络后，“通道”不再等同于颜色。卷积层产生的每个输出通道都是一张 **feature map（特征图）**，表示某种可学习模式在各位置的响应强弱。浅层可能响应于边缘、颜色对比或纹理；深层特征通常更抽象，但不能武断地把某个通道直接命名为“猫耳朵通道”。

### 1.2 一个卷积核实际计算什么

对于：

```python
nn.Conv2d(
    in_channels=C_in,
    out_channels=C_out,
    kernel_size=K,
)
```

权重 shape 为：

```text
[C_out, C_in, K, K]
```

一个输出通道不是只读取一个输入通道。它拥有 `C_in` 张 `K×K` 权重切片，在同一个空间位置分别读取所有输入通道，再把结果相加，得到一个输出值：

```text
output[b, out_c, i, j]
  = Σ input[b, in_c, i+u, j+v]
      × weight[out_c, in_c, u, v]
```

求和遍历 `in_c、u、v`。一共有 `C_out` 组这样的卷积核，因此最终产生 `C_out` 张特征图。

严格地说，PyTorch 的 `Conv2d` 实现的是互相关：计算前不会把卷积核翻转。深度学习中权重本来就由训练学习，因此通常沿用“卷积”这个名称，不影响模型表达能力。

### 1.3 输出空间尺寸公式

单个空间维度的输出大小是：

```text
output_size
  = floor((input_size + 2P - D(K - 1) - 1) / S + 1)
```

- `K`：kernel size；
- `S`：stride；
- `P`：padding；
- `D`：dilation。

本项目卷积使用 `K=3, S=1, P=1, D=1`，所以卷积前后高度和宽度不变。随后的 `2×2, stride=2` 最大池化才把高度和宽度各减半。

### 1.4 局部连接与权重共享

卷积包含两个关键假设：

- **局部连接**：一个 `3×3` 卷积核一次只观察相邻区域，不直接连接整张图片；
- **权重共享**：同一组卷积核参数扫描所有空间位置，而不是每个位置各学一组参数。

因此，一个边缘检测模式在图片左上角和右下角出现时，都能触发相同卷积核。与把图片完全展平后连接全连接层相比，这大幅减少了参数，并把“相同视觉模式可能出现在不同位置”写进了网络结构。

### 1.5 为什么只看局部也能识别整张图片

关键是 **感受野（receptive field）**。某层一个特征值的感受野，是它能间接依赖的原图区域。

第一层 `3×3` 卷积只读取局部 `3×3`；但第二层读取的是第一层相邻特征，而每个第一层特征已经包含一片原图信息。卷积和池化不断叠加后，感受野持续扩大。

本项目经过四组 `3×3 Conv + 2×2 Pool` 后，最后特征的理论感受野约为 `46×46`，已经覆盖整张 `32×32` CIFAR-10 图片。所以 CNN 是“每一步局部处理，经过层级组合获得全局信息”，并不是永远只能看见一个小角落。

## 2. 一个卷积块为什么这样排列

本项目的 `ConvBlock` 为：

```text
Conv2d → BatchNorm2d → ReLU → MaxPool2d
```

### 2.1 `Conv2d`：提取局部模式

卷积负责产生新的特征。这里设置 `bias=False`，因为紧随其后的 BatchNorm 已有可学习平移参数；保留卷积 bias 通常只会增加冗余参数。

### 2.2 `BatchNorm2d`：稳定特征尺度

BatchNorm 对每个通道分别标准化。在训练模式下，它会根据当前 batch 中该通道跨越 `B、H、W` 的元素计算均值和方差，然后进行：

```text
normalized = (x - mean) / sqrt(variance + epsilon)
output = gamma × normalized + beta
```

`gamma` 和 `beta` 是可学习参数，因此 BatchNorm 不是把特征永远强制成固定分布；模型仍能重新调整每个通道的尺度和偏移。

训练时还会更新 `running_mean` 和 `running_var`。调用 `model.eval()` 后，验证与推理改用这些运行统计量，而不再依赖当前 batch。这也是训练、验证时必须正确切换 `model.train()` 和 `model.eval()` 的原因之一。

### 2.3 `ReLU`：加入非线性

```text
ReLU(x) = max(0, x)
```

若没有激活函数，多层线性卷积的复合仍然只是线性变换，堆叠再多层也难以表达复杂分类边界。ReLU 保留正响应并把负响应截断，从而引入非线性。

代码中的 `inplace=True` 表示尽量复用原张量存储，节省少量内存；它不改变 ReLU 的数学含义。

### 2.4 `MaxPool2d`：下采样

`2×2` 最大池化在每个局部窗口保留最大响应，使 `H、W` 各缩小一半。其作用包括：

- 降低后续计算量；
- 让后续特征对应更大的原图区域；
- 对特征的小幅位置变化提供一定稳定性。

代价是丢失精细空间信息。因此下采样不能无限进行，也不是越早越多越好。

## 3. `SimpleCNN` 的完整数据流

### 3.1 Shape 变化

```text
[B, 3, 32, 32]
  → ConvBlock(3, 96)     → [B, 96, 16, 16]
  → ConvBlock(96, 192)   → [B, 192, 8, 8]
  → ConvBlock(192, 384)  → [B, 384, 4, 4]
  → ConvBlock(384, 512)  → [B, 512, 2, 2]
  → AdaptiveAvgPool      → [B, 512, 1, 1]
  → Flatten              → [B, 512]
  → Dropout + Linear     → [B, 10]
```

整体趋势是：

```text
空间尺寸逐渐减小，通道数逐渐增加
精确位置信息逐渐压缩，特征种类和抽象程度逐渐增加
```

通道变多不代表凭空创造信息，而是允许网络用更多不同的特征维度重新组织已经提取的信息。

### 3.2 `ModuleList`：创建并注册多个卷积块

模型根据：

```text
3 → 96 → 192 → 384 → 512
```

动态创建四个 `ConvBlock`，并存入 `nn.ModuleList`。普通 Python `list` 可以保存模块对象，却不会自动把它们注册成当前模型的子模块；`ModuleList` 会让 PyTorch 知道这些模块属于模型，因此其中的参数能够：

- 出现在 `model.parameters()` 中并交给 optimizer；
- 随 `model.to(device)` 移动到 CPU 或 GPU；
- 出现在 `state_dict()` 中并保存进 checkpoint；
- 正确响应 `model.train()` 与 `model.eval()`。

这与之前学过的 `self.xxx = nn.Module(...)` 属于同一个注册机制，只是 `ModuleList` 适合数量动态变化的模块序列。

### 3.3 `AdaptiveAvgPool2d((1, 1))`：全局空间汇聚

它把每个通道的整张特征图平均成一个数：

```text
[B, 512, H, W] → [B, 512, 1, 1]
```

普通固定大小池化需要提前知道输入空间尺寸；Adaptive Pooling 会根据当前输入自动选择汇聚范围，所以分类头不需要把 `32×32` 写死。每个输出通道最终用一个数表示“这种高级特征在整张图片上的总体响应”。

### 3.4 `Flatten、Dropout、Linear`

- `torch.flatten(start_dim=1)`：只展平特征维，保留 batch 维，得到 `[B, 512]`；
- `Dropout`：训练时随机将部分特征置零，减少特征之间的过度依赖；验证时自动关闭；
- `Linear(512, 10)`：把每张图片的 512 维表示映射成 10 个类别 logits。

logits 是未归一化分数，不必在模型末尾手动调用 Softmax。`nn.CrossEntropyLoss` 内部已经以数值更稳定的方式组合了 LogSoftmax 和负对数似然；提前 Softmax 既重复，也可能降低数值稳定性。

## 4. CNN 的归纳偏置

**归纳偏置**是模型在看到训练数据之前，就由结构预先加入的假设。CNN 的主要归纳偏置是：

- 相邻像素通常比远距离像素关系更紧密；
- 相同视觉模式可能出现在图片的不同位置；
- 复杂视觉概念可以由局部模式逐层组合而成。

这些假设与自然图像非常匹配，因此 CNN 在 CIFAR-10 这类小数据集上通常容易训练、数据效率较高。但归纳偏置不是绝对真理：如果任务特别依赖任意远距离位置之间的关系，CNN 需要经过多层传播才能建立联系。

## 5. CNN 与 ViT 的核心区别

| 对比项 | CNN | ViT |
|---|---|---|
| 基本单位 | 像素局部窗口 | Patch token |
| 信息交互 | 先局部，随层数扩大感受野 | 每层 Self-Attention 可直接全局交互 |
| 参数共享方式 | 卷积核跨空间位置共享 | Q/K/V 投影跨 token 共享 |
| 位置处理 | 网格结构隐含在卷积与邻域中 | 通常显式加入位置编码 |
| 主要图像先验 | 局部性、平移共享、层级结构 | 图像先验较弱，关系学习更自由 |
| 小数据表现倾向 | 往往更容易训练 | 往往更依赖数据量和正则化 |
| 全局关系 | 需要多层逐步传播 | Attention 可在一层内直接连接任意 token |

不能简单总结为“CNN 只能看局部，ViT 才能看全局”。准确说法是：CNN 通过堆叠局部算子逐步获得全局感受野；ViT 的 Self-Attention 从一开始就允许任意 token 直接交互。

## 6. 项目实现与公平比较

### 6.1 模型配置与共享训练入口

`configs/minivit.yaml` 和 `configs/cnn.yaml` 都包含 `model.name`：

```text
minivit → 创建 MiniViT
cnn     → 创建 SimpleCNN
```

训练入口复制模型配置后取出 `name`，再把剩余参数传给对应构造函数。使用副本是为了避免修改即将写入 `config.yaml` 的实验配置快照。

模型创建完成后，CNN 与 MiniViT 共用 `engine.py`。训练引擎只要求模型遵守统一契约：输入图片 `[B, 3, 32, 32]`，输出 logits `[B, 10]`；它不需要知道内部使用卷积还是 Attention。

### 6.2 公平比较的含义

本项目保持相同的数据划分、数据增强、batch size、100 epochs、AdamW、指标计算、checkpoint 和最佳模型选择规则，从而尽量减少模型架构之外的差异。

但这仍不是严格单变量消融：CNN 与 MiniViT 的运算结构不同，而且同一套优化器超参数不一定分别是两种模型的最优设置。正确表述是“在统一训练协议下比较相近参数量的 CNN 与 MiniViT 基线”。

### 6.3 参数量拆解

卷积未使用 bias，所以每层卷积参数量是：

```text
C_out × C_in × 3 × 3
```

| 部分 | 可训练参数量 |
|---|---:|
| Conv 3 → 96 + BatchNorm | 2,784 |
| Conv 96 → 192 + BatchNorm | 166,272 |
| Conv 192 → 384 + BatchNorm | 664,320 |
| Conv 384 → 512 + BatchNorm | 1,770,496 |
| Linear 512 → 10 | 5,130 |
| **SimpleCNN 总计** | **2,609,002** |

```text
SimpleCNN：2,609,002 parameters
MiniViT：  2,693,578 parameters
比例：     96.86%
```

两者参数量接近但不完全相等，最终报告应写“相近参数量基线”，不能写成严格等参数比较。

## 7. 自动化验收

2026-09-22 验收结果：

```text
CNN、配置与训练入口定向测试：33 passed
CNN tiny-overfit：             passed
项目全套自动化测试：         157 passed
```

测试覆盖：

- 卷积块输出 shape 与有限数值；
- 完整 CNN 的 `[B, 10]` 输出契约；
- Adaptive Pooling 对不同空间尺寸的支持；
- 所有卷积块和分类头的梯度链；
- 非法模型配置和非法输入 shape；
- 默认参数量级；
- CNN YAML 配置读取；
- 训练入口选择 CNN/MiniViT；
- 未知模型名的主动拒绝。

正式训练前还使用 `configs/cnn.yaml` 在真实 CIFAR-10 上完成 train/val 各 2 batches 的共享入口 smoke test：训练 loss 2.3303、训练准确率 13.28%、验证 loss 2.2726、验证准确率 16.80%，并正确生成独立 checkpoint 与日志。固定 64 张训练图片的 tiny-overfit 测试也达到至少 98% 准确率且最终 loss 低于 0.1，证明正式 CNN 架构具备足够容量且梯度链有效。

## 8. CNN 正式基线结果

CNN 使用 `configs/cnn.yaml`、随机种子 42 和与 MiniViT 相同的训练协议完成 100 epochs。测试集没有参与训练、模型选择或本阶段统计。

```text
最佳验证准确率：89.58%（epoch 74）
最佳轮训练准确率：97.38%
最佳轮验证 loss：0.4123
最低验证 loss：0.3956（epoch 39，val accuracy 88.16%）
最终训练准确率：98.00%
最终验证准确率：87.90%
最终验证 loss：0.4988
总计算耗时：1,074.90 s（约 17 min 55 s）
平均每轮耗时：10.75 s
```

最低验证 loss 出现在 epoch 39，之后训练准确率继续上升，验证 loss 缓慢增加，说明 CNN 后期同样出现过拟合；但验证准确率仍继续提升，并在 epoch 74 达到峰值。`best.pt` 正确保留 epoch 74，`last.pt` 保存 epoch 100。

### 8.1 与 MiniViT 的统一协议比较

| 指标 | MiniViT | SimpleCNN | 差异 |
|---|---:|---:|---:|
| 参数量 | 2,693,578 | 2,609,002 | CNN 少 84,576（约 3.14%） |
| 最佳验证准确率 | 81.84% | **89.58%** | CNN 高 **7.74 个百分点** |
| 最佳 epoch | 80 | 74 | CNN 早 6 轮 |
| 最终训练准确率 | 95.20% | 98.00% | CNN 高 2.80 个百分点 |
| 最终验证准确率 | 81.36% | 87.90% | CNN 高 6.54 个百分点 |
| 平均每轮耗时 | 25.64 s | **10.75 s** | CNN 约快 **2.38 倍** |
| 总计算耗时 | 42 min 44 s | **17 min 55 s** | CNN 少约 24 min 49 s |

在参数量近似、数据和训练协议相同的条件下，CNN 同时取得更高验证准确率与更低计算成本。当前证据支持：CIFAR-10 数据较小且图像具有强局部结构时，CNN 的局部连接、权重共享和层级特征归纳偏置比首版 MiniViT 更具数据效率。该结论只适用于本项目配置，不能外推为“CNN 在所有视觉任务上都优于 ViT”。

归档目录：`outputs/cnn_baseline/`

```text
best.pt     SHA-256 C20802B84993AE29C0B789AB600D2FCDC33FAD7A34C61F269273EF25D0A8E95A
last.pt     SHA-256 450E627F66650A6E24B899B7FC28257F80C6E240C8A2500DA7F3074F8C2F53D7
history.csv SHA-256 5382F7FED6687A958AF7CE1C01C3C800444D83CB44D45099DDCB1A84975BACC2
config.yaml SHA-256 47738D3E16155C945D9203CEA277850FDCB001ECEA7DF2703790F071B95ACFD6
```

阶段 6 验收结论：模型实现、共享训练入口、smoke、tiny-overfit、正式 100 epochs 基线和验证集对照分析全部完成。测试集统一留到阶段 8 加载各自 `best.pt` 后评估。

## 9. 复习自测

读完后应能独立回答：

1. 为什么卷积核的权重 shape 是 `[C_out, C_in, K, K]`？
2. `padding=1` 为什么能让 `3×3, stride=1` 卷积保持空间尺寸？
3. 为什么 CNN 每层只看局部，深层特征仍能包含整张图片的信息？
4. BatchNorm 在 `train()` 和 `eval()` 模式分别使用什么统计量？
5. 为什么卷积后接 BatchNorm 时通常可以设置 `bias=False`？
6. `ModuleList` 与普通 Python `list` 在 PyTorch 参数管理上有什么区别？
7. Adaptive Average Pooling 解决了什么问题？
8. 为什么分类模型使用 CrossEntropyLoss 时不应先手动 Softmax？
9. CNN 和 ViT 分别怎样建立远距离信息交互？
10. 为什么本项目的 CNN/MiniViT 对比是“统一协议的基线比较”，而不是严格单变量消融？

### 自测答案速查

1. 每个输出通道都要分别读取全部输入通道的 `K×K` 邻域，因此需要 `C_out` 组、每组 `C_in×K×K` 权重。
2. 每侧补一个像素，总有效尺寸增加 2，恰好抵消 `3×3` 卷积造成的尺寸减少 2。
3. 后一层读取前一层的局部特征，多层叠加和下采样会不断扩大特征相对于原图的感受野。
4. `train()` 使用当前 batch 的统计量并更新运行统计量；`eval()` 使用已累计的 running mean/variance。
5. BatchNorm 后续的可学习 `beta` 已能完成平移，卷积 bias 通常会被标准化抵消，因而冗余。
6. `ModuleList` 会注册子模块，普通 list 不会；注册决定参数优化、设备移动、模式切换和 checkpoint 保存。
7. 它把不同空间尺寸统一汇聚成指定输出尺寸，使分类头不依赖写死的输入分辨率。
8. CrossEntropyLoss 已稳定地完成 LogSoftmax 与负对数似然；提前 Softmax 会重复运算并降低数值稳定性。
9. CNN 通过多层局部操作逐步扩大感受野；ViT 的 Self-Attention 可让任意 token 在一层内直接交互。
10. 虽然训练协议与参数量接近，但架构不同，且相同超参数未必分别最优，因此它是对照基线而非只改变一个因素的消融实验。
