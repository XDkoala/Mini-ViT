# Mini Vision Transformer 图像分类项目规划

> 项目定位：从零实现一个适用于小尺寸图像的 Vision Transformer，在 CIFAR-10 上完成可复现训练、对照实验、消融分析与注意力可视化，并进一步验证模型在“具身视角图像”上的迁移能力。

## 1. 为什么做这个项目

这个项目不应只停留在“调用现成 ViT 完成分类”。最终需要同时证明三件事：

1. **理解 Transformer**：能独立实现 Patch Embedding、位置编码、多头自注意力、MLP、残差连接、LayerNorm 和分类头，并解释张量形状与计算过程。
2. **具备完整实验能力**：能管理配置、训练与验证模型、保存日志和权重、复现实验，并用消融实验支持结论。
3. **能连接具身智能场景**：理解图像分类不是具身智能本身，但视觉表征是机器人感知的重要基础；能够讨论第一视角、域偏移、数据效率和实时性。

### 最终项目叙事

推荐在简历和面试中这样概括项目：

> 基于 PyTorch 从零实现轻量级 Vision Transformer，在 CIFAR-10 上构建端到端训练与评估管线；通过 CNN 基线、Patch Size、网络深度和数据增强等对照实验分析 ViT 的数据效率与计算开销，并使用 Attention Rollout 展示模型关注区域；进一步在具身视角图像子集上验证迁移能力。

## 2. 项目范围

### 必做内容（MVP）

- 使用 PyTorch，不依赖现成 ViT 模型，手写 MiniViT 的核心模块。
- 使用 CIFAR-10，固定训练集、验证集、测试集划分。
- 实现训练、验证、测试、断点续训、指标记录和配置管理。
- 实现一个参数量相近的 CNN 或 ResNet-18 基线。
- 至少完成 3 组消融实验。
- 输出训练曲线、混淆矩阵、错误案例和注意力可视化。
- 提供清晰 README、一键运行命令和最终实验表。

### 推荐进阶内容

- 混合精度训练、梯度裁剪、学习率 warmup、Cosine Annealing。
- Mixup、CutMix、Label Smoothing 等小数据集正则化方法。
- 统计参数量、FLOPs、吞吐量、单张推理延迟和显存占用。
- 在具身/机器人第一视角数据上进行迁移学习或线性探测。
- 提供 Gradio/命令行推理 Demo。

### 暂不纳入主线

- 从零训练 ImageNet 级别模型。
- 目标检测、分割、视觉语言模型、机器人控制同时展开。
- 为追求准确率大量堆叠训练技巧，却无法解释各技巧的作用。

这些方向可以写入“未来工作”，但不应影响主项目按时闭环。

## 3. 技术方案

### 3.1 数据集

主数据集使用 **CIFAR-10**：共 10 个类别，图像大小为 `3 × 32 × 32`。建议从官方训练集划出 5,000 张作为验证集，测试集只用于最终评估。

推荐划分：

| 子集 | 数量 | 用途 |
|---|---:|---|
| Train | 45,000 | 参数学习 |
| Validation | 5,000 | 调参、早停、选择最佳权重 |
| Test | 10,000 | 最终一次性报告结果 |

基础增强：`RandomCrop(32, padding=4)`、`RandomHorizontalFlip()`、标准化。增强只施加于训练集；验证集和测试集只能标准化。划分索引和随机种子必须保存。

### 3.2 MiniViT 架构

建议先使用以下可在普通单卡上训练的配置：

| 参数 | 初始值 |
|---|---:|
| Image Size | 32 |
| Patch Size | 4 |
| Token 数量 | 64 个图像 token + 1 个 CLS token |
| Embedding Dimension | 192 |
| Transformer Depth | 6 |
| Attention Heads | 6 |
| MLP Ratio | 4 |
| Dropout | 0.1 |
| Classes | 10 |

前向流程：

```text
Image [B, 3, 32, 32]
  -> Patch Embedding [B, 64, 192]
  -> prepend CLS token [B, 65, 192]
  -> add Position Embedding
  -> 6 × Transformer Encoder Block
  -> LayerNorm
  -> CLS representation [B, 192]
  -> Linear Classifier [B, 10]
```

每个 Encoder Block 采用 Pre-Norm：

```text
x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

必须能讲清楚：

- 为什么图片可以被视为 patch token 序列。
- `Q、K、V` 的来源以及 `softmax(QK^T / sqrt(d_k))V` 的意义。
- 多头注意力为何能学习不同关系。
- CLS token 和全局平均池化的区别。
- 位置编码为什么不可缺少。
- Pre-Norm、残差连接对训练稳定性的作用。
- Patch Size 如何影响 token 数与注意力复杂度。

### 3.3 训练配置

第一版推荐配置：

| 项目 | 建议值 |
|---|---|
| Optimizer | AdamW |
| Learning Rate | `3e-4`（根据 batch size 调整） |
| Weight Decay | `0.05` |
| Epochs | 100–200 |
| Batch Size | 128（显存不足则 64） |
| Scheduler | 5–10 epoch warmup + cosine decay |
| Loss | Cross Entropy；稳定后尝试 Label Smoothing |
| Gradient Clipping | `1.0` |
| Precision | 优先 AMP |
| Seeds | 至少固定一个；最终关键实验建议运行 3 个种子 |

训练时记录：train/val loss、top-1 accuracy、learning rate、epoch time、最佳 epoch。最终实验同时报告最佳验证集对应的测试准确率，而不是挑选测试集最高结果。

## 4. 推荐目录结构

```text
mini-vit/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt 或 pyproject.toml
├── configs/
│   ├── minivit_cifar10.yaml
│   └── cnn_baseline.yaml
├── src/
│   ├── data.py
│   ├── models/
│   │   ├── attention.py
│   │   ├── transformer.py
│   │   ├── minivit.py
│   │   └── cnn_baseline.py
│   ├── engine.py
│   ├── metrics.py
│   ├── utils.py
│   └── visualization.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_attention.py
│   ├── test_model_shapes.py
│   └── test_overfit_tiny_batch.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/              # gitignore：权重与本地运行结果
└── assets/               # README 中展示的曲线和可视化
```

原则：模型与训练逻辑写在 `src/`，notebook 只负责分析和展示，避免把全部项目写进一个 `.ipynb`。

## 5. 分阶段执行计划

以下按 5 周、每周约 8–12 小时设计。时间充足时再做进阶项。

### 第 1 周：理论复习与最小数据管线

任务：

- 阅读 ViT 原论文的摘要、方法和实验部分，画出自己的结构图。
- 手算一个极小 self-attention 示例，确认缩放点积、softmax 维度和多头拆分方式。
- 创建项目环境、依赖文件和 Git 忽略规则。
- 完成 CIFAR-10 下载、数据划分、增强和 DataLoader。
- 可视化一批增强前后的样本，检查标签与归一化。

验收标准：

- 固定种子后划分可复现。
- 能打印一个 batch 的图像与标签形状。
- Windows 环境下若 DataLoader 多进程不稳定，可先使用 `num_workers=0`。

### 第 2 周：从零实现 MiniViT

按以下顺序实现并逐个测试：

1. `PatchEmbedding`：建议使用 `Conv2d(kernel_size=patch_size, stride=patch_size)`。
2. `MultiHeadSelfAttention`：自己完成 QKV 投影、reshape、transpose、attention 和输出投影。
3. `MLP`：Linear → GELU → Dropout → Linear → Dropout。
4. `TransformerEncoderBlock`：Pre-Norm + 两个残差分支。
5. `MiniViT`：CLS token、可学习位置编码、encoder stack 和分类头。

验收标准：

- 所有关键张量 shape 均有单元测试。
- 随机输入 `[2, 3, 32, 32]` 输出 `[2, 10]`。
- 模型能在 32–128 个样本上快速过拟合到接近 100% 训练准确率；若不能，应先排查实现再进行完整训练。
- 写出参数量统计，并理解参数主要分布在哪些模块。

### 第 3 周：训练闭环与基线

任务：

- 实现 train/validate/test 循环、AMP、梯度裁剪和 checkpoint。
- 保存 `last.pt` 与 `best.pt`，checkpoint 包含模型、优化器、scheduler、epoch、配置和指标。
- 记录 TensorBoard 或 CSV 日志。
- 训练 MiniViT 基准配置。
- 训练参数量尽量接近的 CNN；也可补充标准 ResNet-18 作为易理解的参考基线。

验收标准：

- 一条命令可以启动训练，意外中断后可以续训。
- 使用相同数据划分和评估代码比较模型。
- README 中给出运行环境、命令、随机种子和首版结果。

### 第 4 周：消融实验与可解释性

优先完成下面 3 类消融，不要一次改变多个变量：

| 实验 | 建议设置 | 要回答的问题 |
|---|---|---|
| Patch Size | 2 / 4 / 8 | token 数、精度、显存和速度如何变化？ |
| Depth | 3 / 6 / 9 | 更深是否一定更好？是否出现过拟合或训练困难？ |
| Data Augmentation | 基础 / 无增强 / 强增强 | 小数据训练时 ViT 对增强有多敏感？ |

有余力再比较：位置编码有/无、CLS/mean pooling、heads 数量、AdamW/SGD、Label Smoothing、Mixup/CutMix。

可视化输出：

- train/val loss 和 accuracy 曲线。
- 10 类混淆矩阵及每类 precision、recall、F1。
- 每类典型正确/错误案例。
- 最后一层 CLS-to-patch attention 或 Attention Rollout 热力图。

注意：attention 图只说明模型内部的注意力分布，不能直接宣称它是完整的因果解释。

### 第 5 周：具身方向延伸与项目包装

选择一个规模可控的延伸，推荐顺序如下：

1. **第一视角场景分类/物体状态分类**：从公开具身或 egocentric 数据集中抽取一个小而清晰的分类子任务。
2. **域偏移实验**：在 CIFAR-10-C 或自制相机扰动（模糊、亮度、遮挡、视角变化）上评估鲁棒性。
3. **迁移学习**：比较从头训练、冻结 encoder 线性探测、全量微调。

延伸实验应回答一个明确问题，例如：

> MiniViT 在干净图像上学习的表征，面对机器人相机常见的运动模糊和光照变化时会退化多少？数据增强能否改善？

最后完成：

- README 项目故事、架构图、结果表、可视化、复现命令。
- 一份 2–4 页实验报告或技术博客。
- 可选推理 Demo。
- 60 秒项目介绍和 5 分钟技术讲解稿。

## 6. 实验记录规范

每次运行至少保存以下字段：

```yaml
experiment_name:
git_commit:
seed:
dataset_split:
model:
  image_size:
  patch_size:
  embed_dim:
  depth:
  num_heads:
training:
  epochs:
  batch_size:
  optimizer:
  learning_rate:
  weight_decay:
  scheduler:
augmentation:
best_val_accuracy:
test_accuracy:
parameter_count:
flops:
throughput:
notes:
```

建议维护总表：

| Run | Model | Params | Patch | Depth | Augmentation | Val Acc | Test Acc | FLOPs | Img/s |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| baseline-01 | MiniViT | TBD | 4 | 6 | Basic | TBD | TBD | TBD | TBD |

关键对照最好运行 3 个随机种子，报告 `mean ± std`。如果受算力限制只能运行一次，需要如实说明。

## 7. 测试与质量检查

训练前必须通过：

- Patch 数量及输出 shape 测试。
- Attention 输入输出 shape 和有限值检查。
- `embed_dim % num_heads == 0` 参数校验。
- 模型前向、反向传播和梯度非空检查。
- 极小 batch 过拟合测试。
- 相同 seed 下数据划分一致性测试。

提交项目前检查：

- 全新环境能依据 README 完成安装。
- 一键命令能跑通短训练和推理。
- 仓库不包含数据集、大权重、密钥和绝对路径。
- 图表包含标题、坐标、单位、图例和实验设置。
- 所有结论都能在结果表或图中找到证据。

## 8. 项目成功标准

不要把成功只定义为某个准确率。项目达到以下条件即可形成有说服力的简历成果：

- 能不看代码画出 MiniViT 数据流，并解释每个模块。
- 从零实现与测试核心 attention，而不是只调用 `nn.TransformerEncoder`。
- 同一协议下完成 MiniViT 与 CNN 的公平比较。
- 至少 3 个单变量消融实验，并能解释现象。
- 同时报告精度、模型大小和推理效率。
- 有错误分析、注意力可视化和复现文档。
- 有一个与机器人视觉/具身视角相关的小型延伸实验。

## 9. 可直接使用的简历表述模板

完成实验后，用真实数据替换占位符：

> **Mini Vision Transformer 图像分类与鲁棒性分析｜PyTorch**  
> 从零实现 Patch Embedding、多头自注意力及 Pre-Norm Transformer Encoder，搭建支持 AMP、断点续训与配置化实验的 CIFAR-10 训练管线；模型以 **[X]M** 参数达到 **[XX.X]%** Top-1 Accuracy。设计 Patch Size、网络深度及数据增强消融实验，并从准确率、FLOPs、吞吐量和注意力热力图分析性能；在 **[具身视角/相机扰动数据]** 上评估域偏移并将鲁棒准确率提升 **[X.X] 个百分点**。

面试时不要只背结果，应准备回答：

- 为什么小数据上 ViT 可能不如 CNN？
- Patch Size 变为一半后注意力矩阵规模如何变化？
- 为什么选 AdamW？weight decay 与 Adam 中的 L2 正则有何区别？
- 如何证明对照实验公平？
- attention 可视化能说明什么、不能说明什么？
- 图像分类能力如何迁移到具身智能中的感知和决策？

## 10. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| 算力不足 | 单次训练太久 | 降低 depth/embed dim，先跑 20 epoch 冒烟实验，再跑正式实验 |
| ViT 精度不高 | 训练集高、验证集低 | 加强增强和正则；分析小数据归纳偏置，不隐瞒结果 |
| 实现错误 | loss 不降 | shape 单测、tiny-batch 过拟合、与 PyTorch attention 做数值对照 |
| 实验不可比 | 每次设置混乱 | 配置文件、固定 split、统一训练预算、记录 git commit |
| 项目过度扩张 | 同时做检测/分割/控制 | 先完成 MVP，再只选一个具身延伸问题 |
| 简历描述空泛 | 只有“实现 ViT” | 用真实精度、参数量、FLOPs、延迟和消融结论表达 |

## 11. 建议的首个里程碑

第一阶段只追求一个 **能证明实现正确的最小闭环**：

1. 建立目录与依赖。
2. 下载并检查 CIFAR-10。
3. 实现 Patch Embedding 和 Multi-Head Self-Attention。
4. 完成 MiniViT 前向与 shape 测试。
5. 在 64 个样本上成功过拟合。
6. 提交第一个可运行版本。

完成后再启动完整训练，能显著减少把算力浪费在隐藏 bug 上的风险。

## 12. 参考资料

- [Vision Transformer 原论文：An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929)
- [PyTorch 官方 CIFAR-10 分类教程](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch 官方基础教程](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

---

**推荐下一步：** 按“第 1 周”和“第 2 周”先搭建 MVP。只有当 shape 测试、反向传播测试和 tiny-batch 过拟合都通过后，再开始正式训练与调参。
