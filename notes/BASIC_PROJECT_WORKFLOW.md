# Mini Vision Transformer 基础项目工作流（实战更新版）

> 本文已根据 Mini ViT 项目的真实实现与实验结果更新。阶段 0–9 已完成；本文既保留可复用的学习路线，也作为代码、笔记、实验和命令的总导航。

最后更新：2026-09-23

## 当前进度

| 阶段 | 内容 | 状态 | 详细记录 |
|---:|---|---|---|
| 0 | 环境、依赖、Git 与项目骨架 | 已完成 | [STAGE_0_NOTES.md](STAGE_0_NOTES.md) |
| 1 | CIFAR-10 数据管线 | 已完成 | [STAGE_1_NOTES.md](STAGE_1_NOTES.md) |
| 2 | MiniViT 基础零件 | 已完成 | [STAGE_2_NOTES.md](STAGE_2_NOTES.md) |
| 3 | 完整 MiniViT 组装 | 已完成 | [STAGE_3_NOTES.md](STAGE_3_NOTES.md) |
| 4 | Tiny Overfit 学习能力验证 | 已完成 | [STAGE_4_NOTES.md](STAGE_4_NOTES.md) |
| 5 | 配置化训练闭环 | 已完成 | [STAGE_5_NOTES.md](STAGE_5_NOTES.md) |
| 6 | SimpleCNN 对照基线 | 已完成 | [STAGE_6_NOTES.md](STAGE_6_NOTES.md) |
| 7 | Patch、Depth、Augmentation 消融 | 已完成 | [STAGE_7_NOTES.md](STAGE_7_NOTES.md) |
| 8 | 冻结模型测试与可视化 | 已完成 | [STAGE_8_NOTES.md](STAGE_8_NOTES.md) |
| 9 | README 与仓库最终收尾 | 已完成 | [STAGE_9_NOTES.md](STAGE_9_NOTES.md) |

实验配置、checkpoint 哈希、训练结果和最终测试指标统一记录在 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。

## 0. 先明确最终要交付什么

完成本流程后，项目应具备以下成果：

1. 一个从零实现核心模块的 Mini Vision Transformer。
2. 一个 CIFAR-10 数据读取与划分模块。
3. 一套可以训练、验证、测试和断点续训的程序。
4. 一个用于公平比较的简单 CNN 基线。
5. 三组基础消融实验。
6. 训练曲线、混淆矩阵、错误样本和注意力图。
7. 单元测试、实验结果表和项目 README。

当前不做以下内容：

- 具身数据集迁移。
- 目标检测、分割或机器人控制。
- Mixup、CutMix、知识蒸馏等额外技巧。
- 分布式训练、部署或在线 Demo。
- 追求论文级精度。

项目的第一个目标是“**正确、可运行、可解释、可复现**”，而不是“指标最高”。

---

## 1. 项目全貌：先分再总

把整个项目想成 6 个互相解耦的系统：

```text
配置系统 ───────────────────┐
                           v
数据系统 ──> 模型系统 ──> 训练系统 ──> 评估系统 ──> 展示与文档
              ^              |
              |              v
           单元测试       权重与训练日志
```

各系统只承担一种职责：

| 系统 | 只负责什么 | 不负责什么 |
|---|---|---|
| 配置 | 保存超参数和路径 | 不创建模型、不读取数据 |
| 数据 | 返回图像和标签 | 不关心模型结构 |
| 模型 | 输入图像，输出分类 logits | 不下载数据、不执行训练循环 |
| 训练 | 更新参数、验证、保存 checkpoint | 不绘图、不分析错误样本 |
| 评估 | 加载权重并计算指标 | 不修改模型参数 |
| 可视化 | 把已有日志和预测结果变成图 | 不重新训练模型 |

这样的好处是：某一步出错时，只需要检查对应模块。例如，模型输出 shape 错误时，不需要怀疑绘图代码；训练准确率异常时，也不需要先修改 README。

### 推荐完成顺序

```text
阶段 0  环境与骨架
  ↓
阶段 1  数据最小闭环
  ↓
阶段 2  模型零件逐个实现
  ↓
阶段 3  模型组装与测试
  ↓
阶段 4  极小数据过拟合
  ↓
阶段 5  正式训练闭环
  ↓
阶段 6  CNN 对照基线
  ↓
阶段 7  三组消融实验
  ↓
阶段 8  评估与可视化
  ↓
阶段 9  README 与项目收尾
```

**硬性规则：每个阶段的验收项全部通过后，再进入下一阶段。**

---

## 2. 最终目录结构

初学阶段不需要过度拆分文件，采用下面的结构即可：

```text
Mini ViT/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── .gitignore
├── notes/
│   ├── BASIC_PROJECT_WORKFLOW.md
│   ├── EXPERIMENT_RESULTS.md
│   └── STAGE_0_NOTES.md ... STAGE_8_NOTES.md
├── configs/
│   ├── minivit.yaml
│   ├── cnn.yaml
│   └── ablations/
│       ├── patch_p2.yaml
│       ├── patch_p8.yaml
│       ├── depth_d3.yaml
│       ├── depth_d9.yaml
│       └── no_augmentation.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── metrics.py
│   ├── utils.py
│   ├── visualization.py
│   └── models/
│       ├── __init__.py
│       ├── minivit.py
│       └── cnn.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_cnn.py
│   ├── test_engine.py
│   ├── test_train_script.py
│   ├── test_evaluate_script.py
│   ├── test_visualization.py
│   ├── test_visualize_script.py
│   ├── test_utils.py
│   └── test_tiny_overfit.py
├── data/                 # 不提交到 Git
├── outputs/              # 不提交权重到 Git
│   └── experiment_name/
│       ├── config.yaml
│       ├── best.pt
│       ├── last.pt
│       ├── history.csv
│       └── evaluation/
│           ├── metrics.json
│           └── predictions.csv
└── assets/               # 提交用于 README 展示的最终图表
    ├── training_curves/
    ├── confusion_matrices/
    ├── error_cases/
    └── attention/
```

项目实际保留了这种学习友好的组织：Patch Embedding、Attention、MLP、Encoder Block 和完整 MiniViT 都在 `src/models/minivit.py` 中，用不同类隔开。模型职责仍然清楚，也便于沿着一次前向传播连续阅读。

---

## 3. 固定的基础技术方案

项目第一版先固定以下设置，不要边写边调参。

### 3.1 数据设置

| 项目 | 固定设置 |
|---|---|
| 数据集 | CIFAR-10 |
| 原训练集 | 50,000 张 |
| 项目训练集 | 45,000 张 |
| 项目验证集 | 5,000 张 |
| 测试集 | 官方测试集 10,000 张 |
| 图像 shape | `[3, 32, 32]` |
| 类别数 | 10 |
| 随机种子 | 42 |

训练集变换：

```text
RandomCrop(32, padding=4)
RandomHorizontalFlip()
ToTensor()
Normalize(mean, std)
```

验证集和测试集变换：

```text
ToTensor()
Normalize(mean, std)
```

注意：验证集和测试集不能使用随机裁剪、随机翻转等随机增强。

### 3.2 MiniViT 第一版设置

| 参数 | 数值 |
|---|---:|
| image size | 32 |
| patch size | 4 |
| patch 数量 | `(32 / 4)² = 64` |
| CLS token 数量 | 1 |
| 最终序列长度 | 65 |
| embedding dimension | 192 |
| encoder depth | 6 |
| attention heads | 6 |
| 单头维度 | `192 / 6 = 32` |
| MLP hidden dimension | `192 × 4 = 768` |
| dropout | 0.1 |
| classes | 10 |

### 3.3 第一版训练设置

| 项目 | 数值 |
|---|---|
| loss | CrossEntropyLoss |
| optimizer | AdamW |
| learning rate | `3e-4` |
| weight decay | `0.05` |
| batch size | 128；显存不足改为 64 |
| epochs | 先跑 2 个 epoch 冒烟测试，再跑 100 个 epoch |
| 模型选择 | 验证集准确率最高的 checkpoint |

为了降低第一次实现的难度，第一版暂不加入 warmup、AMP、Mixup、CutMix 和复杂 scheduler。完整流程正确后，可以再单独增加这些能力。

---

## 4. 阶段 0：建立环境和空项目骨架

### 4.1 本阶段目标

只确认三件事：Python 能运行、PyTorch 能导入、项目目录清晰。此时不写模型。

### 4.2 需要完成的任务

1. 创建独立 Python 虚拟环境。
2. 安装 PyTorch、torchvision、PyYAML、NumPy、Matplotlib、scikit-learn、pytest。
3. 创建第 2 节中的目录和空文件。
4. 在 `.gitignore` 中排除虚拟环境、数据、权重、缓存和临时输出。
5. 初始化 Git 仓库并做第一次提交。

建议依赖先保持简单：

```text
torch
torchvision
numpy
pyyaml
matplotlib
scikit-learn
pytest
```

### 4.3 你需要理解的知识

- 虚拟环境用于隔离不同项目的依赖。
- `requirements.txt` 用于告诉别人项目需要哪些包。
- Git 记录代码历史，但数据集和大模型权重通常不提交。

### 4.4 验收关卡

- [x] `python --version` 能显示版本。
- [x] Python 中可以成功执行 `import torch` 和 `import torchvision`。
- [x] 能打印 `torch.cuda.is_available()` 的结果。
- [x] `pytest` 可以正常启动。
- [x] 目录中没有把虚拟环境、数据集或权重加入 Git。

### 4.5 常见问题

- PyTorch 与 CUDA 版本不匹配：先使用官方安装方式确认版本，不要同时混用多个包管理器。
- Windows 路径带空格：执行命令时给路径加引号。
- 没有独立显卡：仍可先在 CPU 上完成 shape 测试和小数据测试，正式训练再使用可用 GPU。

---

## 5. 阶段 1：完成数据最小闭环

### 5.1 本阶段目标

让 `src/data.py` 独立完成“下载数据 → 固定划分 → 应用变换 → 返回 DataLoader”。此时模型仍不存在。

### 5.2 模块接口

建议提供一个函数：

```python
def build_dataloaders(data_dir, batch_size, num_workers, seed):
    return train_loader, val_loader, test_loader, class_names
```

输入：

- `data_dir`：数据保存位置。
- `batch_size`：一个 batch 中的样本数。
- `num_workers`：并行读取数据的进程数。
- `seed`：保证数据划分可复现。

输出：

- `train_loader`：带训练增强的 45,000 张图片。
- `val_loader`：不带随机增强的 5,000 张图片。
- `test_loader`：官方测试集。
- `class_names`：10 个类别名称。

### 5.3 推荐实现顺序

1. 先只下载 CIFAR-10，不做划分。
2. 读取一个样本，打印图片类型、shape 和标签。
3. 定义 train transform 和 eval transform。
4. 使用固定 seed 生成 45,000/5,000 的索引。
5. 确保训练与验证子集使用不同 transform。
6. 创建三个 DataLoader。
7. 从每个 DataLoader 读取一个 batch 并打印 shape。
8. 写 `tests/test_data.py`。

### 5.4 预期输入输出

```text
images.shape = [B, 3, 32, 32]
labels.shape = [B]
images.dtype = torch.float32
labels.dtype = torch.int64
label range = 0 到 9
```

### 5.5 必须做的检查

- 训练集和验证集索引不能重叠。
- 三个数据集的数量必须分别为 45,000、5,000、10,000。
- 相同 seed 运行两次，验证集索引必须一致。
- 图片中不能出现 NaN 或 Inf。
- 可视化反归一化后的图片，确认颜色正常。

### 5.6 验收关卡

- [x] 能独立运行数据模块并打印三个 batch 的 shape。
- [x] 数据数量正确，训练/验证索引无重叠。
- [x] 固定 seed 后划分不变。
- [x] 训练集有随机增强，验证集和测试集没有。
- [x] `tests/test_data.py` 通过。

### 5.7 常见问题

- **验证集也发生随机变化**：通常是训练集和验证集引用了同一个带增强的 dataset 对象。
- **Windows DataLoader 报多进程错误**：先将 `num_workers` 设为 0，并确保主程序在 `if __name__ == "__main__":` 中启动。
- **图片显示颜色异常**：绘图前需要执行反归一化。

---

## 6. 阶段 2：逐个实现 MiniViT 零件

本阶段只做模型前向传播，不做训练循环。每实现一个类，就立刻写测试，不要等全部写完再测试。

### 6.1 零件 A：Patch Embedding

职责：把二维图片切成 patch，并把每个 patch 映射成一个 token。

推荐接口：

```python
class PatchEmbedding(nn.Module):
    def forward(self, x):
        # x: [B, 3, 32, 32]
        # output: [B, 64, 192]
```

推荐使用：

```python
nn.Conv2d(
    in_channels=3,
    out_channels=192,
    kernel_size=4,
    stride=4,
)
```

卷积输出先是 `[B, 192, 8, 8]`，再展平空间维并交换维度，得到 `[B, 64, 192]`。

检查点：

- 图片尺寸必须能被 patch size 整除。
- patch 数量是 `8 × 8 = 64`。
- 不要把 channel 维和 token 维弄反。

验收：

- [x] 输入 `[2, 3, 32, 32]`，输出严格为 `[2, 64, 192]`。
- [x] 输入尺寸不合法时给出清晰报错。

### 6.2 零件 B：Multi-Head Self-Attention

职责：让每个 token 根据内容聚合其他 token 的信息。

推荐接口：

```python
class MultiHeadSelfAttention(nn.Module):
    def forward(self, x, return_attention=False):
        # x: [B, N, D]
        # output: [B, N, D]
        # optional attention: [B, H, N, N]
```

固定符号：

```text
B = batch size
N = token 数，本项目通常为 65
D = embedding dimension，本项目为 192
H = attention heads，本项目为 6
Dh = 每个 head 的维度，192 / 6 = 32
```

按以下步骤实现：

1. 使用一个 Linear 同时生成 Q、K、V：`[B, N, D] -> [B, N, 3D]`。
2. reshape 成 `[B, N, 3, H, Dh]`。
3. 调整维度，分别得到 `[B, H, N, Dh]` 的 Q、K、V。
4. 计算 `Q @ K.transpose(-2, -1)`，得到 `[B, H, N, N]`。
5. 除以 `sqrt(Dh)`。
6. 在最后一维执行 softmax。
7. attention 与 V 相乘，得到 `[B, H, N, Dh]`。
8. 合并多头，恢复 `[B, N, D]`。
9. 通过输出 Linear 和 Dropout。

必须检查：

- `D % H == 0`。
- softmax 使用 `dim=-1`。
- attention 每一行之和应接近 1。
- 输出中没有 NaN 或 Inf。

验收：

- [x] 输入 `[2, 65, 192]`，输出 `[2, 65, 192]`。
- [x] attention shape 为 `[2, 6, 65, 65]`。
- [x] attention 最后一维求和接近 1。
- [x] 反向传播后 QKV Linear 的参数具有梯度。

### 6.3 零件 C：MLP

职责：对每个 token 分别进行非线性特征变换。

结构：

```text
Linear(192, 768)
  -> GELU
  -> Dropout
  -> Linear(768, 192)
  -> Dropout
```

验收：

- [x] 输入和输出 shape 都是 `[B, N, 192]`。
- [x] MLP 不改变 token 数量。

### 6.4 零件 D：Transformer Encoder Block

职责：组合 LayerNorm、Attention、MLP 和残差连接。

使用 Pre-Norm：

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

不要写成覆盖残差输入的形式。建议明确保存分支结果，方便调试。

验收：

- [x] 输入输出 shape 都是 `[2, 65, 192]`。
- [x] 前向结果有限，无 NaN/Inf。
- [x] loss 反向传播后 Attention 和 MLP 都有梯度。

### 6.5 为什么要逐个测试

如果直接写完整 MiniViT 后发现 loss 不下降，错误可能来自 patch 排列、QKV reshape、softmax 维度、残差连接或分类头。逐个测试后，每个零件都有明确的输入输出契约，排错范围会小很多。

---

## 7. 阶段 3：组装完整 MiniViT

### 7.1 本阶段目标

把经过测试的零件组合成完整模型，使一批图片能够输出 10 类 logits。

### 7.2 组装顺序

1. 图片经过 Patch Embedding：`[B, 3, 32, 32] -> [B, 64, 192]`。
2. 创建可学习 CLS token：单个参数 shape 为 `[1, 1, 192]`。
3. 在 batch 维扩展 CLS token，并拼接到序列开头：`[B, 65, 192]`。
4. 加上可学习位置编码：shape 为 `[1, 65, 192]`。
5. 依次通过 6 个 Encoder Block。
6. 通过最终 LayerNorm。
7. 取第 0 个 token，即 CLS 表征：`[B, 192]`。
8. 通过 Linear 分类头：`[B, 10]`。

推荐接口：

```python
class MiniViT(nn.Module):
    def forward(self, images, return_attention=False):
        # logits: [B, 10]
```

### 7.3 参数初始化

第一版可采用 PyTorch 模块默认初始化，只对 CLS token 和位置编码使用较小标准差的截断正态分布。重点是保持方案简单且记录清楚，不要混用多个未知来源的初始化策略。

### 7.4 测试清单

在 `tests/test_model.py` 中覆盖：

- Patch Embedding shape。
- Attention shape 和概率和。
- Encoder Block shape。
- 完整模型输出 shape。
- 错误输入尺寸能否被发现。
- 完整模型能否完成一次反向传播。
- 关键参数是否收到非空且有限的梯度。

### 7.5 验收关卡

- [x] 输入 `[2, 3, 32, 32]`，输出 `[2, 10]`。
- [x] CrossEntropyLoss 能正常计算。
- [x] `loss.backward()` 不报错。
- [x] Patch Embedding、Attention、MLP、分类头都有梯度。
- [x] 所有模型测试通过。

到这里为止，只能说明“代码在数学形状上可以运行”，还不能说明模型真的能学习。下一阶段用 tiny overfit 验证学习能力。

---

## 8. 阶段 4：极小数据过拟合测试

### 8.1 本阶段目标

让模型记住 64 张训练图片。这个测试不是为了泛化，而是验证数据、前向、loss、反向传播和优化器能共同工作。

### 8.2 流程

1. 从训练集中固定取 64 张图片。
2. 关闭随机数据增强，保证每轮看到相同输入。
3. 暂时将 dropout 设为 0。
4. 使用较小模型或当前 MiniViT 均可。
5. 在同一批数据上反复训练。
6. 记录 loss 和训练准确率。

### 8.3 预期现象

- loss 持续明显下降。
- 训练准确率逐步接近 100%。
- 如果长期停留在随机猜测水平（CIFAR-10 约 10%），说明实现存在问题。

### 8.4 失败时按顺序排查

1. 标签是否处于 0–9，dtype 是否为 `torch.int64`。
2. 模型最终输出是否是原始 logits；不要提前做 softmax。
3. 是否调用 `optimizer.zero_grad()`、`loss.backward()` 和 `optimizer.step()`。
4. 模型是否处于 `train()` 模式。
5. 学习率是否过小或过大。
6. 参数梯度是否存在且有限。
7. Attention softmax 维度是否正确。
8. 残差连接是否正确。

### 8.5 验收关卡

- [x] 固定极小样本的训练 loss 显著下降。
- [x] 训练准确率能够持续接近完全记忆。
- [x] 测试可通过 pytest 重复执行。

**tiny overfit 不通过时，不允许开始 100 epoch 正式训练。**

---

## 9. 阶段 5：搭建正式训练闭环

### 9.1 本阶段目标

将训练拆成可复用函数，使 MiniViT 和后续 CNN 能使用完全相同的训练与评估逻辑。

### 9.2 `engine.py` 的职责

项目最终提供四类核心函数：

```python
train_one_epoch(...)             # 训练并更新参数
evaluate(...)                    # 验证，只累计聚合指标
evaluate_with_predictions(...)   # 最终测试，同时保留逐样本预测
fit(...)                         # 组织 epoch、日志和 checkpoint
```

不要把 CIFAR-10 下载逻辑写入 `engine.py`，也不要在这里画图。

### 9.3 单个训练 batch 的固定顺序

```text
将 images、labels 移到 device
  -> optimizer.zero_grad()
  -> logits = model(images)
  -> loss = criterion(logits, labels)
  -> loss.backward()
  -> optimizer.step()
  -> 累加 loss、正确数和样本数
```

epoch loss 必须按样本数加权：

```text
total_loss += batch_loss × batch_size
epoch_loss = total_loss / total_samples
```

epoch accuracy：

```text
predictions = logits.argmax(dim=1)
correct += (predictions == labels).sum()
accuracy = correct / total_samples
```

### 9.4 验证流程的区别

验证时必须：

- 调用 `model.eval()`。
- 使用 `torch.no_grad()`。
- 不调用 backward 和 optimizer.step。
- 使用验证集，不使用测试集。

### 9.5 checkpoint 设计

每轮保存 `last.pt`，验证准确率刷新时保存 `best.pt`。内容至少包含：

```python
{
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "best_val_accuracy": best_val_accuracy,
    "history": history,
    "config": config,
}
```

断点续训时恢复模型、优化器、当前 epoch 和历史最佳指标。

### 9.6 日志设计

每轮向 `history.csv` 写入：

```text
epoch,train_loss,train_accuracy,val_loss,val_accuracy,epoch_time_seconds
```

训练结束后再由可视化脚本读取 CSV 绘图。训练代码本身不负责展示图表。

### 9.7 分三次运行，不要直接长时间训练

1. **单 batch 测试**：只运行一个 batch，确认无错误。
2. **2 epoch 冒烟测试**：确认训练、验证、日志和 checkpoint 全部产生。
3. **正式 100 epoch**：只有前两步通过后才开始。

本项目的冒烟训练必须显式使用独立输出目录，防止覆盖正式结果：

```powershell
python -m scripts.train --config configs/minivit.yaml --epochs 2 --max-batches 2 --output-dir outputs/smoke_minivit
```

正式训练与断点续训：

```powershell
python -m scripts.train --config configs/minivit.yaml
python -m scripts.train --config configs/minivit.yaml --resume outputs/minivit_baseline/last.pt
```

`--epochs` 表示目标总 epoch，不是从 checkpoint 起额外增加的轮数。

### 9.8 验收关卡

- [x] 单 batch 训练成功。
- [x] 2 epoch 后生成 `last.pt`、`best.pt` 和 `history.csv`。
- [x] 中断后能从 `last.pt` 接着训练。
- [x] train loss 总体下降，准确率明显高于随机猜测。
- [x] 测试集没有参与模型选择。

---

## 10. 阶段 6：实现 CNN 对照基线

### 10.1 为什么需要 CNN

只有 MiniViT 的单个准确率很难说明模型表现好坏。CNN 可以提供一个熟悉且合理的参照，帮助分析 ViT 在小数据集上的特点。

### 10.2 解耦原则

CNN 只新增 `src/models/cnn.py` 和 `configs/cnn.yaml`。以下内容必须复用：

- 相同的训练/验证/测试划分。
- 相同的数据增强。
- 相同的训练 epoch 数。
- 相同的指标计算代码。
- 相同的 checkpoint 和日志格式。

训练脚本根据配置中的 `model.name` 创建不同模型：

```text
model.name = minivit -> 创建 MiniViT
model.name = cnn     -> 创建 SimpleCNN
```

### 10.3 CNN 实际结构

项目使用 4 个卷积模块，通道数为 `96 → 192 → 384 → 512`：

```text
4 × (Conv -> BatchNorm -> ReLU -> MaxPool)
AdaptiveAvgPool
Flatten -> Dropout -> Linear(512, 10)
```

CNN 有 2,609,002 个参数，MiniViT 有 2,693,578 个参数；差异约 3.14%，足够作为接近参数量而非完全相同参数量的对照。

### 10.4 公平比较表

| 条件 | MiniViT | CNN |
|---|---|---|
| 数据划分 | 相同 | 相同 |
| 数据增强 | 相同 | 相同 |
| epoch | 100 | 100 |
| batch size | 相同 | 相同 |
| 最佳模型选择 | 验证准确率 | 验证准确率 |
| 测试次数 | 最终一次 | 最终一次 |

优化器可优先统一使用 AdamW，减少变量。如果后面需要给 CNN 使用不同优化器，应单独说明原因，不能把结果称为严格单变量比较。

### 10.5 验收关卡

- [x] CNN 输出 `[B, 10]`。
- [x] CNN 通过 tiny overfit。
- [x] CNN 使用相同的 `engine.py` 完成训练。
- [x] 结果表包含两个模型的参数量、最佳验证准确率和测试准确率。

---

## 11. 阶段 7：完成三组基础消融实验

### 11.1 消融实验是什么

消融实验的核心是：**一次只改变一个因素，其余条件保持不变**，观察这个因素带来的影响。

基础项目完成以下三组即可。

### 11.2 实验 A：Patch Size

| Run | Patch Size | Token 数 | 其他设置 |
|---|---:|---:|---|
| A1 | 2 | 256 + CLS | 完全相同 |
| A2 | 4 | 64 + CLS | 完全相同 |
| A3 | 8 | 16 + CLS | 完全相同 |

观察：验证/测试准确率、单 epoch 时间、是否显存不足。

要解释的问题：patch 更小保留更多局部细节，但 token 数增加，attention 计算量也明显增加。

### 11.3 实验 B：Encoder 深度

| Run | Depth | 其他设置 |
|---|---:|---|
| B1 | 3 | 完全相同 |
| B2 | 6 | 完全相同 |
| B3 | 9 | 完全相同 |

观察：模型参数量、训练/验证准确率、训练时间、是否过拟合。

### 11.4 实验 C：数据增强

| Run | 设置 | 训练变换 |
|---|---|---|
| C1 | 无增强 | ToTensor + Normalize |
| C2 | 基础增强 | RandomCrop + RandomHorizontalFlip + Normalize |

这一组只需两项即可。观察训练集与验证集之间的差距，并分析增强是否改善泛化。

### 11.5 控制实验成本

第一次可先给所有候选配置运行 10–20 epoch，排除明显错误；确认设置合理后，再执行正式训练。短跑结果和正式结果要分开记录，不要放进同一张最终表混用。

### 11.6 实验命名

项目实际命名：

```text
minivit_baseline
cnn_baseline
ablation_patch_p2_d6_aug_seed42
ablation_patch_p8_d6_aug_seed42
ablation_depth_p4_d3_aug_seed42
ablation_depth_p4_d9_aug_seed42
ablation_aug_p4_d6_noaug_seed42
```

### 11.7 结果记录表

| Run | Patch | Depth | Aug | Params | Best Val Acc | Test Acc | Epoch Time |
|---|---:|---:|---|---:|---:|---:|---:|
| p4/d6 baseline | 4 | 6 | Basic | 2,693,578 | **81.84%** | **80.65%** | 25.64 s |
| patch p2 | 2 | 6 | Basic | 2,723,530 | 80.02% | 78.94% | 97.72 s* |
| patch p8 | 8 | 6 | Basic | 2,712,010 | 77.30% | 76.38% | 10.03 s |
| depth d3 | 4 | 3 | Basic | 1,358,986 | 81.16% | 80.27% | 12.37 s |
| depth d9 | 4 | 9 | Basic | 4,028,170 | 81.46% | 80.36% | 30.09 s |
| no augmentation | 4 | 6 | None | 2,693,578 | 72.64% | 70.93% | 19.72 s |

\* p2 的 epoch 59 因电脑睡眠记录了异常耗时，正常平均值排除了该点。实际排序为：准确率 `p4 > p2 > p8`，速度 `p8 > p4 > p2`；深度实验未观察到 d9 相比 d6 的收益；移除增强造成最大泛化退化。

### 11.8 验收关卡

- [x] 每组实验只改变一个变量。
- [x] 所有实验使用同一数据划分。
- [x] 每次运行均保存独立配置、日志和权重。
- [x] 已在阶段 7 笔记中记录实验前猜想、结果与反思。

---

## 12. 阶段 8：测试集评估与可视化

### 12.1 正确使用测试集

训练期间只看训练集和验证集。配置选定后，加载对应 `best.pt`，在测试集上执行最终评估。不要根据测试集结果继续修改超参数，否则测试集就间接参与了调参。

### 12.2 `evaluate.py` 输出

至少输出：

- test loss。
- test top-1 accuracy。
- 每个样本的真实标签与预测标签。
- 10 × 10 混淆矩阵所需数据。

项目将结果保存到各实验的 `evaluation/metrics.json`，逐样本预测保存到 `evaluation/predictions.csv`，供绘图脚本读取。`metrics.json` 同时记录 checkpoint epoch 与 SHA-256，确保结果可追溯。

### 12.3 四类必做可视化

#### A. 训练曲线

从 `history.csv` 绘制：

- train loss 与 val loss。
- train accuracy 与 val accuracy。

用于判断：是否收敛、是否过拟合、最佳模型大概在哪个 epoch。

#### B. 混淆矩阵

横轴统一标为预测类别，纵轴统一标为真实类别，并在图中写清楚。用于查看哪些类别容易互相混淆。

#### C. 错误样本

展示若干测试图片，并标注：

```text
Ground Truth: cat
Prediction: dog
Confidence: 0.xx
```

至少人工归纳 2–3 类错误原因，例如主体过小、背景干扰、类别外观相似或图像模糊。

#### D. Attention 热力图

为了便于可视化，模型前向允许返回最后一个 Encoder Block 的 attention：

1. 取 CLS token 对 64 个图像 patch 的注意力。
2. 对 6 个 head 求平均，或分别展示。
3. reshape 为 `[8, 8]`。
4. 插值到 `[32, 32]`。
5. 叠加到原图。

注意力图只能描述模型把注意力权重分配到了哪里，不能直接证明这些区域对预测具有因果作用。

### 12.4 验收关卡

- [x] 7 组测试评估均明确加载冻结的 `best.pt`。
- [x] 最终表包含 MiniViT、CNN 和五个消融结果。
- [x] 训练曲线坐标、图例和 best epoch 标记清楚。
- [x] 混淆矩阵固定为纵轴真实类别、横轴预测类别。
- [x] 错误案例包含真实类别、预测类别和置信度。
- [x] Attention 图同时包含正确与错误预测，并注明解释边界。

### 12.5 实际评估和可视化命令

最终评估只接受名为 `best.pt` 的冻结 checkpoint：

```powershell
python -m scripts.evaluate --checkpoint outputs/minivit_baseline/best.pt
python -m scripts.evaluate --checkpoint outputs/cnn_baseline/best.pt
```

局部冒烟评估必须使用独立目录：

```powershell
python -m scripts.evaluate --checkpoint outputs/minivit_baseline/best.pt --max-batches 2 --output-dir outputs/evaluation_smoke
```

在全部正式评估完成后生成图表：

```powershell
python -m scripts.visualize
```

最终测试结果：

| 模型 / 实验 | Test Loss | Test Acc |
|---|---:|---:|
| SimpleCNN baseline | **0.4514** | **88.29%** |
| MiniViT baseline | 0.7335 | 80.65% |
| MiniViT depth 9 | 0.8347 | 80.36% |
| MiniViT depth 3 | 0.7347 | 80.27% |
| MiniViT patch 2 | 0.7653 | 78.94% |
| MiniViT patch 8 | 0.8077 | 76.38% |
| MiniViT no augmentation | 1.5201 | 70.93% |

测试集延续了验证集上的主要结论，没有用于重新选择 epoch 或继续调参。阶段 8 完成时全量自动化测试为 `184 passed`。

---

## 13. 阶段 9：README 与项目收尾

README 是他人理解项目的入口，不要只写安装命令。

### 13.1 README 推荐结构

```text
1. 项目简介
2. 项目目标
3. MiniViT 架构图与数据流
4. 目录结构
5. 环境安装
6. 数据准备
7. 训练、续训、评估和可视化命令
8. MiniViT 与 CNN 结果表
9. 三组消融实验结果与结论
10. 训练曲线、混淆矩阵、错误案例、attention 图
11. 已知限制
12. 参考资料
```

### 13.2 README 中必须回答的问题

- 这个项目解决什么问题？
- 哪些模块是自己实现的？
- 输入经过 MiniViT 时 shape 如何变化？
- 如何复现实验？
- MiniViT 和 CNN 谁表现更好，为什么可能出现该结果？
- Patch Size、Depth、数据增强分别产生了什么影响？
- 项目有哪些不足？

### 13.3 最终仓库检查

- [x] README 提供固定依赖和虚拟环境安装步骤。
- [x] 所有项目文档使用相对路径或可移植占位路径。
- [x] 全量自动化测试为 `184 passed`。
- [x] 2 epoch 隔离冒烟训练命令已经验收。
- [x] 评估命令可从冻结的 `best.pt` 生成指标与逐样本预测。
- [x] 可视化脚本可从日志和预测文件生成 7 张正式图片。
- [x] Git 中没有数据集、大权重、虚拟环境或个人信息。
- [x] README 中的数字与实验结果总账一致。

---

## 14. 每次开发时使用的“小循环”

为了避免一次改动太多，每个模块都遵循同一个循环：

```text
只选一个小目标
  -> 写清楚输入与输出 shape
  -> 实现最少代码
  -> 用随机张量或极小数据测试
  -> 检查数值、shape 和梯度
  -> 通过后提交 Git
  -> 再进入下一个目标
```

一次合理的开发任务示例：

> 今天只实现 PatchEmbedding，使 `[B, 3, 32, 32]` 正确变为 `[B, 64, 192]`，补充两个 shape 测试，然后提交。

一次过大的任务示例：

> 今天实现整个 ViT、训练 100 epoch、调参并画注意力图。

后者会让多个错误混在一起，很难判断问题来自哪里。

---

## 15. 遇到问题时的统一排错顺序

### 15.1 程序直接报错

1. 阅读报错最后一行，确认错误类型。
2. 找到第一个属于项目代码的堆栈位置。
3. 打印该位置前后的 tensor shape、dtype 和 device。
4. 用一个 batch 或随机输入复现，不要用完整训练复现。
5. 修复后增加一个能覆盖该问题的测试。

### 15.2 程序不报错，但 loss 不下降

按以下顺序检查：

1. tiny overfit 是否通过。
2. 标签、logits shape 和 loss 用法是否正确。
3. optimizer 是否确实更新参数。
4. 梯度是否为 None、NaN、全零或异常大。
5. 学习率是否合理。
6. 模型是否在训练时使用 `train()`、验证时使用 `eval()`。
7. 数据归一化是否正确。

### 15.3 训练很好，验证很差

这通常是过拟合，而不一定是代码错误。先检查：

- 训练/验证划分是否正确且无重叠。
- 验证集是否错误地带有随机增强。
- 是否一直根据测试集调参。
- 数据增强和 dropout 是否实际启用。

在基础项目范围内，只需通过数据增强、合理模型规模和诚实分析处理，不必立刻添加大量高级技巧。

---

## 16. 推荐时间安排

不要求严格按周完成，以通过关卡为准。

| 时间 | 主要任务 | 阶段结束成果 |
|---|---|---|
| 第 1 周 | 环境、目录、数据模块 | 三个 DataLoader 与数据测试 |
| 第 2 周 | Patch、Attention、MLP、Encoder | 所有模型零件测试通过 |
| 第 3 周 | MiniViT 组装、tiny overfit | 模型能够正确学习小数据 |
| 第 4 周 | 训练循环、日志、checkpoint | MiniViT 正式基准结果 |
| 第 5 周 | CNN 基线、三组消融 | 完整实验结果表 |
| 第 6 周 | 评估、可视化、README | 可展示、可复现的最终仓库 |

如果某周没有完成，不要跳过验收项赶进度。对第一个项目而言，定位并修复问题本身就是重要学习内容。

---

## 17. 每个阶段的最终产物清单

| 阶段 | 产物 | 通过标志 |
|---|---|---|
| 0 | 环境、目录、依赖 | PyTorch 和 pytest 可运行 |
| 1 | `data.py`、数据测试 | 三个 DataLoader 正确 |
| 2 | 四个模型零件 | 独立 shape/梯度测试通过 |
| 3 | 完整 `MiniViT` | 输出 `[B, 10]` 且可反传 |
| 4 | tiny overfit 脚本 | 64 张样本接近完全记忆 |
| 5 | 训练引擎、日志、checkpoint | 2 epoch 冒烟与续训成功 |
| 6 | CNN | 共用训练引擎并得到基线结果 |
| 7 | 三组消融 | 单变量结果完整可解释 |
| 8 | 指标与四类图 | 能支持项目结论 |
| 9 | README | 他人可按说明复现 |

---

## 18. 当前状态与下一步

阶段 0–9 的基础项目主线已经全部完成：代码、测试、正式实验、最终评估、图表、学习笔记和仓库 README 均已闭环。

接下来没有必须继续执行的基础阶段。若继续研究，可从多随机种子、warmup + cosine scheduler、AMP、Mixup/CutMix、效率统计或新数据集迁移中选择一个方向，并建立新的实验阶段与独立验证协议。不要覆盖当前冻结结果，也不要使用测试集反复试错。
