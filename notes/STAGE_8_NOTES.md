# 阶段 8：测试集评估与可视化

本阶段在所有模型配置、训练轮数和最佳 checkpoint 已经通过验证集冻结后，统一进行一次测试集评估，并生成训练曲线、混淆矩阵、错误样本和 Attention 热力图。测试集只用于最终报告，不参与调参或模型选择。

开始日期：2026-09-23

当前状态：已完成
当前位置：7 组冻结模型已完成最终测试，报告图表与结果总表已生成

## 阶段概览

### 阶段目标

```text
frozen best.pt
  → test DataLoader
  → loss / top-1 accuracy
  → per-sample predictions
  → metrics.json + predictions.csv
  → training curves
  → confusion matrix
  → error cases
  → attention heatmaps
```

### 测试集使用原则

训练和消融分析只使用训练集与验证集。进入本阶段前，模型结构、超参数、训练轮数和 `best.pt` 已依据验证准确率确定。

从现在开始允许查看测试集结果，但必须遵守：

- 每个冻结 checkpoint 使用同一评估代码；
- 不根据测试准确率修改配置后重新训练；
- 不从多个 epoch 中挑选测试结果，始终加载已保存的 `best.pt`；
- 测试结果可以解释最终泛化，但不能反过来充当新的验证集；
- 若后续开发新模型，应视为新一轮研究并重新建立独立评估协议。

## 进度总览

- [x] 冻结待评估 checkpoint 名单与选择规则
- [x] 设计 `metrics.json` 与 `predictions.csv` 格式
- [x] 实现可复用的测试集预测与指标累计
- [x] 实现 `scripts/evaluate.py` 命令行入口
- [x] 由 AI 编写并通过评估代码测试
- [x] 对 7 个冻结的 `best.pt` 各评估一次
- [x] 更新最终实验结果总表
- [x] 生成训练 loss/accuracy 曲线
- [x] 生成混淆矩阵
- [x] 展示并分析错误样本
- [x] 提取并展示 MiniViT Attention 热力图
- [x] 阶段 8 总验收

## 1. 冻结评估清单

以下名单在读取测试指标之前固定。所有 checkpoint 都由验证准确率选择：

| EXP | 模型 | 配置 | Best Epoch | Best Checkpoint |
|---|---|---|---:|---|
| EXP-001 | MiniViT baseline | p4 / d6 / basic aug | 80 | `outputs/minivit_baseline/best.pt` |
| EXP-002 | SimpleCNN baseline | basic aug | 74 | `outputs/cnn_baseline/best.pt` |
| EXP-003 | MiniViT patch 2 | p2 / d6 / basic aug | 51 | `outputs/ablation_patch_p2_d6_aug_seed42/best.pt` |
| EXP-004 | MiniViT patch 8 | p8 / d6 / basic aug | 76 | `outputs/ablation_patch_p8_d6_aug_seed42/best.pt` |
| EXP-005 | MiniViT depth 3 | p4 / d3 / basic aug | 95 | `outputs/ablation_depth_p4_d3_aug_seed42/best.pt` |
| EXP-006 | MiniViT depth 9 | p4 / d9 / basic aug | 96 | `outputs/ablation_depth_p4_d9_aug_seed42/best.pt` |
| EXP-007 | MiniViT no augmentation | p4 / d6 / no aug | 94 | `outputs/ablation_aug_p4_d6_noaug_seed42/best.pt` |

checkpoint 的 SHA-256 已记录在 `notes/EXPERIMENT_RESULTS.md`。阶段 8 不覆盖这些训练产物。

## 2. 统一评估协议

所有模型使用：

```text
dataset：CIFAR-10 官方 test split
samples：10,000
transform：ToTensor + Normalize
batch size：沿用各实验有效配置中的 128
loss：CrossEntropyLoss
metric：sample-weighted mean loss + top-1 accuracy
device：配置中的 auto
checkpoint：对应实验 best.pt
```

测试集不使用 RandomCrop 或 RandomHorizontalFlip，确保每次评估输入完全一致。评估过程调用 `model.eval()` 并关闭梯度，不修改模型参数、optimizer 或 checkpoint。

## 3. 评估产物设计

每个实验的评估结果写入其独立输出目录：

```text
outputs/<experiment>/evaluation/
├── metrics.json
└── predictions.csv
```

### 3.1 `metrics.json`

至少记录：

```text
experiment_name
checkpoint_path
checkpoint_epoch
checkpoint_sha256
test_loss
test_accuracy
num_samples
num_classes
```

### 3.2 `predictions.csv`

每个测试样本一行：

```text
sample_index
true_label
true_class
predicted_label
predicted_class
confidence
correct
```

`confidence` 是预测类别的 Softmax 概率。模型仍输出 logits；Softmax 只在评估后用于解释预测置信度，不送回 CrossEntropyLoss。

这些逐样本结果将作为混淆矩阵和错误样本图的唯一数据源，避免不同图表重复推理后产生不一致。

### 3.3 `evaluate_with_predictions()`

`src/engine.py` 新增通用评估函数，输入 model、loader、criterion 和 device，返回：

```text
aggregate metrics：loss、accuracy、num_samples
prediction records：sample_index、true_label、predicted_label、confidence、correct
```

函数使用 `model.eval()` 与 `@torch.no_grad()`，不会创建梯度或修改模型参数。CrossEntropyLoss 直接接收 logits；Softmax 只在 loss 计算之后用于取得预测类别置信度。

为避免对 10,000 个样本逐个触发 GPU→CPU 同步，labels、predictions 和 confidences 每个 batch 各整体传输一次，再在 CPU 端生成记录。`sample_index` 按 `shuffle=False` 的 DataLoader 顺序递增。

2026-09-23 验收结果：训练引擎定向测试 `24 passed`。测试覆盖样本加权 loss/accuracy、逐样本顺序、Softmax 置信度、正确标记、eval/no-grad 状态、参数不变性，以及 batch 上限不预取额外数据。

### 3.4 `scripts/evaluate.py`

评估入口直接读取 `best.pt` 内嵌的有效配置，重建数据与模型，不依赖可能后来被修改的 YAML。脚本主动拒绝其他文件名（如 `last.pt`），并将 checkpoint epoch、SHA-256、测试指标和逐样本预测写入独立目录。

正式评估前用 2 个 batch 做隔离冒烟测试：256 条记录完整、索引为 0–255，MiniViT checkpoint SHA-256 与冻结清单一致。正式运行后，7 组 `metrics.json` 均标记为非 partial，且每个 `predictions.csv` 恰有 10,000 行。

## 4. 四类必做可视化

### 4.1 训练曲线

从各实验 `history.csv` 绘制：

- train loss 与 val loss；
- train accuracy 与 val accuracy；
- 标记 best epoch。

曲线用于解释收敛、平台期和过拟合，不使用测试数据。

### 4.2 混淆矩阵

根据 `predictions.csv` 构造 `10×10` 计数矩阵：

```text
纵轴：真实类别
横轴：预测类别
```

主要展示 MiniViT baseline 与 CNN baseline，分析两者最常见的类别混淆。

### 4.3 错误样本

从测试预测中选取若干代表性错误案例，展示：

```text
Ground Truth
Prediction
Confidence
```

人工归纳主体过小、背景干扰、图像模糊、视角特殊或类别外观相似等原因。样本选择应服务于错误类型分析，不能只挑最有趣的图片。

### 4.4 Attention 热力图

对 MiniViT baseline 的正确与错误测试样本：

1. 获取最后一个 Encoder Block 的 Attention；
2. 取 CLS token 指向图像 patch 的权重；
3. 对多个 attention head 求平均；
4. reshape 为 `8×8` patch 网格；
5. 插值到 `32×32`；
6. 叠加到反归一化原图。

Attention 图只能描述模型在该层如何分配注意力权重，不能证明被高亮区域对预测具有因果作用。

## 5. 计划生成的图表

```text
assets/
├── training_curves/
│   ├── minivit_baseline.png
│   └── cnn_baseline.png
├── confusion_matrices/
│   ├── minivit_baseline.png
│   └── cnn_baseline.png
├── error_cases/
│   ├── minivit_baseline.png
│   └── cnn_baseline.png
└── attention/
    └── minivit_attention_examples.png
```

消融实验的完整数值保留在结果表中；训练曲线可按分析需要补充对比图，避免一次生成过多重复图片。

## 6. 验收标准

- 所有测试评估明确加载冻结的 `best.pt`；
- 7 个实验均有可追溯的 test loss、test accuracy 和逐样本预测；
- 最终表包含 MiniViT、CNN 和五个消融结果；
- 训练曲线坐标、单位、图例和 best epoch 标记清楚；
- 混淆矩阵明确标注真实轴与预测轴；
- 错误案例包含真实类别、预测类别和置信度，并有文字归纳；
- Attention 图同时包含正确与错误样本，并注明解释边界；
- 测试结果没有用于继续调参或覆盖训练 checkpoint。

## 7. 最终测试结果

| 模型 / 消融 | Best Epoch | Test Loss | Test Acc |
|---|---:|---:|---:|
| SimpleCNN baseline | 74 | **0.4514** | **88.29%** |
| MiniViT baseline（p4 / d6 / aug） | 80 | 0.7335 | 80.65% |
| MiniViT depth 9 | 96 | 0.8347 | 80.36% |
| MiniViT depth 3 | 95 | 0.7347 | 80.27% |
| MiniViT patch 2 | 51 | 0.7653 | 78.94% |
| MiniViT patch 8 | 76 | 0.8077 | 76.38% |
| MiniViT no augmentation | 94 | 1.5201 | 70.93% |

这些结果没有用于重新选择 epoch、修改配置或继续训练。验证集上的三项主要结论全部在测试集保持：CNN 优于 MiniViT、patch 4 优于 patch 2/8、基础数据增强非常重要。Depth 3、6、9 的差距较小；单 seed 只能支持“当前协议未观察到加深收益”，不能证明普遍的最优深度。

## 8. 可视化实现与读图方法

### 8.1 训练曲线

`plot_training_history()` 把 loss 与 accuracy 分为两个面板，训练/验证曲线使用相同 epoch 横轴，并以红色虚线标记由验证准确率选出的 best epoch。它回答的是“怎样收敛、何时开始过拟合”，不包含测试数据。

### 8.2 混淆矩阵

`build_confusion_matrix()` 固定“行 = 真实类别、列 = 预测类别”。绘图时按每个真实类别的样本数做行归一化，因此同一行表示该真实类被分到各预测类的比例；单元格同时保留百分比和原始计数。

MiniViT 最难识别 `cat`（58.4%），最大混淆为 `cat → dog`（205）；CNN 的 `cat` 准确率为 76.4%，`cat → dog` 降为 121。CNN 测试准确率相对 MiniViT 高 7.64 pp，优势分布在多数类别，而不是只来自某一个类别。

### 8.3 错误案例

`select_error_cases()` 先按置信度排序，再优先覆盖不同的 `(true, predicted)` 混淆对，最后才用剩余高置信错误补足。这样既能观察模型“很自信却判断错”的案例，又避免整张图都被同一种错误占满。图片通过 `sample_index` 从确定性 test dataset 回取，与 `predictions.csv` 一一对应。

### 8.4 Attention 热力图

MiniViT 只对挑选出的样本额外前向一次：读取最后一个 Encoder Block 的 `[B, H, N, N]` 注意力，取 `attention[:, :, 0, 1:]`，即 CLS query 指向所有 patch key 的权重；对 head 求平均后恢复为 `8×8`，再双线性插值到 `32×32` 并逐图缩放到 `[0, 1]`。

图中同时包含正确和错误预测。热区说明最后一层 CLS 在这些 patch 上分配了较高注意力，但 Attention 权重不是梯度归因或干预实验，不能据此断言“模型因为该区域才作出预测”。

## 9. 最终产物与测试

```text
assets/
├── training_curves/
│   ├── minivit_baseline.png
│   └── cnn_baseline.png
├── confusion_matrices/
│   ├── minivit_baseline.png
│   └── cnn_baseline.png
├── error_cases/
│   ├── minivit_baseline.png
│   └── cnn_baseline.png
└── attention/
    └── minivit_attention_examples.png
```

最终全量自动化测试：`184 passed`。新增测试覆盖评估 CLI、best checkpoint 约束、正式/partial 结果隔离、逐样本文件格式、混淆矩阵方向、错误案例多样性、Attention 网格还原与归一化，以及绘图批次契约。

7 张 PNG 均通过文件、分辨率与非空像素检查。Codex 内置图片查看器曾因 Windows sandbox 初始化错误无法打开工作区图片；随后由用户直接查看 `assets/` 中的正式图片，并确认视觉呈现没有问题。至此，程序化检查与人工视觉验收均已完成。
