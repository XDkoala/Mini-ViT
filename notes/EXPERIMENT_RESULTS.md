# Mini ViT 实验结果总账

本文件记录可用于后续分析和实验报告的固定实验事实。原始逐 epoch 数据、配置与 checkpoint 保存在对应的 `outputs/` 目录；本文件只汇总关键结果，不用人工估计值替换真实日志。

## 记录规则

- 每个实验使用独立 `output_dir`，禁止覆盖其他实验；
- 同时保留有效 `config.yaml`、完整 `history.csv`、`best.pt` 与 `last.pt`；
- 模型选择只依据验证集，测试集只在最终模型确定后评估；
- 表格中的结果必须能追溯到原始文件；
- 若训练协议或参数量不同，必须明确说明，不能称为严格单变量实验。

## 实验总表

| 实验 | 参数量 | Epochs | 最佳 Val Acc | 最佳 Epoch | 最终 Train Acc | 最终 Val Acc | Test Acc | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MiniViT baseline | 2,693,578 | 100 | **81.84%** | 80 | 95.20% | 81.36% | **80.65%** | 已完成 |
| SimpleCNN baseline | 2,609,002 | 100 | **89.58%** | 74 | 98.00% | 87.90% | **88.29%** | 已完成 |
| MiniViT depth 3 | 1,358,986 | 100 | **81.16%** | 95 | 91.88% | 80.54% | **80.27%** | 已完成 |
| MiniViT patch 8 | 2,712,010 | 100 | **77.30%** | 76 | 92.14% | 76.76% | **76.38%** | 已完成 |
| MiniViT no augmentation | 2,693,578 | 100 | **72.64%** | 94 | 97.94% | 70.86% | **70.93%** | 已完成 |
| MiniViT depth 9 | 4,028,170 | 100 | **81.46%** | 96 | 96.22% | 81.44% | **80.36%** | 已完成 |
| MiniViT patch 2 | 2,723,530 | 100 | **80.02%** | 51 | 96.21% | 78.96% | **78.94%** | 已完成 |

## 阶段 8：冻结模型的最终测试结果

7 个 `best.pt` 均在读取测试结果前由验证准确率确定。每个 checkpoint 只执行一次完整 CIFAR-10 test split 评估（10,000 张），使用相同的确定性预处理、CrossEntropyLoss 与 top-1 accuracy。逐样本预测和 checkpoint SHA-256 保存在对应实验的 `evaluation/` 目录。

| 实验 | Test Loss | Test Acc | Best Val Acc | Test - Best Val |
|---|---:|---:|---:|---:|
| SimpleCNN baseline | **0.4514** | **88.29%** | 89.58% | -1.29 pp |
| MiniViT baseline / patch 4 / depth 6 | 0.7335 | 80.65% | 81.84% | -1.19 pp |
| MiniViT depth 9 | 0.8347 | 80.36% | 81.46% | -1.10 pp |
| MiniViT depth 3 | 0.7347 | 80.27% | 81.16% | -0.89 pp |
| MiniViT patch 2 | 0.7653 | 78.94% | 80.02% | -1.08 pp |
| MiniViT patch 8 | 0.8077 | 76.38% | 77.30% | -0.92 pp |
| MiniViT no augmentation | 1.5201 | 70.93% | 72.64% | -1.71 pp |

测试集延续了验证集的主要结论：CNN 明显优于 MiniViT；patch 排序仍为 `p4 > p2 > p8`；depth 6 最好，而 depth 3 与 depth 9 非常接近；移除数据增强造成最大退化。各实验测试准确率均低于其最佳验证准确率 0.89–1.71 个百分点，没有出现足以推翻验证结论的异常反转。

### Baseline 类别错误分析

MiniViT 各类测试准确率中，`cat` 最低（58.4%），最大单向混淆是 `cat → dog`（205 张），其次为 `dog → cat`（103 张）。CNN 的 `cat` 仍是最难类别，但准确率提高到 76.4%，对应 `cat → dog` 降至 121 张、`dog → cat` 降至 69 张。两种模型都容易混淆外观接近的动物类别以及 `automobile ↔ truck`；CNN 的卷积归纳偏置并未消除这些难点，但显著降低了错误数量。

训练曲线、归一化混淆矩阵、高置信错误案例和 MiniViT 最后一层 CLS Attention 图保存在 `assets/`。Attention 只用于描述权重分配，不能作为某一区域对预测具有因果作用的证明。

## EXP-001：MiniViT Baseline

### 实验身份

```text
实验编号：EXP-001
实验名称：minivit_baseline
完成日期：2026-09-22
随机种子：42
数据集：CIFAR-10
训练/验证划分：45,000 / 5,000
测试集：本实验阶段未使用
输出目录：outputs/minivit_baseline/
```

### 模型与训练协议

```text
参数量：2,693,578
patch size：4
embed dim：192
depth：6
attention heads：6
MLP dim：768
dropout：0.1
batch size：128
optimizer：AdamW
learning rate：0.0003（固定，无 scheduler）
weight decay：0.05
epochs：100
训练增强：RandomCrop(32, padding=4) + RandomHorizontalFlip
```

### 核心结果

| 指标 | 结果 |
|---|---:|
| 最佳验证准确率 | **81.84%（epoch 80）** |
| 最佳轮训练准确率 | 94.08% |
| 最佳轮验证 loss | 0.7255 |
| 最低验证 loss | 0.5879（epoch 40） |
| 最低 loss 对应验证准确率 | 80.30% |
| 最终训练准确率 | 95.20% |
| 最终验证准确率 | 81.36% |
| 最终验证 loss | 0.7793 |
| 总计算耗时 | 2,563.73 s（约 42 min 44 s） |
| 平均每轮耗时 | 25.64 s |

### 初步分析

模型成功学习且明显高于十分类随机猜测，但约 epoch 40 后开始出现明显泛化平台：训练准确率继续提高，验证准确率只小幅波动，验证 loss 持续上升。该现象符合过拟合及错误样本置信度升高，而不是训练链路失效。

当前结果应保留为首版训练协议下的原始 ViT 基线。后续先与相近参数量 CNN 在同一协议下比较，再通过消融实验判断数据增强、结构或优化策略的影响。

### 产物与完整性

```text
best.pt     SHA-256 250E2CC425C7AA080DA37BE036C734D31548ACC6D563D7DAF4B6FF72AA6726A4
last.pt     SHA-256 DB264FF20AFD289C7D34548F1D293EB0299B019A0838EB016342C8C444D4021C
history.csv SHA-256 6B39CEF579FD870C5DEFA4B7DBA20C867CD7B7A1675E7EEB36A318C1A0B7757D
config.yaml SHA-256 EA2CF20ED784D6DB7224B99A6491F2738B46278EE41F46500577E9036879C0F7
```

阶段 8 最终测试必须加载 `outputs/minivit_baseline/best.pt`，不能使用 `last.pt`，也不能根据测试集结果重新挑选 epoch。

## EXP-002：SimpleCNN Baseline

### 实验身份

```text
实验编号：EXP-002
实验名称：cnn_baseline
完成日期：2026-09-22
随机种子：42
数据集：CIFAR-10
训练/验证划分：45,000 / 5,000
测试集：本实验阶段未使用
输出目录：outputs/cnn_baseline/
```

### 模型与训练协议

```text
参数量：2,609,002
channels：96 → 192 → 384 → 512
卷积块：Conv2d → BatchNorm2d → ReLU → MaxPool2d
dropout：0.1
batch size：128
optimizer：AdamW
learning rate：0.0003（固定，无 scheduler）
weight decay：0.05
epochs：100
训练增强：RandomCrop(32, padding=4) + RandomHorizontalFlip
```

### 核心结果

| 指标 | 结果 |
|---|---:|
| 最佳验证准确率 | **89.58%（epoch 74）** |
| 最佳轮训练准确率 | 97.38% |
| 最佳轮验证 loss | 0.4123 |
| 最低验证 loss | 0.3956（epoch 39） |
| 最低 loss 对应验证准确率 | 88.16% |
| 最终训练准确率 | 98.00% |
| 最终验证准确率 | 87.90% |
| 最终验证 loss | 0.4988 |
| 总计算耗时 | 1,074.90 s（约 17 min 55 s） |
| 平均每轮耗时 | 10.75 s |

### 与 MiniViT 的比较结论

CNN 参数量比 MiniViT 少 84,576（约 3.14%），最佳验证准确率高 7.74 个百分点，平均每轮约快 2.38 倍。在当前 CIFAR-10 小数据训练协议下，CNN 的图像归纳偏置带来了更好的泛化效率和计算效率。

CNN 后期也存在训练准确率继续提高、验证 loss 上升的过拟合，但其验证准确率平台约为 89%，明显高于 MiniViT 的约 81%。这说明差距不能简单归因于两者都过拟合；架构先验是当前证据支持的重要解释。

### 产物与完整性

```text
best.pt     SHA-256 C20802B84993AE29C0B789AB600D2FCDC33FAD7A34C61F269273EF25D0A8E95A
last.pt     SHA-256 450E627F66650A6E24B899B7FC28257F80C6E240C8A2500DA7F3074F8C2F53D7
history.csv SHA-256 5382F7FED6687A958AF7CE1C01C3C800444D83CB44D45099DDCB1A84975BACC2
config.yaml SHA-256 47738D3E16155C945D9203CEA277850FDCB001ECEA7DF2703790F071B95ACFD6
```

阶段 8 最终测试必须加载 `outputs/cnn_baseline/best.pt`，并与 MiniViT 使用完全相同的测试评估代码。

## EXP-005：MiniViT Depth 3

### 实验身份与唯一变量

```text
实验名称：ablation_depth_p4_d3_aug_seed42
输出目录：outputs/ablation_depth_p4_d3_aug_seed42/
唯一变量：depth 6 → 3
参数量：2,693,578 → 1,358,986
其余数据、模型与训练设置：与 EXP-001 完全相同
```

### 核心结果

| 指标 | Depth 3 | Depth 6 Baseline | 差异 |
|---|---:|---:|---:|
| 最佳验证准确率 | **81.16%** | 81.84% | -0.68 pp |
| 最佳 epoch | 95 | 80 | +15 |
| 最终训练准确率 | 91.88% | 95.20% | -3.32 pp |
| 最终验证准确率 | 80.54% | 81.36% | -0.82 pp |
| 最终泛化差距 | 11.34 pp | 13.84 pp | -2.50 pp |
| 平均 epoch 时间 | **12.37 s** | 25.64 s | 约快 2.07 倍 |
| 总计算耗时 | 20 min 37 s | 42 min 44 s | 少约 22 min 7 s |

Depth 3 以约一半参数和计算时间获得了接近 baseline 的验证准确率。训练准确率较低说明容量与拟合能力下降；泛化差距同时缩小，说明过拟合有所减轻。当前只能得出“从 6 层减到 3 层性价比很高但上限略降”，完整深度趋势需结合 EXP-006 depth 9。

### 产物完整性

```text
best.pt     SHA-256 2279E7CF1C4D37B32C287B738E1580048F5D10F4B7288F8A2452A5F52580175D
last.pt     SHA-256 6DCFA971DFE5618971ED8F52A162CF8B194E4A8DD2BA814BE3D8B1A7C2047AE2
history.csv SHA-256 CE0F60A0A9E0B151CE9D59B14364AC01B587D2E6D47F158E91394A385C886A5C
config.yaml SHA-256 1A0447591992DC8FD12D53B08A4A9F8D054128344536AE66FA040C789B6466DE
```

## EXP-004：MiniViT Patch Size 8

### 实验身份与唯一变量

```text
实验名称：ablation_patch_p8_d6_aug_seed42
输出目录：outputs/ablation_patch_p8_d6_aug_seed42/
唯一变量：patch_size 4 → 8
图像 token：64 → 16
参数量：2,693,578 → 2,712,010
其余数据、模型与训练设置：与 EXP-001 完全相同
```

### 核心结果

| 指标 | Patch 8 | Patch 4 Baseline | 差异 |
|---|---:|---:|---:|
| 最佳验证准确率 | **77.30%** | 81.84% | -4.54 pp |
| 最佳 epoch | 76 | 80 | -4 |
| 最终训练准确率 | 92.14% | 95.20% | -3.05 pp |
| 最终验证准确率 | 76.76% | 81.36% | -4.60 pp |
| 最终泛化差距 | 15.38 pp | 13.84 pp | +1.54 pp |
| 平均 epoch 时间 | **10.03 s** | 25.64 s | 约快 2.56 倍 |
| 总计算耗时 | 16 min 43 s | 42 min 44 s | 少约 26 min 1 s |

Patch 8 显著降低 token 数和计算成本，但对 32×32 CIFAR-10 的空间压缩过强，训练拟合能力与验证性能均下降。结果支持实验前猜想中的 `p4 > p8` 和速度 `p8 > p4`；patch 2 尚未完成，因此不能提前宣布完整排序成立。

### 产物完整性

```text
best.pt     SHA-256 AB5D6A70D8DDDB93FB14FEBCB3D6FD45D10BB3B8E385503F0F88DD2C5C58F890
last.pt     SHA-256 F5B339F39A404B8620EAFCD3C36E000D20DF4144853137DC9859AADE751A73FC
history.csv SHA-256 A988F2FD07FD95A8D65818716493B3F76A360ED6F2D388871DE2F5011BE9D35D
config.yaml SHA-256 C32A6BE0E6E7B30F78B22EA320B93CCDDB97B7BAACC659F4FABDBB6A32D6AC29
```

## EXP-007：MiniViT No Augmentation

### 实验身份与唯一变量

```text
实验名称：ablation_aug_p4_d6_noaug_seed42
输出目录：outputs/ablation_aug_p4_d6_noaug_seed42/
唯一变量：augmentation basic → none
模型参数量：均为 2,693,578
其余数据、模型与训练设置：与 EXP-001 完全相同
```

### 核心结果

| 指标 | No Augmentation | Basic Augmentation | 差异 |
|---|---:|---:|---:|
| 最佳验证准确率 | **72.64%** | 81.84% | -9.20 pp |
| 最佳 epoch | 94 | 80 | +14 |
| 最终训练准确率 | 97.94% | 95.20% | +2.74 pp |
| 最终验证准确率 | 70.86% | 81.36% | -10.50 pp |
| 最终泛化差距 | 27.08 pp | 13.84 pp | +13.24 pp |
| 最低验证 loss | 0.9201 | 0.5879 | +0.3322 |
| 平均 epoch 时间 | 19.72 s | 25.64 s | 少 5.92 s |

结果完整支持实验前猜想：移除增强后，模型更容易记忆固定训练图片，但对未见验证图片的泛化显著恶化。基础 RandomCrop 与 RandomHorizontalFlip 对当前小数据 MiniViT 不是可有可无的装饰，而是关键正则化来源。

最低验证 loss 在 epoch 13，远早于最高验证 accuracy 的 epoch 94。后期模型对错误样本的置信度不断升高，导致交叉熵显著恶化，即使离散的预测正确数偶尔仍有改善。

### 产物完整性

```text
best.pt     SHA-256 3187321147E967993F1260D31C09081D59997F9F242E8B7211403B9B2BA51795
last.pt     SHA-256 5B8125CB97EEC7A41B9BB3B7B24B38D83446E2CC5581A2EEE15F66D4954184FD
history.csv SHA-256 68820F1A35660661D604CF10F724FE32557CE05A33F6F736EEC33B49F577CA99
config.yaml SHA-256 C8D3CE114C51EA47013AF525702B05E860566C6041CC6CB06522B4C3CB9769DF
```

## EXP-006：MiniViT Depth 9

### 实验身份与唯一变量

```text
实验名称：ablation_depth_p4_d9_aug_seed42
输出目录：outputs/ablation_depth_p4_d9_aug_seed42/
唯一变量：depth 6 → 9
参数量：2,693,578 → 4,028,170
其余数据、模型与训练设置：与 EXP-001 完全相同
```

### 核心结果

| 指标 | Depth 9 | Depth 6 Baseline | 差异 |
|---|---:|---:|---:|
| 最佳验证准确率 | **81.46%** | 81.84% | -0.38 pp |
| 最佳 epoch | 96 | 80 | +16 |
| 最终训练准确率 | 96.22% | 95.20% | +1.02 pp |
| 最终验证准确率 | 81.44% | 81.36% | +0.08 pp |
| 最终泛化差距 | 14.78 pp | 13.84 pp | +0.94 pp |
| 平均 epoch 时间 | 30.09 s | 25.64 s | 慢 4.45 s |
| 总计算耗时 | 50 min 9 s | 42 min 44 s | 多约 7 min 25 s |

实验前预测的验证排序未完全成立：实际为 `d6 > d9 > d3`。但 depth 9 训练准确率略升且泛化差距扩大，两项方向性预测成立。结合参数与时间，当前协议下 depth 6 是三者中更合理的准确率/成本平衡点，depth 3 则拥有突出的效率。

由于每个配置只运行 seed 42，d6 与 d9 的 0.38 个百分点差异可能包含随机波动；结论应表述为“未观察到增加深度的收益”，而非“证明九层一定更差”。

### 产物完整性

```text
best.pt     SHA-256 114A8711B267689C0D6397D83D0704A930467DDA42CB8D5727AA7C16704191B5
last.pt     SHA-256 02C60F4C8102093A24EEC4466EE21AEF604741BA9126FC8609D2DB41854A51DE
history.csv SHA-256 AEBFD5AD3FBF3B7CCB8185F60478065BD52905408024B759B57C573E21007D6D
config.yaml SHA-256 219846C19E9E13E2A404C2D5CDD348D925582AA75CC0B8D320A79EFB8544FCA9
```

## EXP-003：MiniViT Patch Size 2

### 实验身份与唯一变量

```text
实验名称：ablation_patch_p2_d6_aug_seed42
输出目录：outputs/ablation_patch_p2_d6_aug_seed42/
唯一变量：patch_size 4 → 2
图像 token：64 → 256
参数量：2,693,578 → 2,723,530
其余数据、模型与训练设置：与 EXP-001 完全相同
```

### 核心结果

| 指标 | Patch 2 | Patch 4 Baseline | 差异 |
|---|---:|---:|---:|
| 最佳验证准确率 | **80.02%** | 81.84% | -1.82 pp |
| 最佳 epoch | 51 | 80 | -29 |
| 最终训练准确率 | 96.21% | 95.20% | +1.01 pp |
| 最终验证准确率 | 78.96% | 81.36% | -2.40 pp |
| 最终泛化差距 | 17.25 pp | 13.84 pp | +3.41 pp |
| 正常平均 epoch 时间 | 97.72 s | 25.64 s | 约慢 3.81 倍 |
| 最低验证 loss | 0.6579 | 0.5879 | +0.0700 |

实际 Patch Size 排序为 `p4 > p2 > p8`，速度排序为 `p8 > p4 > p2`。细粒度信息使 p2 优于 p8，但增加到 256 个图像 token 后，额外优化与过拟合成本超过其相对 p4 的细节收益。

epoch 59 因系统睡眠记录 31,252.01 秒。速度统计排除此异常值；模型指标与 checkpoint 不受影响，原始 history 保留以保证可追溯性。

### 产物完整性

```text
best.pt     SHA-256 5964446C5287478A83E9DC9CBE8D4847FE6BB7D5CD623EC2008D772A00C3E008
last.pt     SHA-256 CD0305A4810C6A30609F7DA47356B5DD543445011FE92CA9C5D782A0A0B18835
history.csv SHA-256 4E6CAF2CA6095BB4F26405FEE04C492AB8B02E846818EE27A002A4B9403F5A41
config.yaml SHA-256 088CF590C764A3C549A36D466EC5F4EDDA9A2D4037401894452D7147C3EAA251
```
