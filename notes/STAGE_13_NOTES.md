# 阶段 13：多随机种子确认与统计分析

本阶段不再搜索新方法，而是确认阶段 12 中 CutMix 的收益能否跨训练随机种子复现，并在使用 test 集之前冻结后续阶段采用的 advanced MiniViT 配方。

开始日期：2026-09-27

当前状态：已完成

当前位置：三 seed 统计与 advanced MiniViT 配方冻结均已完成

## 进度

- [x] 确定进入多 seed 确认的候选
- [x] 冻结 split seed、train seeds 与训练预算
- [x] 创建四份新增训练配置
- [x] 完成 seed 123 的 R0/R3
- [x] 完成 seed 2026 的 R0/R3
- [x] 聚合原始值、mean 与 sample std
- [x] 计算 CutMix 相对 R0 的逐 seed 差值
- [x] 冻结 advanced MiniViT 最终配方与 checkpoint 清单
- [x] 阶段 13 总验收

## 1. 为什么还要换 seed

固定 `train_seed` 会固定参数初始化、训练集 shuffle、随机增强以及 CutMix 的采样序列。单次实验只能说明某一条随机训练轨迹的结果，不能证明方法在一般随机条件下都有效。

三个 seed 只能提供初步稳定性证据，不足以进行强显著性推断。这里的目标是检查收益方向是否一致、波动是否可接受，而不是宣称得到精确的总体性能。

## 2. 冻结实验矩阵

所有运行固定 `split_seed=42`，只改变 `train_seed`：

| 配方 | seed 42 | seed 123 | seed 2026 |
|---|---|---|---|
| R0：O3 + AMP，无高级正则化 | 已完成 | 待运行 | 待运行 |
| R3：O3 + AMP + CutMix 0.5 | 已完成 | 待运行 | 待运行 |

R0 seed 42 复用阶段 11 clean run；R3 seed 42 复用阶段 12 clean run。R1 与 R2 不进入多 seed 阶段，避免把已经落选的候选继续扩张成无必要的实验矩阵。

除 `train_seed` 与输出目录外，同一配方内部的模型、数据划分、优化器、scheduler、clipping、AMP、batch size 和 100-epoch 预算保持不变。R0 与 R3 之间只有 CutMix 这一研究变量不同。

## 3. 配对比较

每个 seed 内计算：

~~~text
delta_seed = best_val(R3, seed) - best_val(R0, seed)
~~~

配对差值能在相同随机条件下比较两种配方，避免把 R3 的有利初始化与 R0 的不利初始化误认为方法收益。不能删除较差 seed，也不能看到均值后临时补跑直到结果变好。

## 4. 统计量

对三个 seed 的 best validation accuracy、best epoch 和训练时间报告全部原始值、均值与样本标准差：

~~~text
mean = sum(x_i) / n
sample_std = sqrt(sum((x_i - mean)^2) / (n - 1))
~~~

分母使用 `n-1`，因为三个观测被看作随机训练过程总体中的样本。还需报告三个逐 seed delta 的原始值和均值，不能只展示最好的单次结果。

## 5. 数据隔离与冻结规则

本阶段只使用 train/validation，不读取 CIFAR-10 test 结果进行选择。完成统计后，根据 validation mean、波动、计算成本和实现复杂度冻结 `advanced_minivit_final` 的完整配置、三个 seed 的 best checkpoint 与评估规则。后续实际安排为阶段 15 评估 seed 42 的 clean/corrupted test，阶段 18 复用并补齐另外两个 seed 的 clean test；所有 test 均不反向参与配方选择。

## 6. 三 seed 正式结果

六个正式运行均完成 100 epochs 且可追溯；阶段 13 新增四组均为 `git_is_dirty=false`、`aggregation_eligible=true`。结果如下：

| 配方 | Train Seed | Best Val Acc | Best Epoch | 时间 |
|---|---:|---:|---:|---:|
| R0 | 42 | 83.64% | 93 | 23.61 min |
| R0 | 123 | 84.06% | 95 | 26.55 min |
| R0 | 2026 | 83.94% | 80 | 27.40 min |
| R3 CutMix | 42 | 86.92% | 98 | 26.91 min |
| R3 CutMix | 123 | 86.42% | 98 | 27.25 min |
| R3 CutMix | 2026 | 87.14% | 94 | 36.61 min |

汇总使用样本标准差：

| 配方 | Best Val Acc | Best Epoch | 训练时间 |
|---|---:|---:|---:|
| R0 | 83.880 ± 0.216% | 89.333 ± 8.145 | 25.852 ± 1.988 min |
| R3 CutMix | **86.827 ± 0.369%** | 96.667 ± 2.309 | 30.255 ± 5.508 min |

CutMix seed 2026 的时间明显偏长，但模型、数据和预算没有变化，因此不能将该次系统耗时波动解释为 train seed 导致的固定计算成本。阶段 14 将在受控计时协议下单独测量效率。

## 7. 配对提升

| Train Seed | R3 - R0 |
|---:|---:|
| 42 | +3.28 pp |
| 123 | +2.36 pp |
| 2026 | +3.20 pp |
| mean ± sample std | **+2.947 ± 0.510 pp** |

三个差值方向全部为正，说明阶段 12 中 CutMix 的明显优势不是 seed 42 独有的偶然结果。三个 seed 仍不足以支持强显著性声明，但已提供一致的初步稳定性证据。

## 8. 最终冻结

阶段 13 根据 validation 冻结 `advanced_minivit_final`：MiniViT p4/d6、O3 优化栈、CUDA AMP、basic augmentation、CutMix alpha 0.5。模型参数量为 2,693,578。

三个 train seed 的 best checkpoint、SHA-256、来源 commit、评估脚本版本与 test 隔离规则记录在 [ADVANCED_MINIVIT_FINAL.md](ADVANCED_MINIVIT_FINAL.md)。冻结后不再依据 test 修改配方；阶段 14–17 使用该冻结配方。实际后续执行中，阶段 15 按冻结鲁棒性协议完成 seed 42 的 clean/corrupted test；阶段 18 复用该 clean 结果，只补齐 seeds 123、2026 的 clean test，再汇总三 seed。此补充说明执行时间线，不改变阶段 13 的原始选择依据。

聚合器最终报告 `eligible=13, skipped=30`。阶段 13 没有意外排除项。至此，多 seed 原始值、统计量、配对差值、运行溯源和最终候选冻结全部闭环。
