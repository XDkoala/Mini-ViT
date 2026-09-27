# 阶段 13：多随机种子确认与统计分析

本阶段不再搜索新方法，而是确认阶段 12 中 CutMix 的收益能否跨训练随机种子复现，并在使用 test 集之前冻结后续阶段采用的 advanced MiniViT 配方。

开始日期：2026-09-27

当前状态：进行中

当前位置：实验矩阵与统计规则已冻结；下一步补跑 seed 123、2026

## 进度

- [x] 确定进入多 seed 确认的候选
- [x] 冻结 split seed、train seeds 与训练预算
- [x] 创建四份新增训练配置
- [ ] 完成 seed 123 的 R0/R3
- [ ] 完成 seed 2026 的 R0/R3
- [ ] 聚合原始值、mean 与 sample std
- [ ] 计算 CutMix 相对 R0 的逐 seed 差值
- [ ] 冻结 advanced MiniViT 最终配方与 checkpoint 清单
- [ ] 阶段 13 总验收

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

本阶段仍只使用 train/validation，不读取 CIFAR-10 test 结果进行选择。完成统计后，根据 validation mean、波动、计算成本和实现复杂度冻结 `advanced_minivit_final` 的完整配置、三个 seed 的 best checkpoint 与评估规则；阶段 18 才进行统一 clean test。
