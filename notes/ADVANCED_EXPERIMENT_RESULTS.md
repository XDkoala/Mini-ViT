# Mini Vision Transformer 进阶实验结果

本文是阶段 11–18 正式实验的统一结果总表。它只记录能够从结构化产物复核的结论，不从终端输出手工猜测指标。

创建日期：2026-09-25

当前状态：阶段 10 基础设施已完成，尚无进阶正式实验

## 1. 记录规则

正式结果必须同时满足：

1. `run_status.json` 的状态为 `completed`；
2. 不是 smoke 或受限 batch 运行；
3. Git commit 可追溯且工作区为 clean；
4. config、完整 history、best.pt 和 last.pt 均存在；
5. epoch 序列完整；
6. 模型选择只使用 validation，不使用 test 调参；
7. 通过 `scripts/aggregate.py` 的重新核验。

测试集结果仅在策略冻结后记录，不参与训练策略选择。

## 2. 基础项目锚点

以下结果来自阶段 0–9，仅作为进阶实验的固定参照，不属于本文件的进阶聚合结果：

| 模型 | Best Validation Accuracy | Test Accuracy |
|---|---:|---:|
| MiniViT baseline | 81.84% | 80.65% |
| CNN baseline | 89.58% | 88.29% |

## 3. 进阶正式实验总表

当前暂无合格的进阶正式实验。阶段 11 开始后，由聚合结果更新下表。

| 阶段 | Experiment ID | Split Seed | Train Seed | Best Epoch | Best Val Acc | Final Val Acc | Test Acc | Git Commit | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|

## 4. 阶段 10 基础设施验证

以下运行仅用于验证协议，不进入正式结果总表：

| 运行 | 范围 | 状态 | 聚合结果 | 原因 |
|---|---|---|---|---|
| stage10 single-batch smoke | 1 epoch，每轮 1 train/val batch | completed | excluded | smoke test |
| stage10 protocol smoke | 2 epochs，每轮 2 train/val batches | completed | excluded | smoke test |

两次运行均成功生成有效配置、完整 history、best.pt、last.pt 和 run_status.json。排除是实验协议的正确行为，不是训练失败。

## 5. 后续更新要求

每组正式实验结束后：

1. 运行 `python -m scripts.aggregate`；
2. 检查 `summary_skipped.csv` 中是否有意外排除；
3. 从 `summary.csv` 更新正式实验总表；
4. 多 seed 实验同时报告均值、标准差和单次结果；
5. 写明实验前猜想、实际结果、偏差原因与下一步判断；
6. 若最终使用 test set，记录 checkpoint 哈希和评估产物位置。
