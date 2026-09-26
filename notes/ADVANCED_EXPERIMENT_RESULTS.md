# Mini Vision Transformer 进阶实验结果

本文是阶段 11–18 正式实验的统一结果总表。它只记录能够从结构化产物复核的结论，不从终端输出手工猜测指标。

创建日期：2026-09-25

当前状态：阶段 11 O0–O3 优化消融已完成，AMP 效率对照待运行

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

| 阶段 | Experiment ID | Split Seed | Train Seed | Best Epoch | Best Val Acc | Final Val Acc | Test Acc | Git Commit | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 11 | stage11_o0_fixed_lr_split42_train42 | 42 | 42 | 80 | 81.84% | 81.36% | 未使用 | c58bc61 | 固定 LR 基线，与基础项目复现一致 |
| 11 | stage11_o1_cosine_split42_train42 | 42 | 42 | 79 | 82.94% | 82.88% | 未使用 | c58bc61 | cosine 相对 O0 提升 1.10 pp |
| 11 | stage11_o2_warmup_cosine_split42_train42 | 42 | 42 | 100 | 82.92% | 82.92% | 未使用 | c58bc61 | 与 O1 接近；warmup 使早期梯度更平稳 |
| 11 | stage11_o3_warmup_cosine_clip_split42_train42 | 42 | 42 | 91 | 83.00% | 82.98% | 未使用 | c58bc61 | 数值最高，但只比 O2 高 0.08 pp；裁剪几乎持续触发 |

## 4. 阶段 10 基础设施验证

以下运行仅用于验证协议，不进入正式结果总表：

| 运行 | 范围 | 状态 | 聚合结果 | 原因 |
|---|---|---|---|---|
| stage10 single-batch smoke | 1 epoch，每轮 1 train/val batch | completed | excluded | smoke test |
| stage10 protocol smoke | 2 epochs，每轮 2 train/val batches | completed | excluded | smoke test |

两次运行均成功生成有效配置、完整 history、best.pt、last.pt 和 run_status.json。排除是实验协议的正确行为，不是训练失败。

## 5. 阶段 11：O0–O3 优化消融

四组均完成 100 epochs，来自同一 clean commit `c58bc61317c4c2c3d6c7dc5bfc018b6f058b2ee2`，聚合结果为 `eligible=4, skipped=0`。模型、数据、seed、batch size、epoch 与 AdamW 基础设置完全一致；只累积改变 cosine、warmup 与 clipping。整个选择过程未读取 test split。

| Run | 策略 | Best Val Acc | Best Epoch | Final Train Acc | Final Val Acc | Mean Epoch Time |
|---|---|---:|---:|---:|---:|---:|
| O0 | fixed LR | 81.84% | 80 | 95.20% | 81.36% | 20.67 s |
| O1 | cosine | 82.94% | 79 | 98.20% | 82.88% | 21.02 s |
| O2 | 5-epoch warmup + cosine | 82.92% | 100 | 98.47% | 82.92% | 21.28 s |
| O3 | O2 + clip norm 1.0 | **83.00%** | 91 | 98.53% | **82.98%** | 21.54 s* |

`*` O3 的 epoch 98–100 出现明显系统耗时异常，median epoch time 为 20.98 s；不能把表中微小耗时差异解释为 clipping 的稳定成本。

主要结论：

1. O1 比 O0 提升 1.10 pp，支持在 100-epoch MiniViT 上使用 cosine decay；
2. O1 与 O2 只差 0.02 pp，当前单 seed 不支持 warmup 带来明确最终精度收益；
3. O2 前 5 轮 loss 下降更慢，但梯度范数标准差从 O1 的 0.343 降到 0.117，说明 warmup 改变早期更新轨迹并使梯度规模变化更平稳；
4. O3 比 O2 只高 0.08 pp，属于预定义的“接近”；
5. O3 平均 clip fraction 为 99.54%，100 个 epoch 全部不低于 90%，其中 78 轮为 100%。阈值 1.0 不是偶发保险丝，而是持续优化约束；
6. O1–O3 最大差距仅 0.08 pp，且目前只有一个 train seed，不能据此宣称三者存在稳定优劣。

当前数值排序：

~~~text
O3 83.00% > O1 82.94% > O2 82.92% > O0 81.84%
~~~

阶段 11 仍需使用同一 O3 配方完成 AMP 开/关效率对照。后续配方选择只使用 validation、复杂度和稳定性证据，不使用 test。

## 6. 后续更新要求

每组正式实验结束后：

1. 运行 `python -m scripts.aggregate`；
2. 检查 `summary_skipped.csv` 中是否有意外排除；
3. 从 `summary.csv` 更新正式实验总表；
4. 多 seed 实验同时报告均值、标准差和单次结果；
5. 写明实验前猜想、实际结果、偏差原因与下一步判断；
6. 若最终使用 test set，记录 checkpoint 哈希和评估产物位置。
