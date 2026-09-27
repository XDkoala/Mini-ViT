# Mini Vision Transformer 进阶实验结果

本文是阶段 11–18 正式实验的统一结果总表。它只记录能够从结构化产物复核的结论，不从终端输出手工猜测指标。

创建日期：2026-09-25

当前状态：阶段 11–14 已完成；advanced MiniViT 配方已冻结并完成效率基准

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
| 11 | stage11_amp_efficiency_fp32_split42_train42 | 42 | 42 | 91 | 83.00% | 82.98% | 未使用 | 842555c | AMP 效率对照的 FP32 组 |
| 11 | stage11_amp_efficiency_cuda_split42_train42 | 42 | 42 | 93 | 83.64% | 82.76% | 未使用 | 842555c | AMP 约 1.455× 加速，allocated 显存减少 26.47% |
| 12 | stage12_r1_label_smoothing_005_split42_train42 | 42 | 42 | 95 | 83.52% | 83.02% | 未使用 | e3693f0 | Label Smoothing 0.05；与 R0 接近 |
| 12 | stage12_r2_mixup_alpha02_split42_train42 | 42 | 42 | 96 | 84.56% | 83.72% | 未使用 | 6ba1005 | Mixup 0.2；相对 R0 提升 0.92 pp |
| 12 | stage12_r3_cutmix_alpha05_split42_train42 | 42 | 42 | 98 | 86.92% | 86.64% | 未使用 | 5ae1a9f | CutMix 0.5；阶段 12 最佳 |
| 13 | stage13_r0_o3_amp_split42_train123 | 42 | 123 | 95 | 84.06% | 83.48% | 未使用 | 2ef26ee | R0 多 seed 确认 |
| 13 | stage13_r3_cutmix05_split42_train123 | 42 | 123 | 98 | 86.42% | 86.02% | 未使用 | 2ef26ee | CutMix 多 seed 确认 |
| 13 | stage13_r0_o3_amp_split42_train2026 | 42 | 2026 | 80 | 83.94% | 83.70% | 未使用 | 2ef26ee | R0 多 seed 确认 |
| 13 | stage13_r3_cutmix05_split42_train2026 | 42 | 2026 | 94 | 87.14% | 86.96% | 未使用 | 2ef26ee | CutMix 多 seed 确认 |

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

后续配方选择只使用 validation、复杂度和稳定性证据，不使用 test。

## 6. 阶段 11：AMP 效率对照

FP32 与 AMP 两组均完成 100 epochs，来自同一 clean commit `842555c88762d6c8c35e66376e45f8326e102955`。除 AMP 开关和输出标识外，其余训练配置一致。计时覆盖完整训练与验证计算，并在计时边界同步 CUDA；不包含日志和 checkpoint 磁盘写入。

| 指标 | FP32 | AMP | AMP 相对变化 |
|---|---:|---:|---:|
| Epoch 6–100 时间中位数 | 20.603 s | 14.156 s | -31.29% |
| 100 轮同步总时间 | 2047.24 s | 1416.57 s | -30.81% |
| Peak allocated | 860.83 MiB | 632.95 MiB | -26.47% |
| Peak reserved | 962 MiB | 780 MiB | -18.92% |
| Best Val Acc | 83.00% | 83.64% | +0.64 pp |
| Final Val Acc | 82.98% | 82.76% | -0.22 pp |

AMP 按主要时间指标获得约 `1.455×` 加速。两组全部数值有限并完成 100 轮；AMP 的 GradScaler 跳过 14 / 35200 次潜在溢出更新（约 0.040%），scheduler 只随 35186 次有效 optimizer update 前进，行为符合协议。

单 seed 下 AMP 的 best accuracy 更高、final accuracy 略低，不能将其解释为稳定的精度收益或损失。本实验支持的结论是：在当前硬件和配方上，AMP 明显提高速度并降低峰值显存，同时保持训练数值稳定。

阶段 11 聚合结果为 `eligible=6, skipped=5`；5 个 skipped 全部是预期 smoke test。

## 7. 阶段 12：小数据正则化

在统一的 O3+AMP 底座上，单 seed 正式结果为：R1 Label Smoothing 83.52%、R2 Mixup 84.56%、R3 CutMix 86.92%。R3 相对无高级正则化的 R0 提升 3.28 pp，是进入阶段 13 的唯一高级正则化候选。

30-epoch screening 只用于各方法内部选择强度；其跨方法排序与 100-epoch 结果相反，不能把短跑绝对值用作不同正则化机制的最终排名。

## 8. 阶段 13：多随机种子确认与冻结

| 配方 | Seed 42 | Seed 123 | Seed 2026 | Mean ± Sample Std |
|---|---:|---:|---:|---:|
| R0 | 83.64% | 84.06% | 83.94% | 83.880 ± 0.216% |
| R3 CutMix | 86.92% | 86.42% | 87.14% | **86.827 ± 0.369%** |
| R3 - R0 | +3.28 pp | +2.36 pp | +3.20 pp | **+2.947 ± 0.510 pp** |

三个 seed 的配对提升方向一致。基于 validation 均值与一致性，冻结 R3 CutMix 0.5 为 `advanced_minivit_final`；完整配方、checkpoint 哈希与评估约束见 [ADVANCED_MINIVIT_FINAL.md](ADVANCED_MINIVIT_FINAL.md)。冻结过程未使用新的 test 结果。

## 9. 阶段 14：效率与资源基准

正式 benchmark 使用 RTX 5060 Laptop GPU、PyTorch 2.14.0+cu130、CUDA 13.0，输入为 CIFAR-10 `[B,3,32,32]`。batch-1 latency 使用 50 次 warmup 与 200 次测量；batch-128 throughput 使用 20 次 warmup 与 100 次测量。每次计时均在 CUDA synchronize 之间完成，不包含 DataLoader。

| 模型 | Best Val | Params | MACs | FP32 latency p50 | AMP throughput | AMP train peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| CNN baseline | **89.58%** | 2.609 M | 115.906 M | **0.414 ms** | **54,757 img/s** | **254.42 MiB** |
| MiniViT baseline | 81.84% | 2.694 M | 182.850 M | 1.692 ms | 20,274 img/s | 632.59 MiB |
| advanced MiniViT | 86.92% | 2.694 M | 182.850 M | 1.644 ms | 20,456 img/s | 632.59 MiB |

advanced MiniViT 在不增加推理结构成本的情况下比基础 MiniViT 提升 5.08 pp，但 CNN 在当前小图像任务上仍同时具有更高准确率和更低成本。AMP 提高 batch-128 吞吐量并降低训练显存，却因固定调度开销使 batch-1 latency 变慢。完整协议、FP32/AMP p50/p95、推理/训练显存和模型文件拆分见 [STAGE_14_NOTES.md](STAGE_14_NOTES.md)。

## 10. 后续更新要求

每组正式实验结束后：

1. 运行 `python -m scripts.aggregate`；
2. 检查 `summary_skipped.csv` 中是否有意外排除；
3. 从 `summary.csv` 更新正式实验总表；
4. 多 seed 实验同时报告均值、标准差和单次结果；
5. 写明实验前猜想、实际结果、偏差原因与下一步判断；
6. 若最终使用 test set，记录 checkpoint 哈希和评估产物位置。
