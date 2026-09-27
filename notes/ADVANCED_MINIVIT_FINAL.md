# advanced_minivit_final 冻结清单

冻结日期：2026-09-27

选择依据：阶段 13 三个 train seed 的 validation 结果。冻结时未读取新的 CIFAR-10 test 指标。

## 1. 完整配方

~~~yaml
randomness:
  split_seed: 42
  train_seeds: [42, 123, 2026]

data:
  dataset: CIFAR-10
  batch_size: 128
  augmentation: basic

model:
  name: minivit
  image_size: 32
  patch_size: 4
  embed_dim: 192
  depth: 6
  num_heads: 6
  mlp_dim: 768
  dropout: 0.1
  parameter_count: 2693578

training:
  epochs: 100
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.05
  scheduler: warmup_cosine
  warmup_epochs: 5
  start_factor: 0.1
  min_factor: 0.1
  gradient_clip_norm: 1.0
  amp: cuda_float16

regularization:
  label_smoothing: 0.0
  batch_mixing: cutmix
  cutmix_alpha: 0.5
~~~

## 2. 冻结 checkpoint

| Train Seed | Best Val Acc | Best Epoch | Best Checkpoint | SHA-256 | Source Commit |
|---:|---:|---:|---|---|---|
| 42 | 86.92% | 98 | `outputs/advanced/stage12/formal/r3_cutmix_alpha05/split_42__train_42/best.pt` | `D7D1F42C5DE6BB742A1196E70D16B49CDB45B79F986C69F65477FD65E7DB7309` | `5ae1a9f` |
| 123 | 86.42% | 98 | `outputs/advanced/stage13/r3_cutmix05/split_42__train_123/best.pt` | `9359EF01AF998F7A472662FD30001F0B39BD677A785434AA2FE5C47EDCA0B490` | `2ef26ee` |
| 2026 | 87.14% | 94 | `outputs/advanced/stage13/r3_cutmix05/split_42__train_2026/best.pt` | `8E40E191A5FA2385E68EE79616F3BB40A3BDF0C1D5DF2D277D95E0EDECC498C8` | `2ef26ee` |

这些 checkpoint 保持在原始运行目录，不复制、不改名。使用前应重新计算 SHA-256；不允许用后续训练权重覆盖。

## 3. 评估版本与规则

- 评估入口：`scripts/evaluate.py`；
- 冻结源码 Git blob：`4ce2edd171f61e2c579774a2efb98430b737389f`；
- validation 负责配方选择，test 不参与调参；
- 阶段 14–17 不因 clean test 表现更改模型配方；
- 阶段 15 已在冻结扰动协议后完成 seed 42 checkpoint 的 clean/corrupted test；不得据此修改模型或 severity；
- 阶段 18 复用 seed 42 clean test，并只补评估 seed 123/2026 的冻结 best checkpoint，最终报告三个 seed 的 mean 与 sample std；
- test 结果无论好坏都必须完整保留，不能删除不利 seed 或另加 seed 改善均值。

## 4. 冻结统计

~~~text
best validation accuracy：86.827 ± 0.369%
相对 R0 的配对提升： +2.947 ± 0.510 pp
~~~

以上 `±` 均为三个 train seed 的样本标准差，不表示置信区间或统计显著性。

## 5. 阶段 15 冻结鲁棒性评估

~~~text
evaluation commit：2a4a3e316aea482e79ab40c1e298a723a4e8938e
evaluator blob：2a389fc84518a283343417b37bac33a74cbefa1e
corruption seed：1501
advanced seed-42 clean test：86.38%
advanced seed-42 corrupted macro：69.01%
advanced seed-42 worst：25.80%，gaussian_noise severity 3
~~~

结果产物 SHA-256：

| Artifact | SHA-256 |
|---|---|
| `assets/advanced/stage15/corruption_grid.png` | `5E88C3EBE697A4D80226F3CBBCD7551D1A7821EAE9057213E1427AE57872085E` |
| `assets/advanced/stage15/robustness_conditions.csv` | `16868C7EDB5AD80462B390E0E94FAFC076D372E033C94A76152ADE3A718EE6F6` |
| `assets/advanced/stage15/robustness_summary.csv` | `8022C5CCB8E93E0EA5E141CCE440E9F1020A738D40D4CFBA9B194984BAFD31C1` |
| `assets/advanced/stage15/robustness_raw.json` | `85767358715E3545F3D7DEBC52D566A601752CBFB01268FC9304CEE6CEE7ED04` |

这些结果是在 clean worktree 上生成的正式 frozen-test 产物。后续分析可以读取，但不得重跑后挑选更有利输出或覆盖原文件。
