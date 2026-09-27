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
- 阶段 15 先在 corrupted validation 上冻结扰动协议，之后才能生成 corrupted test；
- 阶段 18 对三个冻结 best checkpoint 进行统一 clean test，并报告单次结果、mean 与 sample std；
- test 结果无论好坏都必须完整保留，不能删除不利 seed 或另加 seed 改善均值。

## 4. 冻结统计

~~~text
best validation accuracy：86.827 ± 0.369%
相对 R0 的配对提升： +2.947 ± 0.510 pp
~~~

以上 `±` 均为三个 train seed 的样本标准差，不表示置信区间或统计显著性。
