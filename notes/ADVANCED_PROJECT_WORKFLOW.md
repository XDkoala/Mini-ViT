# Mini Vision Transformer 进阶项目工作流

> 本文承接已完成的阶段 0–9，从阶段 10 开始。设计依据为 PROJECT_PLAN.md 的推荐进阶内容与具身方向延伸，并沿用基础工作流的方式：理解机制、拆成微步骤、测试、冒烟、正式实验、验收和记录。

设计日期：2026-09-25

当前状态：阶段 11 进行中

## 0. 进阶项目目标

基础项目已经证明 MiniViT 能正确学习，并完成 CNN 对照、三组消融、最终评估和可视化。进阶项目不再重复“把模型跑起来”，而要回答：

1. 更成熟的优化与正则化能否稳定改善小数据 MiniViT？
2. 改善是否跨随机种子成立，而不是 seed 42 的偶然结果？
3. 精度提升付出了多少 FLOPs、延迟、吞吐量和显存成本？
4. 模型面对相机扰动、域偏移和具身/第一视角图像时能否迁移？

最终证据链：

~~~text
基础冻结结果
  → 进阶实验协议
  → 优化与正则化候选
  → 多随机种子确认
  → 效率与鲁棒性分析
  → 具身视角迁移
  → 推理与解释工具
  → 进阶实验报告
~~~

## 1. 范围与边界

### 1.1 必做

- 解耦数据划分种子与训练随机种子；
- 加入 warmup、cosine decay、梯度裁剪与 AMP；
- 独立比较 Label Smoothing、Mixup、CutMix；
- 对最终候选运行至少 3 个训练种子，报告 mean ± std；
- 统计参数量、计算量、吞吐量、延迟和峰值显存；
- 建立模拟机器人相机扰动的鲁棒性评估；
- 完成一个规模可控的具身/第一视角迁移任务；
- 实现 Attention Rollout 和单图推理 CLI；
- 输出进阶结果总账、技术报告和讲解材料。

### 1.2 可选

- Gradio 本地 Demo；
- ResNet-18 参考基线；
- 校准误差与可靠性图；
- 更强的小图像 ViT 配方；
- 参数高效微调。

### 1.3 暂不纳入

- 从零训练 ImageNet；
- 同时展开检测、分割、视觉语言和机器人控制；
- 分布式多机训练；
- 根据测试集反复调参；
- 把 Attention 图称为因果解释。

## 2. 冻结基础结果

阶段 0–9 的结果是历史锚点，不覆盖、不改名：

| 模型 | Best Val Acc | Test Acc | 参数量 |
|---|---:|---:|---:|
| SimpleCNN baseline | 89.58% | 88.29% | 2,609,002 |
| MiniViT baseline | 81.84% | 80.65% | 2,693,578 |

进阶内容物理隔离：

~~~text
configs/advanced/
outputs/advanced/<experiment_name>/
assets/advanced/
notes/ADVANCED_EXPERIMENT_RESULTS.md
~~~

不得用新运行覆盖 minivit_baseline、cnn_baseline 或已有消融目录。

## 3. 数据隔离与随机性

### 3.1 测试集规则

- 阶段 10–13 只依据 train/validation 选择策略；
- 不为每个候选运行 clean test；
- 阶段 13 冻结最终配方、seed 和 checkpoint；
- 阶段 18 才统一评估冻结候选；
- test 结果不能反过来决定 scheduler、增强强度或结构。

鲁棒性开发只使用 corrupted validation。扰动类型、强度和模型冻结后，才生成 corrupted test。

具身数据必须按主体、场景、环境或录制序列拆分，不能把同一视频相邻帧随机分到不同 split。

### 3.2 拆分种子语义

基础代码用一个 seed 同时控制划分和训练。进阶配置必须拆分：

~~~yaml
randomness:
  split_seed: 42
  train_seed: 42
~~~

- split_seed：只决定 train/validation 索引，对照实验固定为 42；
- train_seed：控制初始化、训练顺序、增强和 dropout；
- corruption_seed：控制噪声和遮挡；
- benchmark_seed：生成固定效率测试输入。

正式训练 seed 建议为 42、123、2026。多种子实验只改变 train_seed，不改变 split_seed。

## 4. 阶段路线图

~~~text
阶段 10  进阶实验协议与基础设施
  ↓
阶段 11  优化训练栈：warmup / cosine / clipping / AMP
  ↓
阶段 12  小数据正则化：Label Smoothing / Mixup / CutMix
  ↓
阶段 13  多随机种子确认与统计分析
  ↓
阶段 14  参数量、FLOPs、吞吐量、延迟与显存
  ↓
阶段 15  相机扰动与域偏移鲁棒性
  ↓
阶段 16  具身/第一视角迁移学习
  ↓
阶段 17  Attention Rollout、推理 CLI 与可选 Demo
  ↓
阶段 18  冻结评估、技术报告与项目包装
~~~

| 阶段 | 核心问题 | 主要产物 |
|---:|---|---|
| 10 | 后续实验如何公平、可追溯？ | 配置协议、元数据、聚合工具 |
| 11 | 成熟训练策略是否改善收敛？ | scheduler、AMP、裁剪对照 |
| 12 | 哪种正则化改善小数据泛化？ | 三类正则化消融 |
| 13 | 提升是否跨 seed 成立？ | mean ± std、最终配方冻结 |
| 14 | 精度付出多少计算代价？ | 效率基准报告 |
| 15 | 相机扰动造成多大退化？ | corruption benchmark |
| 16 | CIFAR 表征能否迁移？ | scratch / probe / fine-tune |
| 17 | 如何提供解释与推理入口？ | rollout、predict CLI、Demo |
| 18 | 如何形成可复现成果？ | 最终 test、报告、README |

上一阶段未通过，不进入依赖它的下一阶段。

## 5. 建议目录

只在对应阶段需要时创建：

~~~text
Mini ViT/
├── configs/advanced/
│   ├── optimization/
│   ├── regularization/
│   ├── robustness/
│   └── transfer/
├── src/
│   ├── augmentations.py
│   ├── profiling.py
│   └── transfer.py
├── scripts/
│   ├── aggregate.py
│   ├── benchmark.py
│   ├── evaluate_robustness.py
│   ├── train_transfer.py
│   └── predict.py
├── tests/
│   ├── test_augmentations.py
│   ├── test_scheduler.py
│   ├── test_profiling.py
│   ├── test_robustness.py
│   └── test_transfer.py
├── notes/
│   ├── ADVANCED_PROJECT_WORKFLOW.md
│   ├── ADVANCED_EXPERIMENT_RESULTS.md
│   └── STAGE_10_NOTES.md ... STAGE_18_NOTES.md
├── reports/ADVANCED_REPORT.md
├── outputs/advanced/          # Git ignored
└── assets/advanced/
~~~

文件名是建议，不为每个小函数单独创建模块。

## 6. 进阶实验记录协议

每个正式 run 至少记录：

~~~yaml
experiment:
  id:
  name:
  group:
  output_dir:
  parent_experiment:

provenance:
  git_commit:
  created_at:
  python_version:
  torch_version:
  cuda_version:
  device_name:

randomness:
  split_seed:
  train_seed:

model:
  name:
  image_size:
  patch_size:
  embed_dim:
  depth:
  num_heads:
  parameter_count:

training:
  epochs:
  batch_size:
  optimizer:
  base_learning_rate:
  weight_decay:
  scheduler:
  warmup_epochs:
  gradient_clip_norm:
  amp:

regularization:
  label_smoothing:
  mixup_alpha:
  cutmix_alpha:

results:
  best_epoch:
  best_val_accuracy:
  test_accuracy:
  checkpoint_sha256:

efficiency:
  flops:
  throughput_images_per_second:
  latency_p50_ms:
  latency_p95_ms:
  peak_memory_mib:
~~~

不适用字段写 null，不静默省略。实验状态统一为：

~~~text
planned → smoke_passed → running → completed → evaluated
                              ↘ failed
~~~

smoke、失败运行与正式结果不能混在同一聚合表。

## 7. 阶段 10：进阶实验协议与基础设施

### 7.1 目标

在加入新技巧前，先保证实验可识别、恢复、聚合和公平比较。

### 7.2 必须理解

- 可复现不等于只调用一次 manual_seed；
- 划分随机性和训练随机性是不同变量；
- Git commit、有效配置和 checkpoint 哈希共同构成实验身份；
- 聚合脚本读取结构化结果，不能从终端日志猜数字；
- 新格式必须考虑旧配置和旧 checkpoint 的兼容性。

### 7.3 微步骤

1. 创建 STAGE_10_NOTES.md 和 ADVANCED_EXPERIMENT_RESULTS.md。
2. 设计 randomness.split_seed / randomness.train_seed，保留旧 seed 的兼容路径。
3. 给配置与 checkpoint 增加格式版本。
4. history 增加 learning_rate、gradient_norm 等可选字段。
5. checkpoint 支持 scheduler 与 AMP scaler 状态。
6. 记录环境版本、设备和 Git commit。
7. 定义 outputs/advanced 的目录命名。
8. 实现 aggregate.py，只聚合 completed run。
9. 编写兼容性、种子、续训与聚合测试。
10. 完成单 batch 和 2 epoch 冒烟。

建议新增 checkpoint 字段：

~~~python
{
    "format_version": 2,
    "scheduler_state": ...,
    "scaler_state": ...,
    "run_metadata": ...,
}
~~~

### 7.4 验收

- [x] 相同 split_seed、不同 train_seed 使用相同数据索引。
- [x] 不同 train_seed 改变训练采样等随机事件。
- [x] 旧 YAML 与旧 checkpoint 仍能加载。
- [x] optimizer 可续训，scheduler/scaler 状态可 round-trip；启用后的数值连续性在阶段 11 验证。
- [x] smoke 不进入正式聚合。
- [x] run 可追溯到 commit、有效配置和 seed。
- [x] 全量回归测试通过。

### 7.5 常见问题

- 改 train_seed 时也改变数据划分；
- 只保存模型，续训后学习率从头开始；
- 用目录名代替结构化配置；
- 失败运行被聚合成正式结果。

## 8. 阶段 11：优化训练栈

### 8.1 目标

加入 AMP、梯度裁剪、线性 warmup 和 cosine decay，并区分执行效率功能与学习策略。

| 能力 | 主要作用 | 是否改变学习轨迹 |
|---|---|---|
| AMP | 降低显存、加速 GPU | 有轻微数值差异 |
| Gradient Clipping | 限制异常梯度 | 触发时改变更新 |
| Warmup | 初期逐渐升高 LR | 是 |
| Cosine Decay | 后期平滑降低 LR | 是 |

### 8.2 微步骤与实验

1. 实现并测试 scheduler 的 step 语义。
2. 先记录 gradient norm，再加入 clip_grad_norm。
3. 加入 AMP 和 scaler 的设备兼容分支。
4. 无 CUDA 时安全回退。
5. 验证 checkpoint 续训后的 LR 与连续训练一致。
6. 运行 2 epoch smoke。
7. 只根据 validation 比较：

~~~text
O0：基础 AdamW + 固定 LR
O1：AdamW + cosine
O2：AdamW + warmup + cosine
O3：O2 + gradient clipping
~~~

8. AMP 用同一 O3 配方做速度/显存对照，不与 scheduler 消融混成一个变量。

若按 batch 调度：

~~~text
total_steps = epochs × steps_per_epoch
warmup_steps = warmup_epochs × steps_per_epoch
~~~

代码、配置、日志和续训必须统一按 batch 或按 epoch。

### 8.3 验收

- [x] warmup 学习率单调升至峰值。
- [x] cosine 平滑下降且不低于最小 LR。
- [x] scheduler step 数等于成功 optimizer update 数。
- [x] AMP 下先 unscale，再统计或裁剪梯度。
- [x] AMP 关闭时保持 FP32 路径。
- [x] scaler 与 scheduler 可续训。
- [x] history 记录 LR 和 gradient norm。
- [x] 最佳方案只依据 validation。

## 9. 阶段 12：小数据正则化

### 9.1 目标

在阶段 11 的固定优化协议上，独立研究 Label Smoothing、Mixup 和 CutMix。

| 方法 | 改变输入 | 改变目标 | 直觉 |
|---|---|---|---|
| Label Smoothing | 否 | 硬标签变软 | 减少过度自信 |
| Mixup | 两图线性混合 | 标签按比例混合 | 样本间更平滑 |
| CutMix | 粘贴局部区域 | 标签按面积混合 | 保留局部纹理 |

### 9.2 微步骤

1. loss 支持 label_smoothing。
2. 测试 Mixup 输入、软标签和 lambda。
3. 测试 CutMix 边界框、实际面积和标签比例。
4. 保证增强只作用于 train batch。
5. 保存参数与随机种子。
6. 分别 smoke。
7. 运行单变量实验：

~~~text
R0：无高级正则化
R1：Label Smoothing
R2：Mixup
R3：CutMix
~~~

只有单项证据明确后，才增加 R4：Mixup/CutMix 随机策略。

建议预注册小范围：

~~~text
label smoothing：0.05 / 0.10
mixup alpha：0.2 / 0.4
cutmix alpha：0.5 / 1.0
~~~

先短跑筛除明显无效值，正式结果与短跑分开。validation/test 使用原图和硬标签。

### 9.3 验收

- [ ] Label Smoothing 只改变训练 loss。
- [ ] Mixup 软标签概率和为 1。
- [ ] CutMix 用真实裁剪面积重算 lambda。
- [ ] validation/test 不使用 batch 混合。
- [ ] 配置可准确恢复增强策略。
- [ ] 每个正式实验只改变一个因素。
- [ ] 猜想与选择规则在看结果前记录。

训练 soft-label accuracy 与普通 hard-label accuracy 不等价，主要依据 validation 指标选择。

## 10. 阶段 13：多随机种子与统计

### 10.1 目标

确认进阶收益不是单次偶然波动，并冻结后续阶段使用的 clean-data 模型。

进入本阶段前最多保留：

- 基础 MiniViT 锚点；
- 一个阶段 11 最佳优化配方；
- 一个阶段 12 最佳正则化配方；
- 可选 CNN 锚点。

不要把所有失败候选都跑 3 seeds。

### 10.2 正式矩阵

~~~text
split_seed = 42
train_seed = 42, 123, 2026
~~~

每个 seed 使用相同预算与模型选择规则，不能删除较差 seed 或补跑到均值变好。

至少报告：

- best val accuracy 的 mean 与 sample std；
- 全部单次值；
- best epoch 分布；
- 时间 mean ± std；
- 参数量；
- 相对基础配方的逐 seed 差值。

样本标准差：

~~~text
s = sqrt(sum((x_i - mean)^2) / (n - 1))
~~~

3 个 seed 只能提供初步稳定性证据，不支持强显著性结论。

### 10.3 冻结

在任何新 clean test 前，根据 validation mean、波动、成本和复杂度选出 advanced_minivit_final，并冻结：

- 完整配置；
- 三个 seed；
- 每个 seed 的 best checkpoint；
- 评估脚本版本；
- clean/corruption test 规则。

### 10.4 验收

- [ ] 所有候选使用同一 split_seed。
- [ ] 每个配置完成预注册的全部 seed。
- [ ] 聚合结果可追溯到单次 run。
- [ ] 报告 mean、sample std 和原始值。
- [ ] 没有排除不利 seed。
- [ ] 最终配方在 test 前冻结。
- [ ] 冻结清单写入进阶结果总账。

## 11. 阶段 14：效率与资源基准

### 11.1 目标

回答达到某个精度需要多少计算、时间和显存。

| 指标 | 含义 | 注意 |
|---|---|---|
| Parameters | 可学习参数量 | 与 batch 无关 |
| FLOPs/MACs | 单次前向规模 | 说明计数约定 |
| Throughput | images/s | 固定 batch |
| Latency p50/p95 | 延迟分布 | 不能只测一次 |
| Peak Memory | 峰值显存 | 测量前重置 |
| Model Size | 权重体积 | 说明 dtype |

### 11.2 测量协议

- 同一设备、版本、输入 shape 和 dtype；
- FP32 与 AMP 分开；
- eval + inference_mode；
- 先 warmup 再计时；
- CUDA 计时前后同步；
- 重复多次，报告 median/p95；
- batch 1 测延迟，训练 batch 测吞吐量；
- 数据加载与纯模型时间分开；
- 不计首次 CUDA 初始化。

至少比较 CNN baseline、MiniViT baseline、advanced final，可选 depth 3。

### 11.3 验收

- [ ] 参数量与已有结果一致。
- [ ] FLOPs 约定记录清楚。
- [ ] benchmark 包含 warmup、同步和重复。
- [ ] latency 报告 p50/p95。
- [ ] throughput 写明 batch 与 precision。
- [ ] 显存区分训练和推理。
- [ ] 原始测量保存为 CSV/JSON。
- [ ] 结论同时讨论精度和成本。

## 12. 阶段 15：相机扰动与域偏移

### 12.1 目标

模拟机器人相机常见退化，测量 clean 到 corrupted 图像的性能下降。

第一版选 4–6 类确定性扰动，每类 3 个严重度：

~~~text
brightness shift
contrast reduction
gaussian noise
motion blur
occlusion
small translation
~~~

参数在查看最终结果前固定，并生成可视化样例。

### 12.2 流程

1. 在 validation 上实现 deterministic corruption dataset。
2. 验证标签、顺序和数量不变。
3. 保存 corruption/severity 样例网格。
4. 比较 CNN、MiniViT baseline、advanced MiniViT。
5. 只依据 corrupted validation 讨论鲁棒训练。
6. 若加鲁棒增强，使用独立配置。
7. 模型与 corruption 冻结后，运行 corrupted test 一次。

报告每种扰动/强度 accuracy、clean accuracy drop、macro average 和最坏情况。不能只报平均值。

### 12.3 验收

- [ ] corruption 不改变标签和样本数量。
- [ ] 同一 corruption seed 生成相同输入。
- [ ] severity 单调增强并完成图像验收。
- [ ] corrupted validation/test 隔离。
- [ ] 三个模型使用相同 corruption 输入。
- [ ] 同时报告 clean、各扰动和最坏情况。
- [ ] 结论不超出所测范围。

不要把 Gaussian Blur 直接称为真实运动模糊；不同模型也不能各自随机生成一套噪声。

## 13. 阶段 16：具身/第一视角迁移

### 13.1 目标

选择规模可控、标签清楚的第一视角或机器人视觉分类子任务，比较从头训练、线性探测和全量微调。

### 13.2 数据集选择关卡

候选必须满足：

- 许可证允许使用；
- 约 5–10 个清楚类别；
- 规模适合当前硬件；
- 能按主体、场景或录制序列拆分；
- 与具身感知有明确关系；
- 能记录来源、版本和筛选规则。

选择时需在线核查当前主页、许可证和规模，并创建 DATASET_CARD.md。本工作流不硬编码可能变化的下载链接。

划分优先级：

~~~text
跨主体 > 跨环境/场景 > 跨录制序列 > 随机图片
~~~

同一视频连续帧不得跨 split。

### 13.3 三种设置

| 设置 | Encoder | 分类头 | 问题 |
|---|---|---|---|
| Scratch | 随机初始化、训练 | 新建、训练 | 目标数据能学多少 |
| Linear Probe | 冻结 | 新建、训练 | 表征能否直接复用 |
| Fine-tune | 加载并训练 | 新建、训练 | 预训练是否是好起点 |

类别数改变时替换 classifier。第一版统一 resize/crop 到 32×32并记录细节损失；位置编码插值作为独立扩展。

### 13.4 验收

- [ ] DATASET_CARD 记录来源、许可、类别与筛选。
- [ ] 无相邻帧/同序列泄漏。
- [ ] 检查各 split 类别分布。
- [ ] 三种方法使用同一协议。
- [ ] Linear Probe 的 encoder 确实不更新。
- [ ] 分类头维度等于目标类别数。
- [ ] 报告 accuracy、macro F1、每类结果和错误案例。
- [ ] 分析域差异与 32×32 限制。

若冻结参数但 BatchNorm 统计仍更新，不能称为完全冻结。

## 14. 阶段 17：Attention Rollout 与推理入口

### 14.1 目标

实现跨层 Attention Rollout，并提供单图推理 CLI。Gradio 仅是可选展示层。

模型在可选模式返回 depth 个注意力张量：

~~~text
depth × [B, H, N, N]
~~~

Rollout 流程：

1. 明确融合每层多个 head 的方法；
2. 加入单位矩阵表示残差路径；
3. 对每行重新归一化；
4. 从浅到深做矩阵连乘；
5. 读取最终 CLS 到 patch 的联合权重；
6. reshape、插值并叠加原图。

至少比较最后一层 Attention 与 Rollout、正确与错误、clean 与 corrupted。Rollout 仍不是因果解释。

### 14.2 推理 CLI

目标命令：

~~~powershell
python -m scripts.predict --checkpoint outputs/advanced/<run>/best.pt --image <image_path> --top-k 5 --attention rollout
~~~

输出：

- top-k 类别与概率；
- 预处理说明；
- checkpoint 身份和 SHA-256；
- device；
- 可选热图保存路径；
- 清晰的格式/路径错误。

### 14.3 验收

- [ ] 普通 forward 向后兼容。
- [ ] 可选模式返回全部层 attention。
- [ ] 各层 attention 行归一化。
- [ ] rollout 包含残差并重新归一化。
- [ ] CLI 与正式评估使用相同预处理/类别映射。
- [ ] top-k 排序正确。
- [ ] CPU/CUDA 均可推理。
- [ ] 图片注明方法与解释边界。
- [ ] Demo 不复制模型逻辑或硬编码路径。

## 15. 阶段 18：冻结评估、报告与包装

### 15.1 目标

停止开发性调参，冻结模型、鲁棒性和迁移协议，完成最终评估与对外材料。

test 前写入冻结清单：

- advanced MiniViT 配置；
- seed 与 best checkpoint；
- clean test 规则；
- corruption 类型、强度与 seed；
- 迁移 split 与配置；
- benchmark 硬件与协议；
- Git commit；
- 最终图表清单。

一次性生成：

- clean CIFAR-10 多 seed test mean ± std；
- corrupted test 各扰动与 severity；
- 具身迁移 test；
- 效率总表；
- clean/corruption/transfer 错误案例；
- final Attention 与 Rollout 对比。

评估脚本 bug 可以修复并完整重跑，但必须记录原因和影响；不能因结果不理想换 checkpoint。

### 15.2 报告

2–4 页报告或等价 Markdown：

~~~text
1. 问题与动机
2. 基础 MiniViT/CNN 结论
3. 进阶训练方法
4. 多 seed 统计
5. 精度—效率权衡
6. 鲁棒性与具身迁移
7. Rollout 与错误分析
8. 限制和未来工作
~~~

项目包装：

- 更新 README 的 Advanced Experiments；
- 完成 ADVANCED_EXPERIMENT_RESULTS.md；
- 给出复现命令与资源预算；
- 准备 60 秒项目介绍；
- 准备 5 分钟技术提纲；
- 用真实数字更新简历表述；
- 可选创建 release/tag；
- 补充 LICENSE 后再鼓励外部复用。

### 15.3 验收

- [ ] 最终配置/checkpoint 在 test 前冻结。
- [ ] 所有 test 结果绑定 SHA-256 与 commit。
- [ ] clean、corruption、transfer、效率表完整。
- [ ] 报告包括失败实验。
- [ ] 图表有标题、坐标、单位、图例和身份。
- [ ] README 命令在干净环境可执行。
- [ ] Git 无数据、大权重、密钥和绝对路径。
- [ ] 全量测试通过。
- [ ] 结论不超出实验范围。

## 16. 预算与停止规则

### 16.1 最小完成版

| 阶段 | 正式长训练数 |
|---|---:|
| 11 优化 | 3–4 |
| 12 正则化 | 4 |
| 13 多 seed | 6–9 |
| 15 鲁棒性 | 不训练或 1 个鲁棒候选 |
| 16 迁移 | 3 |

短跑只用于排错/筛除明显无效候选，不进入最终表。

### 16.2 停止扩张

出现以下情况停止增加实验：

- 新实验无法回答新问题；
- 一次改变多个变量；
- 需要看 test 才能决定下一步；
- 数据许可或 split 说不清；
- 运行成本高但没有决策价值；
- 项目同时扩张到分类、检测、分割和控制。

## 17. 时间安排

| 时间 | 阶段 | 成果 |
|---|---|---|
| 第 1 周 | 10 | 配置、种子、元数据、聚合 |
| 第 2 周 | 11 | 优化训练栈 |
| 第 3 周 | 12 | 正则化实验 |
| 第 4 周 | 13 | 多 seed 与配方冻结 |
| 第 5 周 | 14–15 | 效率和 corruption |
| 第 6–7 周 | 16 | 具身迁移 |
| 第 8 周 | 17–18 | Rollout、推理、报告 |

训练等待时可并行写不依赖结果的测试、数据卡模板和报告骨架，但不能提前查看未冻结的 test。

## 18. 全局验收

- [ ] 基础与进阶结果隔离且可复现。
- [ ] split_seed / train_seed 解耦。
- [ ] scheduler、AMP、clipping 可配置且可续训。
- [ ] 三类正则化有独立证据。
- [ ] 最终配方至少 3 seeds。
- [ ] 报告 mean ± sample std 与全部单次值。
- [ ] 同时报告精度、参数量、FLOPs、延迟、吞吐量和显存。
- [ ] 完成固定 corruption 套件。
- [ ] 完成无帧泄漏的具身迁移。
- [ ] Rollout 与推理 CLI 可复现。
- [ ] test 未参与选择。
- [ ] 测试、报告、README 和结果总账一致。

## 19. 现在只做哪一步

当前只进入阶段 10，不立即实现 AMP、Mixup，也不下载具身数据。

第一微步是创建 STAGE_10_NOTES.md，暂时只写：

1. 阶段目标；
2. 基础协议的不可变锚点；
3. split_seed 与 train_seed 的定义；
4. 计划新增的 config/checkpoint/history 字段；
5. 尚未执行的验收清单。

随后逐个实现种子解耦、元数据和聚合契约。阶段 10 的兼容性测试、smoke 和全量回归通过后，才进入阶段 11。
