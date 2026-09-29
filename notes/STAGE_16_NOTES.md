# 阶段 16：具身/第一视角迁移学习

本阶段研究 CIFAR-10 上训练得到的 MiniViT encoder，能否迁移到类别完全不同、拍摄方式也不同的具身视角物体分类任务。

开始日期：2026-09-28

当前状态：正式训练与 validation 分析完成；测试协议待冻结后执行

当前位置：三组正式训练及逐类 validation 复核完成；下一步冻结评估协议并评估 test

## 进度

- [x] 明确迁移问题与三种训练设置
- [x] 在线核查候选数据集的来源、规模、组织方式与许可
- [x] 选择 CORe50 的 10 类 category-level 任务
- [x] 冻结抽帧规则和 session-level split
- [x] 创建 `DATASET_CARD.md`
- [x] 下载并校验原始数据
- [x] 生成固定 manifest 与类别统计
- [x] 检查无 session/相邻帧泄漏
- [x] 人工验收目标域图像与 32×32 缩放效果
- [x] 实现 CORe50 dataset/dataloader 与契约测试
- [x] 实现分类头替换和 encoder 冻结机制
- [x] 完成 scratch / linear probe / fine-tune smoke
- [x] 在查看结果前记录实验猜想
- [x] 完成三组正式训练并只依据 validation 分析
- [ ] 冻结协议后一次性评估 test
- [ ] 报告 accuracy、macro F1、逐类结果与错误案例
- [ ] 阶段 16 总验收

## 1. 核心问题

CIFAR-10 和 CORe50 的类别并不相同，因此不能复用旧分类头。本阶段真正要迁移的是 encoder 学到的视觉表示：边缘、颜色组合、局部纹理、形状和 token 间关系等。

~~~text
CIFAR-10 checkpoint
  ├─ Patch Embedding             保留候选
  ├─ Transformer Encoder         保留候选
  └─ Linear(192, 10)             丢弃
                                   ↓
                              新建 Linear(192, 10)
                              对应 CORe50 新类别
~~~

新旧任务恰好都是 10 类只是数值巧合。两个分类头的每个输出位置语义完全不同，因此旧分类头仍必须删除并随机初始化。

## 2. 为什么最终选择 CORe50

比较过的主要候选是 CORe50、OpenLORIS-Object 和 iCubWorld。CORe50 最适合当前项目：

- 官方定义 10 个日常物品类别，与当前分类头规模相近；
- 11 个独立 sessions 可用于无相邻帧泄漏的分组划分；
- 操作者视角、手持距离、背景/光照/遮挡变化与具身感知有关；
- 有裁剪好的 128×128 RGB 图像，不需要先实现检测器；
- 间隔抽帧后约 1.65 万张，适合当前单卡训练预算；
- 官方项目材料提供明确归属与 CC BY 4.0 许可文本。

OpenLORIS-Object 有 19 类、约 21.5 万张并包含更多环境因子，第一版工程负担更大；iCubWorld 也很相关，但不同版本和下载组织更复杂，许可表述不如 CORe50 当前资料清晰。它们保留为后续扩展，不纳入本阶段主实验。

## 3. 防止视频帧泄漏

若随机打散 164,866 张图片再切分，同一段约 20 fps 视频中的相邻帧可能一张进入 train、下一张进入 validation。模型只需记住背景、手和几乎相同的物体姿态，就会得到虚高的验证准确率。

因此本阶段采用两层约束：

1. 每 10 帧固定抽取一张，减少时间冗余；
2. 以完整 session 切分，任何 session 不能跨 split。

冻结划分为：

~~~text
train:      s1, s2, s4, s5, s6, s9, s11
validation: s8
test:       s3, s7, s10
~~~

测试 sessions 沿用官方划分；validation session 在查看任何训练结果前固定。详细协议见 `DATASET_CARD.md`。

## 4. 三种训练设置

| 设置 | Encoder 初值 | Encoder 是否训练 | 新分类头 | 回答的问题 |
|---|---|---|---|---|
| Scratch | 随机 | 是 | 随机并训练 | 只靠目标数据能学到多少 |
| Linear Probe | CIFAR checkpoint | 否 | 随机并训练 | 旧表征不修改时能否直接分开新类别 |
| Fine-tune | CIFAR checkpoint | 是 | 随机并训练 | 旧表征是否提供了更好的训练起点 |

公平比较要求三组使用相同的数据 manifest、分辨率、normalization、训练预算和选择指标。除了初始化与 encoder 是否可训练之外，不能顺手改变其他变量。

Linear Probe 中“冻结”必须同时满足：

- encoder 参数 `requires_grad=False`；
- optimizer 不包含 encoder 参数；
- 训练后 encoder 权重逐元素不变；
- 当前 MiniViT 没有 BatchNorm，因此不存在冻结参数但运行统计仍变化的问题。

## 5. 预期解释边界

可能出现的结果包括：

- Fine-tune 优于 Scratch：CIFAR 表征提供了有用起点；
- Linear Probe 高于随机水平但低于 Fine-tune：已有特征可复用，但必须适应目标域；
- Linear Probe 很弱：冻结的 CIFAR 特征无法直接分开 CORe50；
- Scratch 接近或超过 Fine-tune：小型 CIFAR 预训练的迁移收益有限，甚至存在负迁移。

无论出现哪种情况，都不能据此泛化到“大规模预训练 ViT 的迁移能力”。我们的源模型只在 CIFAR-10 上从头训练，数据规模和视觉多样性远小于 ImageNet 等预训练语料。

## 6. 下一微步

用户依次启动三条正式训练命令。每组结束后验收 checkpoint、完整 validation 指标与状态；三组全部完成后比较预注册猜想。训练期间不读取 CORe50 test，也不根据单组结果修改剩余配置。

## 7. 数据完整性验收结果

官方压缩包大小为 5,892,103,007 字节，SHA-256 为 `f3d65017490dfc4392bc5e41b66930600f1d00898cae47fc4bd1a01cf92e1c6a`。ZIP 共 165,428 个条目，完整 CRC 检查没有坏条目；解压后恰有 164,866 张 PNG，与官方规模一致。

固定规则 `frame_id % 10 == 0` 选择 16,425 张图片：

| Split | Images | Sessions | 每类范围 |
|---|---:|---|---:|
| train | 10,452 | s1, s2, s4, s5, s6, s9, s11 | 1,043–1,050 |
| validation | 1,491 | s8 | 147–150 |
| test | 4,482 | s3, s7, s10 | 446–450 |

三组均覆盖全部 10 类，类别分布近似平衡；session 集合互斥，因此不存在相邻视频帧跨 split。所有 16,425 张入选图片都已实际解码并转换为 RGB，尺寸均为 128×128，没有静默跳过。

manifest SHA-256 为 `d12f46f688851a08287f0220c85f49c5d2ee05e0c084b81066d9b79d101633d0`。准备脚本首次按文件方式直接运行时暴露 `ModuleNotFoundError: src`，随后加入仓库根目录引导和真实子进程测试；最终直接命令与模块导入均可用，数据文件未受失败入口影响。

## 8. 32×32 视觉验收

`assets/advanced/stage16/core50_samples.png` 对每个类别展示一张 train 原图和真实 bilinear+antialias 降采样后的 32×32 输入；为了看清像素，右侧仅使用 nearest-neighbor 放回 128×128 展示。代表样本由每类排序后的中间记录确定，没有查看 validation/test 或人工挑图。

十类缩小后仍保留可辨认的整体形状和主要颜色。剪刀、灯泡、眼镜、杯子和遥控器轮廓尤其清楚；插头、马克笔等小结构丢失较多，但主体仍可见，因此 32×32 协议具有可学习性，同时保留了真实难度。

样例还显示手部和地砖背景占据大量像素。模型可能学习“手、背景、拍摄 session”与标签间的偶然相关性，而非只看物体。按完整 session 隔离不能消除所有背景捷径，但能阻止几乎相同的背景和连续帧直接跨 split；后续错误分析必须关注这一限制。

## 9. Dataset 与 DataLoader 契约

`Core50Dataset` 只读取 manifest 中的相对路径和 category label。每次取样打开 PNG、转为 RGB、完成读取后关闭文件，再执行 transform；不从目录结构临时猜标签，也不会复制整套图片到内存。

训练管线先 bilinear+antialias 缩放到 32×32，再沿用 CIFAR 的 `RandomCrop(32, padding=4)` 与 `RandomHorizontalFlip`，最后使用 CIFAR-10 mean/std normalization。validation/test 只有缩放、张量转换和归一化，因此同一图片重复读取逐元素一致。沿用源任务 normalization 是为了不把数值约定变成 scratch/probe/fine-tune 之外的额外变量。

CORe50 无需 `split_seed`：数据成员由冻结 manifest 决定。`train_seed` 控制训练 RandomSampler；多 worker 时 DataLoader 由同一 generator 派生 worker seed，`num_workers=0` 时随机 transform 还依赖训练入口调用进程级 `set_seed`。validation/test 使用 SequentialSampler 且不 shuffle。

真实数据冒烟结果：

~~~text
dataset sizes: train=10452, validation=1491, test=4482
loader batches @ batch_size 128: 82, 12, 36
train batch:      [128, 3, 32, 32], float32, labels 0–9
validation batch: [128, 3, 32, 32], float32
samplers: RandomSampler / SequentialSampler / SequentialSampler
~~~

validation 的首个 batch 因确定性排序只包含 label 0，这是预期现象；完整遍历覆盖全部十类。验收只读取 train/validation batch，没有提前读取或评估 test 图片。

## 10. 迁移模型实现

MiniViT 现将 `forward_features()` 与 `forward()` 分开：前者完成 Patch Embedding、CLS/位置编码、全部 Transformer Blocks 和最终 LayerNorm，返回 `[B, 192]` CLS 表征；后者只负责把该表征送入 `classifier`。普通训练、Attention 返回值和旧 checkpoint 键名保持兼容。

三种模式统一使用 `TransferMiniViT`：

| Mode | Encoder 初值 | Encoder 梯度 | Encoder 训练模式 | Optimizer 参数 |
|---|---|---|---|---|
| scratch | 随机 | 有 | train | 全部 80 个参数张量 |
| linear_probe | CIFAR checkpoint | 无 | eval | 仅 classifier weight/bias |
| fine_tune | CIFAR checkpoint | 有 | train | 全部 80 个参数张量 |

### 10.1 旧分类头必须显式排除

CIFAR-10 与 CORe50 都是 10 类，因此旧、新分类头 shape 都是 `[10, 192] + [10]`。PyTorch 无法凭 shape 发现语义已经变化。加载器明确过滤 `classifier.weight` 和 `classifier.bias`，要求其余 encoder keys 与目标模型完全一致；若缺少 encoder tensor、出现额外 key 或架构字段不同则立即失败。新分类头保留目标模型创建时的随机初始化。

### 10.2 Linear Probe 是四层冻结

1. 所有 encoder 参数设为 `requires_grad=False`；
2. `trainable_parameters()` 只向 optimizer 返回新 classifier；
3. encoder forward 位于 `torch.no_grad()` 内，不建立反向图；
4. 调用整个模型的 `train()` 后，encoder 仍保持 `eval()`，关闭其中 Dropout，只有 classifier 为 train。

契约测试实际执行一次 SGD update：classifier 权重发生变化，encoder 所有参数与训练前 clone 逐元素 `torch.equal`，且 encoder `.grad` 全为 `None`。Fine-tune 的对应测试则确认 encoder 与 classifier 都发生更新。

一次早期结构测试在 train 模式下连续调用 `forward_features(x)` 与 `forward(x)` 并要求完全相等，因此失败；两次调用采样了不同 Dropout mask。将“结构等价”测试改在 eval 模式后通过。这个失败没有修改模型逻辑，反而验证了 Linear Probe 单独关闭 encoder Dropout 的必要性。

### 10.3 真实冻结 checkpoint 冒烟

使用阶段 12/13 冻结的 seed-42 advanced checkpoint：

~~~text
source best epoch: 98
source best val: 86.92%
source SHA-256: d7d1f42c5de6bb742a1196e70d16b49cdb45b79f986c69f65477fd65e7db7309

scratch:      output [2,10], trainable 2,693,578, encoder train
linear_probe: output [2,10], trainable     1,930, encoder eval
fine_tune:    output [2,10], trainable 2,693,578, encoder train
~~~

SHA-256 与 `ADVANCED_MINIVIT_FINAL.md` 冻结清单一致。该冒烟只在 CPU 上前向，没有进行 CORe50 正式训练或读取 test 图片。

## 11. 正式训练协议

三组配置位于 `configs/advanced/stage16/`。除实验身份、输出目录、transfer mode 和是否存在 source checkpoint 外，其余字段完全一致：

~~~yaml
train_seed: 42
batch_size: 128
augmentation: basic
epochs: 50
optimizer: AdamW
learning_rate: 0.0003
weight_decay: 0.05
scheduler: 3-epoch linear warmup + cosine decay
min_factor: 0.1
gradient_clip_norm: 1.0
amp: true
label_smoothing: 0.0
batch_mixing: none
~~~

第一版没有给 Linear Probe 单独提高 head LR，也没有为 Fine-tune 使用更小 encoder LR。虽然不同模式可能各自偏好不同超参数，但本实验优先保持“只改变初始化和 encoder 是否冻结”的单变量解释。若未来研究每种模式的最佳性能，必须作为独立扩展矩阵，不能覆盖本结果。

目标域不使用 CutMix 或 Label Smoothing。源 encoder 来自 advanced CutMix checkpoint，但 CORe50 三种模式的目标域训练正则化保持一致，避免把“迁移方式”与“目标域高级正则化”混合。

训练入口 `scripts/train_transfer.py` 具备：

- manifest 与 source checkpoint SHA-256 启动前复核；
- 直接脚本运行的项目根目录引导；
- AMP、warmup/cosine、gradient clipping 与逐 epoch 指标；
- `best.pt`、`last.pt`、`history.csv`、有效 `config.yaml` 和 `run_status.json`；
- smoke 必须提供独立 `--output-dir`，不能覆盖正式目录；
- 只构建 train/validation 训练循环，不评估 CORe50 test；
- Linear Probe 在训练结束及 resume 后再次检查 encoder 是否与源权重完全一致。

## 12. 三组端到端 smoke

每组使用真实 CORe50、真实 source checkpoint、CUDA AMP，运行 4 epochs × 每轮最多 2 个 train/validation batches。smoke 输出位于 `outputs/advanced/stage16/smoke/`，状态均为 `completed` 且 `aggregation_eligible=false`。

| Mode | Trainable params | Peak allocated memory | 产物 | 冻结复核 |
|---|---:|---:|---|---|
| Scratch | 2,693,578 | 633.0 MiB | 完整 | 不适用 |
| Linear Probe | 1,930 | 178.4 MiB | 完整 | 78 个 encoder tensors 全相等 |
| Fine-tune | 2,693,578 | 633.0 MiB | 完整 | encoder 可更新 |

Linear Probe 的新 classifier 与 source classifier 不相等，证明旧 head 未被沿用。smoke validation 只遍历最前两个 sequential batches，类别覆盖不完整，因此显示的准确率不能作为模式优劣证据，也不会写入正式实验总账。

## 13. 正式实验前猜想（用户原始判断）

记录日期：2026-09-29。以下判断由用户在查看正式迁移实验结果前给出；此前的短 smoke 准确率因 validation 类别覆盖不完整，不作为预测依据。

| 问题 | 预测 | 用户理由或保留意见 |
|---|---|---|
| 最佳 validation accuracy | Fine-tune > Scratch > Linear Probe | 预训练应有明显帮助，冻结编码器的适应能力有限 |
| 前期学习最快 | Linear Probe | 只训练新分类头，很快收敛，但无法进一步深入学习 |
| 最晚达到最佳 epoch | Scratch | 从头学习视觉特征，收敛较慢 |
| Linear Probe 是否明显超过随机水平 10% | 能 | 已有视觉表征应该具有一定迁移能力 |
| Linear Probe 与 Fine-tune 的最佳准确率差距 | 大 | 可训练 encoder 能适应新的图像和类别 |
| Fine-tune 相比 Scratch | 应有明显优势 | 预训练提供更好的初始表征 |
| 负迁移 | 有可能 | 最不确定的判断；旧表征也可能妨碍新任务 |
| 训练准确率最高 | Fine-tune | 用户预测 |
| train–validation accuracy gap 最大 | Scratch | 训练集较小，更易过拟合 |

结果复盘时，“最佳验证准确率”统一取每组 50 epoch 的最高完整 validation accuracy；“最佳 epoch”取首次达到最高值的 epoch；泛化差距统一计算同一最佳 epoch 的 `train_accuracy - val_accuracy`，不拿不同 epoch 的最大值相减。Linear Probe 是否“明显高于 10%”与 Fine-tune 是否“明显优于 Scratch”，先报告精确百分点差，不擅自补写用户未指定的显著性阈值。单 seed 结果只能作为本次对照的初步证据，不能据此断言迁移效果跨 seed 稳定。

## 14. 正式训练与 validation 结果

三组均为用户启动的正式 50-epoch、seed-42 训练。`run_status.json` 均为 `completed`、`aggregation_eligible=true`、干净 Git 快照；训练配置与 manifest 一致。Linear Probe 的 78 个 encoder tensors 在 `best.pt` 与 `last.pt` 均与源 checkpoint 逐元素相等，Fine-tune 的 78 个 encoder tensors 则均有改变。评估脚本重新加载三组 `best.pt`，逐张遍历全部 1,491 张 validation 图像，其 accuracy 与训练历史最佳 epoch 完全一致。

| 模式 | 最佳 epoch | 最佳 val accuracy | 同轮 train accuracy | 同轮 gap | Val macro F1 | 最终 epoch val accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | 27 | 54.19% | 91.80% | 37.61 pp | 55.10% | 49.30% |
| Linear Probe | 46 | 42.45% | 52.17% | 9.72 pp | 40.07% | 42.19% |
| Fine-tune | 33 | **71.83%** | 98.94% | 27.11 pp | **70.82%** | 70.09% |

Fine-tune 比 Scratch 高 17.64 pp，比 Linear Probe 高 29.38 pp；Scratch 比 Linear Probe 高 11.74 pp。三个模式都远高于 10% 的十类均匀随机基线，但这不是统计显著性检验。Fine-tune 到 epoch 1 已达到 44.13% val accuracy，Scratch 与 Linear Probe 分别仅为 12.54%、12.01%；到 epoch 3 分别为 67.67%、22.67%、18.51%。所以前期最快的是 Fine-tune，不是参数最少的 Linear Probe。

逐类 validation 结果中，Fine-tune 的 `light_bulbs` F1 仅约 0.32，明显低于其总体水平；`cans` 和 `plug_adapters` 约 0.62。下游分析不应把 71.83% 误读为每类都同样可靠。完整的混淆矩阵、各类 precision/recall/F1、逐张预测和高置信错误样本见 `outputs/advanced/stage16/evaluation/validation_v1/`。这些错误是固定 validation 上的诊断结果，不用于改动本轮模型。

### 14.1 实验前猜想复盘

| 预测 | 对照结果 | 判断 |
|---|---|---|
| Fine-tune > Scratch > Linear Probe | 71.83% > 54.19% > 42.45% | 符合 |
| Linear Probe 前期最快 | Fine-tune 在 epoch 1、3 明显领先 | 不符合；只训练少量参数不等于表征立即适配新任务 |
| Scratch 最晚到最佳 epoch | Probe 46 > Fine-tune 33 > Scratch 27 | 不符合；共享 LR 下 Probe 仍缓慢改善，不能据此断定它已充分优化 |
| Probe 明显超过 10%，但与 Fine-tune 差距大 | +32.45 pp；落后 Fine-tune 29.38 pp | 符合本次定性判断 |
| Fine-tune 比 Scratch 有明显优势 | +17.64 pp | 符合；本协议下未观察到负迁移，但不能排除其他数据或配方 |
| Fine-tune train accuracy 最高 | 最佳 epoch 98.94% | 符合 |
| Scratch 的 train–validation gap 最大 | 37.61 pp > 27.11 pp > 9.72 pp | 符合 |

反思：模型可训练参数少只表示每步更便宜，不保证优化轮次更少。Linear Probe 只能重新组合固定的源域特征；当类别和拍摄域变化较大时，新分类头可能受特征可分性限制。Fine-tune 能调整特征，因此既快速受益于源域初始化，又能适应目标域；但它的 27.11 pp gap 与最终轮验证准确率回落也显示明显过拟合。所有结论仅针对这一次 session-level split、seed 42、共用超参数；尤其不能从一个 seed 推出稳定性或因果归因。

## 15. 冻结的最终评估协议

`configs/advanced/stage16/evaluation.yaml` 固定 manifest SHA-256、三组 `best.pt` 路径与 SHA-256、各自最佳 epoch、batch size。`scripts/evaluate_transfer.py` 严格校验哈希、mode、epoch 和样本顺序；先仅运行 validation 核验。协议和代码通过测试、提交后，才一次性对三组相同 test split 运行评估；test 不用于模型选择、调参或重跑筛选。记录 accuracy、macro F1、逐类 precision/recall/F1、混淆矩阵及高置信错误实例。不同 session 的 test 反映拍摄条件迁移，但同一物体实例仍可能跨 session 出现，不能称作新实例泛化。
