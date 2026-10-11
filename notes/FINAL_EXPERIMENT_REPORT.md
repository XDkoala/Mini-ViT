# MiniViT 实验报告

本项目从零实现小型 Vision Transformer，研究在 CIFAR-10 小图像分类上，结构、训练策略和正则化如何影响泛化，并检验效率、相机扰动和 CORe50 跨 session 迁移。最终 MiniViT 的 CIFAR-10 clean test 为 **86.260 ± 0.159%**，比基础 MiniViT 的单次结果高 5.61 个百分点，但仍低于 CNN 基线。改进训练配方没有增加模型的推理结构成本；鲁棒性和迁移结果则显示，clean accuracy 的提高不意味着所有场景都更可靠。

结果截止日期为 2026-10-11。`±` 均表示训练 seed 的样本标准差，不是置信区间；百分点写作 pp。

项目状态：阶段 0–18 已完成，代码、轻量实验产物与文档已上传 GitHub；未发布完整数据或 checkpoint，未创建 release 或添加代码许可证。2026-10-11 的验收接受 7 组基础历史评估缺原运行 commit 的明确例外，不补造历史版本。完成范围见[阶段 18 验收记录](STAGE_18_NOTES.md)。

## 1 问题与实验协议

CIFAR-10 官方训练集按 split seed 42 划分为 45,000 张训练图片与 5,000 张验证图片，官方 10,000 张测试图片只用于冻结后的评价。训练使用随机裁剪与水平翻转，验证和测试使用确定性预处理。配方和最佳 epoch 根据 validation 选择，不能根据 test 重选模型。

基础 MiniViT 为 patch 4、embedding 192、depth 6、heads 6、MLP 768，使用可学习的 CLS token 和位置编码，共 2,693,578 个参数。与 2,609,002 参数的 SimpleCNN 共用训练与评价管线。基础和进阶源域实验均训练 100 epochs；进阶配方用 train seeds 42、123、2026 确认。

所有最终结果保留 checkpoint 身份。进阶配方先按 validation 冻结；seed 42 的 clean test 复用阶段 15，seed 123、2026 在阶段 18 从 clean commit `2ed2a93` 补测，没有额外挑选 seed。三个 seed 共用一个测试集，因此统计只描述本划分上的训练随机性。

## 2 基础模型与结构消融

| 模型或设置 | Best validation | Test accuracy |
|---|---:|---:|
| SimpleCNN | 89.58% | 88.29% |
| MiniViT patch 4、depth 6、基础增强 | 81.84% | 80.65% |
| MiniViT patch 2 | 80.02% | 78.94% |
| MiniViT patch 8 | 77.30% | 76.38% |
| MiniViT depth 3 | 81.16% | 80.27% |
| MiniViT depth 9 | 81.46% | 80.36% |
| MiniViT 无基础增强 | 72.64% | 70.93% |

更小的 patch 并未获得更高准确率：patch 2 增加 token 数和注意力成本，patch 4 在当前协议下更合适。加深到 9 层也没有观察到收益，而 depth 3 用更少参数取得接近的结果。去掉增强后 test 降低 9.72 pp，表明记忆训练图片不等于泛化。

这些结构消融只有 seed 42，不能将微小差异推广为稳定排序。CNN 的优势与其局部性归纳偏置相符，但本对照不能单独证明归纳偏置是唯一原因。基础旧测试的权重哈希与预测明细可核验，但原评估运行 commit 未记录；结果归档版本不作为运行版本补填。原始指标和异常记录见[基础实验总账](EXPERIMENT_RESULTS.md)，追溯限制见[验收清单](../assets/advanced/stage18/evidence_audit.json)。

## 3 优化与正则化

单 seed 的优化消融中，固定学习率、cosine、warmup cosine、warmup cosine 加裁剪的最佳 validation 分别为 81.84%、82.94%、82.92%、83.00%。Cosine 相对固定学习率提高 1.10 pp；后三者差异很小，不足以断言 warmup 或 clipping 稳定提高最终精度。

Warmup 使前 5 轮梯度范数的标准差从 0.343 降到 0.117，但早期 loss 下降更慢。clip norm 1.0 的平均触发比例为 99.54%，与实验前“偶发保险丝”的预测不符：在本配方里它是持续的优化约束。FP32/AMP 对照的主要时间指标为 epoch 6–100 中位数，AMP 获得约 1.455 倍加速，peak allocated 显存减少 26.47%；这不是稳定的精度提升证据。

在统一 O3+AMP 底座上，R0、Label Smoothing 0.05、Mixup 0.2、CutMix 0.5 的单 seed validation 为 83.64%、83.52%、84.56%、86.92%。Label Smoothing 没有超过 R0；30-epoch screening 的跨方法排名也没有延续到正式 100-epoch 结果。因此短跑只用于方法内部筛选强度，不替代正式比较。

最终配方保留原模型结构，使用 AdamW、学习率 0.0003、weight decay 0.05、5-epoch warmup cosine、clip norm 1.0、AMP 与 CutMix alpha 0.5。完整条件和冻结权重见[配方冻结清单](ADVANCED_MINIVIT_FINAL.md)。

## 4 多 seed 与最终测试

| 指标或配方 | Seed 42 | Seed 123 | Seed 2026 | Mean ± sample std |
|---|---:|---:|---:|---:|
| R0 best validation | 83.64% | 84.06% | 83.94% | 83.880 ± 0.216% |
| CutMix best validation | 86.92% | 86.42% | 87.14% | 86.827 ± 0.369% |
| 配对 validation 提升 | +3.28 pp | +2.36 pp | +3.20 pp | +2.947 ± 0.510 pp |
| 冻结 CutMix clean test | 86.38% | 86.32% | 86.08% | **86.260 ± 0.159%** |

CutMix 相对 R0 的 validation 提升在三个 seed 上方向一致，但没有做统计显著性检验。最终 test 均值相对基础 MiniViT 单次 test 数值上提高 5.61 pp；这是整套训练配方的对照，不是 CutMix 单独贡献的 test 估计。

seed 2026 的 validation 最高，test 却最低，说明验证排名不保证测试排名。seed 42 复用 FP32 batch 256，其余两组使用冻结 checkpoint 的 FP32 batch 128；预处理相同、没有测试时增强，批大小相关的微小浮点差异仍属于协议边界。原始身份与统计见[最终测试 JSON](../assets/advanced/stage18/clean_test_summary.json)。

## 5 精度与效率

在 RTX 5060 Laptop GPU、PyTorch 2.14.0+cu130 上，batch 1 的 FP32 latency 使用 50 次 warmup、200 次测量；batch 128 的 AMP throughput 使用 20 次 warmup、100 次测量。CUDA 计时边界同步，不包含数据读取。MACs 使用相同算子计数口径，不等同于实际耗时。

| 模型 | Params | MACs 每图 | FP32 latency p50 | AMP throughput | AMP 训练 peak allocated |
|---|---:|---:|---:|---:|---:|
| CNN | 2.609 M | 115.906 M | 0.414 ms | 54,757 img/s | 254.42 MiB |
| 基础 MiniViT | 2.694 M | 182.850 M | 1.692 ms | 20,274 img/s | 632.59 MiB |
| 进阶 MiniViT | 2.694 M | 182.850 M | 1.644 ms | 20,456 img/s | 632.59 MiB |

训练改进没有增加参数量与推理计算结构；两套 MiniViT 的小幅延迟差异不能当作算法级加速。CNN 在当前任务上仍兼具更高 clean accuracy 与更低计算成本。AMP 的 batch 吞吐优势也不能直接推成单图延迟优势。完整 FP32/AMP 表见[效率基准](../assets/advanced/stage14/benchmark_summary_5c1dbfb.csv)。

## 6 扰动与跨域迁移

相机扰动协议包含 brightness、contrast、Gaussian noise、motion blur、occlusion、translation，各 3 个 severity，noise seed 固定为 1501。每个条件生成一次供三个模型共享；macro 是 18 个条件准确率的等权均值，不是类别 macro F1。

| 模型，seed 42 | Clean test | Corruption macro | Worst condition |
|---|---:|---:|---|
| CNN | 88.28% | 68.77% | 18.81%，noise S3 |
| 基础 MiniViT | 80.65% | 64.79% | 26.77%，contrast S3 |
| 进阶 MiniViT | 86.38% | 69.01% | 25.80%，noise S3 |

CNN 与进阶 MiniViT 的 macro 仅差约 0.24 pp，不能据此宣布后者全面更鲁棒。进阶配方在 occlusion 上更强，而基础 MiniViT 在 Gaussian noise 上反而最好。CNN 本表的 clean 88.28% 来自阶段 15 同协议评价，与基础阶段的 88.29% 有 0.01 pp 差异；两个记录分别保留，不能择优替换。各 severity 结果见[扰动明细](../assets/advanced/stage15/robustness_conditions.csv)。

CORe50 将源域分类头替换为目标域 10 类分类头，并比较随机初始化、冻结 encoder 的 Linear Probe、全模型 Fine-tune。每 10 帧固定抽一张，按 session 划分：train 为 s1/s2/s4/s5/s6/s9/s11，validation 为 s8，test 为 s3/s7/s10；三组分别为 10,452、1,491、4,482 张，均缩放到 32×32。

| 迁移方式，seed 42 | Validation accuracy | Test accuracy | Test macro F1 |
|---|---:|---:|---:|
| Scratch | 54.19% | 36.88% | 36.76% |
| Linear Probe | 42.45% | 35.36% | 33.13% |
| Fine-tune | 71.83% | 53.82% | 53.15% |

Fine-tune 的 test 比 Scratch 高 16.93 pp，说明本协议下源域初始化和目标域适应有价值；仅训练新头明显受限。但 Fine-tune 从 validation 到 test 降低约 18.02 pp，不能称为稳定跨场景识别。划分隔离相邻帧，却允许同物体实例跨 session，因此没有证明新实例泛化。详见[数据卡](DATASET_CARD.md)与[迁移结果](../assets/advanced/stage16/test_summary.json)。

## 7 Rollout 与错误分析

末层 CLS Attention 描述最后一个 block 的直接读取权重；Rollout 对各层多头取均值，加单位矩阵近似残差、逐行归一化，再按后层乘先层组合，追踪最终 CLS 到原始 patch 的近似路径。它忽略 Value 内容、MLP 和归一化等影响，因此不是因果解释。

两个冻结模型各从 validation 按固定扫描规则选两个正确、两个错误案例，并配对 Gaussian noise S2。案例中 CORe50 正确图像加入噪声后出现高 softmax 错判，说明高分并不保证可靠；这不是整体错误率估计。两种热图分别归一化，颜色不能跨图直接比较。四张固定对照图与 16 条诊断记录见[阶段 17 产物](../assets/advanced/stage17/diagnostics.json)，单图推理入口为 `python -m scripts.predict`。

## 8 局限与后续问题

进阶源域只有三个训练 seed、一个数据划分；基础消融、扰动与迁移主要只有一个 seed。当前结论不能推广到大分辨率、大规模预训练或所有 CNN/ViT。CORe50 的域差异同时包含背景、光照、物体尺度与缩放损失，现有实验不能分离其贡献；分类任务也不等同于机器人检测、分割或控制。

后续应把新实例划分、目标域学习率、输入分辨率和更强 CNN 对照注册为新实验，以新的验证协议回答具体问题，不继续利用已查看的 test 反向优化本轮结论。项目的主要成果是从模型实现到冻结评估的完整实验链路，而不是证明某类架构普遍优越。原环境与独立新环境的代码回归均为 479 passed，新环境还通过受限冒烟与单图推理；复现入口和资源条件见[复现指南](REPRODUCTION_GUIDE.md)，所有正式实验见[进阶实验总账](ADVANCED_EXPERIMENT_RESULTS.md)。
