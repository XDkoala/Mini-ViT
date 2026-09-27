# 阶段 12：小数据正则化

本阶段在阶段 11 的固定优化协议上，独立研究 Label Smoothing、Mixup 和 CutMix，理解它们分别如何改变监督信号，并通过单变量实验判断它们是否改善 MiniViT 在 CIFAR-10 上的泛化。

开始日期：2026-09-26

当前状态：进行中

当前位置：R2 已验收并可正式聚合；下一步运行 R3

## 进度

- [x] 建立阶段 12 学习地图与实验边界
- [x] 理解硬标签、one-hot 标签与交叉熵
- [x] 拆分训练与验证 criterion
- [x] 理解并实现 Label Smoothing
- [x] 测试 Label Smoothing 只改变训练 loss
- [x] 理解并实现 Mixup
- [x] 测试 Mixup 的输入、软标签和 lambda
- [x] 理解并实现 CutMix
- [x] 测试边界框、实际面积和修正后的 lambda
- [x] 保证 batch 混合只作用于 train
- [x] 将正则化策略接入配置、checkpoint 与实验元数据
- [x] 完成各策略 smoke test
- [x] 在查看结果前记录实验猜想和选择规则
- [x] 完成 6 组 screening 并冻结正式实验参数
- [ ] 完成 R0–R3 单变量正式实验与聚合
- [ ] 根据单项证据决定是否增加组合策略 R4
- [ ] 阶段 12 总验收

## 1. 本阶段研究什么

| 方法 | 是否改变输入 | 是否改变训练目标 | 核心直觉 |
|---|---|---|---|
| Label Smoothing | 否 | 是：硬标签变为平滑目标 | 减少过度自信 |
| Mixup | 是：两张完整图线性混合 | 是：标签按同一比例混合 | 约束样本之间的预测更平滑 |
| CutMix | 是：把一张图的局部区域贴到另一张图 | 是：标签按实际区域比例混合 | 保留更真实的局部纹理 |

三种方法都叫正则化，但作用机制不同，不能一次全部打开后再猜是谁带来了变化。

## 2. 固定不变的阶段 11 协议

阶段 12 暂定使用 O3 + AMP 作为共同训练底座：

~~~text
model/data：沿用 p4、depth 6 的 MiniViT 与固定数据划分
optimizer：AdamW，lr=3e-4，weight_decay=0.05
scheduler：5-epoch linear warmup + cosine decay
gradient clipping：max_norm=1.0
precision：CUDA AMP
batch size：128
epochs：100
split_seed / train_seed：42 / 42
~~~

选择 O3 不是因为已经证明它稳定优于 O1/O2，而是因为三者 validation 表现接近，O3 数值略高且训练机制已经完整验证。AMP 作为执行方式固定开启，主要用于节省时间和显存。

## 3. 实验边界

- validation 和 test 始终使用原图与硬标签；
- Label Smoothing、Mixup、CutMix 只作用于训练过程；
- 每个正式单变量实验只启用一种高级正则化；
- 主要依据 validation 选择配置，不读取 test 调参；
- soft-label 训练下的 train accuracy 与普通 hard-label accuracy 含义不同，不能机械横向比较；
- 短跑筛选与 100-epoch 正式实验分开记录。

## 4. 计划中的实验矩阵

~~~text
R0：无高级正则化
R1：Label Smoothing
R2：Mixup
R3：CutMix
~~~

候选范围：

~~~text
label smoothing：0.05 / 0.10
mixup alpha：0.2 / 0.4
cutmix alpha：0.5 / 1.0
~~~

只有单项证据明确后，才考虑 R4：在每个 train batch 中随机选择 Mixup 或 CutMix。R4 不会提前混入 R0–R3。

## 5. 验收标准

- Label Smoothing 只改变训练 loss；
- Mixup 产生的软标签每行概率和为 1；
- CutMix 按真实裁剪面积重新计算 lambda；
- validation/test 不使用 batch 混合；
- 配置和 checkpoint 能准确恢复正则化策略；
- 正式实验之间只有预定的单一变量不同；
- 猜想与选择规则在查看正式结果前记录。

## 6. 当前微步

在学习 Label Smoothing 之前，先复习普通分类训练中的硬标签、one-hot 表示与交叉熵。只有先看清原始目标要求模型做什么，才能理解“平滑”究竟改了哪里。

## 7. 硬标签交叉熵

类别索引 `y` 在数学上等价于 one-hot 目标 (q)。模型输出 logits (z)，softmax 将它转成预测概率 (p)：

\[
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
\]

交叉熵为：

\[
L=-\sum_k q_k\log p_k=-\log p_y
\]

其 logit 梯度是：

\[
\frac{\partial L}{\partial z_k}=p_k-q_k
\]

硬标签持续把正确类别推向 1、错误类别推向 0。PyTorch 的 `CrossEntropyLoss` 直接接收 logits 和整数类别索引；不要先手动 softmax。

## 8. Label Smoothing 的机制

PyTorch 使用：

\[
q^{smooth}=(1-\varepsilon)q+\varepsilon/C
\]

其中 (C) 是类别数。CIFAR-10 在 `epsilon=0.1` 时，正确类别目标为 0.91，九个错误类别各为 0.01。梯度仍为 (p-q^{smooth})，所以它不是简单缩小 loss 或梯度，而是改变目标分布；当模型过度自信时，部分梯度可以反向，把预测从极端分布拉回。

平滑强度过小可能作用不明显，过大则会削弱监督信号。不同目标下的 train loss 绝对值不能直接横向判断模型优劣。

## 9. 训练与验证 loss 分离

原来的 `fit()` 把同一个 `criterion` 同时传给训练和验证。若直接启用 smoothing，validation loss 也会改变，导致 R0 与 R1 不再使用同一评价尺度。

现在调用链改为：

~~~text
train_criterion -> train_one_epoch()
val_criterion   -> evaluate()
~~~

训练入口目前先给二者都创建普通 `CrossEntropyLoss`，因此行为保持不变。后续只让 `train_criterion` 从 YAML 读取 `label_smoothing`；`val_criterion` 继续使用硬标签交叉熵。最终测试脚本本来就独立使用硬标签 loss。

新增契约测试用两个返回不同常数的 criterion，确认 train/validation loss 分别来自正确对象。相关训练引擎与入口测试结果：55 passed。

## 10. Label Smoothing 配置接入

新增 `src/regularization.py`，集中负责：

- 将 `training.label_smoothing` 校验并规范化为 `[0, 1)` 内的有限浮点数；
- 拒绝 bool、NaN、无穷、负数和 1.0；
- 创建带 smoothing 的训练交叉熵和硬标签验证交叉熵。

YAML 省略该项时默认使用 `0.0`，有效值会写回配置快照，因此 checkpoint 能记录实际训练目标。相关测试结果：68 passed。

## 11. Mixup 核心组件

`mixup_batch()` 对一个 batch 采样单个 `lambda ~ Beta(alpha, alpha)` 和一个随机排列，并返回：

~~~text
mixed images：[B, C, H, W]
soft targets：[B, num_classes]
lambda：实际混合比例
permutation：实际配对关系
~~~

图片和目标严格共用同一 lambda 与排列。组件不原地修改输入；软标签每行和为 1。测试还验证了软标签交叉熵等价于两个硬标签交叉熵的 lambda 加权和。

Beta 采样和 `randperm` 使用 PyTorch 随机数流：相同 seed 与相同调用顺序可复现整段序列，但同一次训练的不同 batch 仍会得到不同 lambda 和排列。纯组件测试结果：26 passed。

## 12. Soft Accuracy 与训练接入

硬标签 `[B]` 继续使用普通 top-1 accuracy；软标签 `[B, K]` 使用预测类别对应的目标概率质量：

\[
soft\ accuracy=\tilde y_{argmax(logits)}
\]

它等价于 Mixup 两个硬标签正确性的 lambda 加权值，只用于观察同一混合配置内部的训练趋势，不能与 R0 的硬标签 train accuracy 机械比较。模型选择仍依据统一的 validation accuracy。

训练引擎现在接受可选 `batch_mixer`，其输出只替换训练图片和训练 targets；`evaluate()` 与最终逐样本评估没有该入口，因此不会意外执行 Mixup。相关回归测试：94 passed。

## 13. Mixup 配置接入

YAML 使用：

~~~yaml
training:
  batch_mixing:
    name: mixup
    alpha: 0.2
~~~

省略时会规范化保存为 `{name: none}`；启用时保存策略名与有效 alpha，并构建只传给训练循环的 `Mixup` 对象。当前阶段强制 `label_smoothing > 0` 与 batch mixing 互斥，确保 R1/R2 是单变量实验。相关测试：90 passed。

## 14. CutMix 实现

CutMix 以 A 为底图，把配对图片 B 的同位置矩形粘贴进来。采样 lambda 决定期望面积，矩形边界采用右开区间 `[x1, x2) × [y1, y2)`；边界裁剪后使用：

\[
lambda_{actual}=1-\frac{(x_2-x_1)(y_2-y_1)}{HW}
\]

重新构造软标签。结果同时保存 sampled lambda、actual lambda 和边界框，便于复核。

固定 `H=6, W=8, sampled lambda=0.75, center=(1,5)` 的边缘测试得到实际框 `(0,4)-(3,6)`、面积 6、actual lambda 0.875，与手算一致。原 batch 不会被原地修改。YAML 使用 `name: cutmix` 与正 alpha 即可构建策略。相关测试：92 passed。

## 15. CutMix 训练级 Smoke Test

使用 O3 + AMP 底座运行 2 epochs，每轮限制 2 个 train/val batch。CutMix、soft-target loss、soft train accuracy、AMP、gradient clipping、warmup/cosine、history 与 checkpoint 均正常工作；无 NaN/Inf。

运行状态为 `completed`，`is_smoke_test=true`、`aggregation_eligible=false`。聚合器结果为 `eligible=6, skipped=6`，新增排除项的原因正是 smoke test，因此没有污染正式实验结果。短跑准确率只用于检查管线，不作模型效果结论。

## 16. 三种策略实现层总验收

Label Smoothing、Mixup、CutMix 均使用 O3 + AMP 底座完成 2 epochs × 2 train/val batches 的独立 CUDA smoke。三组状态均为 `completed`，配置快照分别准确保存 smoothing、Mixup alpha 与 CutMix alpha。

聚合器结果为 `eligible=6, skipped=8`；阶段 12 的三个新增 skipped 均因 `is_smoke_test=true` 被预期排除。加入 screening 运行类型与六组配置后，全量回归测试结果：340 passed。至此实现层完成，smoke 指标不用于选择超参数。

## 17. 参数筛选协议

六组 screening 使用完整 train/validation 数据、30 epochs、2-epoch warmup、O3 + AMP、固定 split/train seed 42。候选为：

~~~text
Label Smoothing：0.05 / 0.10
Mixup alpha：    0.2 / 0.4
CutMix alpha：   0.5 / 1.0
~~~

短跑只在同一种方法内部筛强度，不用于建立三种方法之间的最终排序。若一个候选的 best val 至少高 0.5 pp，且最后 5 轮平均 val 至少高 0.3 pp，则选择它；若证据冲突或差距不足，则选择更温和的预注册值 0.05 / 0.2 / 0.5。

配置使用 `experiment.run_type: screening`。状态文件只有 `run_type=formal` 才可能成为正式聚合候选；screening 即使完整、clean 也会被明确排除。协议和六组配置契约测试：62 passed。

## 18. Screening 前猜想

用户没有足够直觉对每一对强度给出明确排序。原因是每种方法都存在相反作用：强度过低可能正则化效果不明显，强度过高又可能削弱监督信号，使模型难以充分学习；仅凭机制无法确定当前小训练集上的最佳平衡点。

因此本轮不预言具体差距，将结果视为探索性筛选。若必须给出方向性猜测，则以较低置信度偏向强度较高的一方：

~~~text
Label Smoothing：0.10
Mixup alpha：    0.4
CutMix alpha：   1.0
~~~

该倾向不表示断言“越强越好”，只表示猜测当前小数据训练可能从更明显的正则化中获益。最终仍严格使用第 17 节预注册的双指标规则选择；证据不足时回退到较温和值。

## 19. Screening 结果与参数选择

六组实验均完成 30 epochs，状态为 `completed`，没有出现 NaN、Inf 或中途退出。总训练时间约 43.8 分钟。结果只用于同一种方法内部选强度：

| 方法 | 候选值 | best val | best epoch | 最后 5 轮 val 均值 |
|---|---:|---:|---:|---:|
| Label Smoothing | 0.05 | 80.38% | 30 | 79.992% |
| Label Smoothing | 0.10 | 80.70% | 29 | 80.288% |
| Mixup | 0.2 | 79.52% | 30 | 79.116% |
| Mixup | 0.4 | 79.06% | 28 | 78.444% |
| CutMix | 0.5 | 78.64% | 30 | 77.840% |
| CutMix | 1.0 | 77.28% | 29 | 76.784% |

按第 17 节事前规则，正式参数冻结为：

~~~text
R1 Label Smoothing：0.05
R2 Mixup alpha：    0.2
R3 CutMix alpha：   0.5
~~~

选择理由：

- Label Smoothing 0.10 的 best val 比 0.05 高 0.32 pp，最后 5 轮均值高 0.296 pp；两项都没有达到预注册的 `0.5 pp / 0.3 pp` 双阈值，因此回退到更温和的 0.05。
- Mixup 0.2 的 best val 和末 5 轮均值分别比 0.4 高 0.46 pp、0.672 pp，因此选择 0.2。
- CutMix 0.5 的 best val 和末 5 轮均值分别比 1.0 高 1.36 pp、1.056 pp，因此选择 0.5。

### 对 screening 前猜想的反思

低置信度猜想偏向三种较强设置，实际只有 Label Smoothing 0.10 略占优势，但证据仍不足以通过选择阈值；Mixup 0.4 和 CutMix 1.0 均弱于温和设置。说明“训练集较小，所以应使用更强正则化”只描述了正则化的潜在收益，却遗漏了过强策略会同时削弱有效监督信号：Mixup 会让整张图和标签长期变得模糊，CutMix 还会产生更剧烈的局部内容替换。正则化强度与泛化通常不是单调关系，需要通过固定协议下的验证数据寻找平衡点。

这轮猜想仍有价值：它在看结果之前暴露了自己的倾向，而预注册阈值阻止我们因 Label Smoothing 0.10 仅高 0.32 pp 就事后追逐小幅波动。

### 正式实验衔接

阶段 11 的 O3+AMP 已完成 100 epochs，且模型、数据、优化器、scheduler、clipping、AMP 与 seed 都和阶段 12 的固定底座一致，因此直接复用为 R0：best val 83.64%（epoch 93）。R1–R3 各自只加入一种已筛选的高级正则化，正式训练仍为 100 epochs、5-epoch warmup。

## 20. 正式实验前猜想

在运行 R1–R3 的 100-epoch 正式实验前，预测最佳验证准确率为：

~~~text
R1 > R2 > R3 > R0
~~~

也就是说，预测三种高级正则化都会优于无高级正则化的 R0；其中 Label Smoothing 0.05 表现最好，Mixup 0.2 次之，CutMix 0.5 再次之。

对泛化与曲线的预测：

- 后期 validation 曲线最稳定：R1；
- 最能缓解过拟合：R1；
- 最可能因正则化过强而欠拟合：R2 或 R3，目前无法明确区分；
- 最不确定的比较：R2 与 R3 的相对表现。

这一猜想背后的整体直觉是：R1 只软化监督目标，不破坏原始图像，可能在抑制过度自信的同时保留较完整的学习信号；R2/R3 会直接改变训练输入和标签，正则化作用可能更强，也更可能妨碍模型充分拟合。这里的机制解释只用于明确实验前判断，不作为结果成立的证据。

## 21. R1：Label Smoothing 正式结果

第一次 R1 虽完整运行 100 epochs，但启动时阶段 12 笔记与正式 YAML 尚未提交，因此状态记录为 `git_is_dirty=true`、`aggregation_eligible=false`。该结果保持原样，仅作为被协议排除的运行，不通过修改状态文件伪装成合格实验。

提交冻结配置后，使用相同 seed 和相同参数写入新目录完成 clean rerun：

| 项目 | 结果 |
|---|---:|
| status | completed |
| completed epochs | 100 / 100 |
| git commit | e3693f0 |
| git dirty | false |
| aggregation eligible | true |
| best val accuracy | 83.52% |
| best epoch | 95 |
| final val accuracy | 83.02% |
| last-5 val mean | 83.148% |
| total time | 27.15 min |

两次 R1 的每个 epoch 中，train/validation loss、train/validation accuracy、learning rate、gradient norm 与 clipping fraction 均逐项一致；差异仅来自耗时和显存等运行时记录。这说明固定数据划分、训练 seed 和确定的调用顺序在当前环境中实现了高度可复现的训练轨迹。

R1 的 best val 为 83.52%，比复用的 R0 83.64% 低 0.12 pp。差距很小，先记录事实，不在 R2/R3 完成前下最终结论。

## 22. R2：Mixup 正式结果

R2 使用 `alpha=0.2` 完成 100 epochs，配置快照确认 Mixup 只作用于训练 batch，validation 仍使用原图和硬标签。运行基于 clean commit `6ba1005`，可进入正式聚合。

| 项目 | 结果 |
|---|---:|
| status | completed |
| completed epochs | 100 / 100 |
| git dirty | false |
| aggregation eligible | true |
| best val accuracy | 84.56% |
| best epoch | 96 |
| final val accuracy | 83.72% |
| last-5 val mean | 83.968% |
| last-10 val mean | 84.050% |
| total time | 27.45 min |

R2 当前比 R0 的 best val 高 0.92 pp，比 R1 高 1.04 pp。最终 train accuracy 87.405% 是对软标签概率质量计算的 soft accuracy，定义与 R0/R1 的硬标签准确率不同，不能据此判断 Mixup 拟合更差或过拟合更少。最终比较等待 R3 完成后统一进行。
