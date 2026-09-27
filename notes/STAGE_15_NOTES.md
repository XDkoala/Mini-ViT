# 阶段 15：相机扰动与域偏移鲁棒性

本阶段冻结已训练模型，在 CIFAR-10 validation 图像上施加可控、可复现的相机退化，测量 clean accuracy 到 corrupted accuracy 的下降。阶段目标不是证明模型能覆盖真实世界全部域偏移，而是建立一个边界清楚、可重复的第一版鲁棒性协议。

开始日期：2026-09-27

当前状态：进行中

当前位置：deterministic corruption dataset 已实现并通过契约测试；下一步生成样例网格

## 进度

- [x] 明确鲁棒性问题与评估边界
- [x] 预注册 6 类扰动 × 3 个 severity
- [x] 实现 deterministic corruption dataset
- [x] 测试标签、顺序、数量与 seed 可复现性
- [ ] 生成样例网格并完成人工视觉验收
- [ ] 在 validation 上评估三个冻结模型
- [ ] 报告每项 accuracy、clean drop、macro average 与 worst case
- [ ] 冻结 corruption 协议后一次性评估 corrupted test
- [ ] 阶段 15 总验收

## 1. 我们在测什么

普通 validation 假设输入分布与训练数据接近。机器人相机可能遇到曝光变化、低对比度、传感器噪声、运动拖影、遮挡与视角偏移，使输入分布发生变化，而标签语义通常不变。

本阶段测量的是模型在指定合成扰动下的经验鲁棒性：

~~~text
accuracy_drop = clean_accuracy - corrupted_accuracy
~~~

drop 越小，表示在该扰动与强度下保留的性能越多。但只能对本协议覆盖的退化作结论，不能把合成扰动结果直接推广成“真实机器人环境中一定鲁棒”。

## 2. 扰动放在 normalization 之前

当前 evaluation pipeline 是：

~~~text
PIL / uint8 0–255 -> Tensor float32 [0,1] -> Normalize
~~~

所有相机扰动都在 `[0,1]` 像素空间完成，随后 clamp 到 `[0,1]`，最后使用 CIFAR-10 mean/std normalization。若直接给已归一化张量增加 0.1，不同通道会因 std 不同而对应不同的真实像素变化，强度将失去直观含义。

## 3. 预注册扰动矩阵

| Corruption | Severity 1 | Severity 2 | Severity 3 | 固定含义 |
|---|---:|---:|---:|---|
| brightness decrease | factor 0.8 | 0.6 | 0.4 | 模拟逐渐欠曝光 |
| contrast reduction | factor 0.75 | 0.50 | 0.25 | 围绕图像灰度均值压缩对比度 |
| Gaussian noise | std 0.04 | 0.08 | 0.12 | 独立加性像素噪声 |
| horizontal motion blur | kernel 3 | 5 | 7 | 水平方向均匀线性核，不称为 Gaussian blur |
| occlusion | square 6 | 10 | 14 px | 固定 seed 决定位置，黑色方块遮挡 |
| translation | 1 | 2 | 4 px | 固定方向平移，空白区域填黑 |

这些参数在查看模型结果前冻结。severity 表示参数幅度单调增强，不要求 accuracy 严格单调；任何非单调结果都原样报告。

## 4. 确定性与模型公平性

统一使用 `corruption_seed=1501`。Gaussian noise、occlusion 位置和 translation 方向若需要随机性，必须由“corruption seed + 原始 CIFAR 样本索引 + corruption 标识”建立局部随机数生成器。同一 corruption 的三个 severity 共用同一份基础随机样本，只改变噪声幅度、遮挡尺寸或位移距离；因此 severity 不参与随机 seed，避免强度变化时同时换掉噪声图案或方向。

因此同一 `(sample index, corruption, severity)` 永远产生相同图像，不依赖：

- DataLoader 访问顺序；
- worker 数量；
- 被哪个模型评估；
- 前面已经生成过多少张图片。

三个模型必须共享完全相同的 validation 索引、样本顺序和 corrupted inputs。不能为每个模型重新顺序采样一套噪声，否则模型差异会混入输入差异。

## 5. 数据隔离

- 开发、样例验收和第一轮比较只使用固定的 5,000 张 validation；
- corruption 不改变标签、样本数量或样本顺序；
- 模型参数全程冻结，使用 `eval()` 与 `inference_mode()`；
- 不读取 clean/corrupted test 调整 severity；
- 三模型与 corruption 参数冻结后，corrupted test 只运行一次并完整保存；
- test 无论有利或不利都不能反向修改扰动或模型配方。

## 6. 报告规则

对每个模型同时报告：

1. clean accuracy；
2. 每个 corruption × severity 的 accuracy；
3. 相对该模型 clean accuracy 的 drop；
4. 18 个 corrupted 条件的 macro average；
5. 每种 corruption 的 severity 平均；
6. worst-case accuracy 与对应条件。

不能只报告 macro average，因为两个平均值相同的模型可能分别害怕噪声和遮挡；机器人应用需要知道具体失效模式。

## 7. Deterministic Corruption Dataset 实现

`src/corruptions.py` 将原始 PIL/uint8 图像转为 `[0,1]` float Tensor，执行 corruption、clamp，最后才使用 CIFAR-10 mean/std normalization。clean 与 corrupted validation 因而共享同一预处理路径，区别只在 corruption 步骤。

随机键使用稳定的 BLAKE2b 摘要生成局部 `torch.Generator`，不依赖 Python 的进程随机 hash，也不消费全局 PyTorch RNG。相同样本与 corruption 的三个 severity 共用基础噪声、遮挡中心或平移方向，只改变预注册幅度。

契约测试覆盖：

- 6 类 × 3 severity 均保持 `[C,H,W]`、dtype 和 `[0,1]` 范围；
- 输入 Tensor 不被原地修改；
- 相同 key 输出逐像素相同；
- Gaussian noise 的 severity 1/2 使用同一噪声图案且幅度为 1:2；
- 遮挡面积严格为 `side²`；
- dataset 长度、索引顺序和标签不变；
- 改变访问顺序不会改变某个样本的输出；
- clean wrapper 只执行 normalization。

专项测试 32 passed，全量回归测试 383 passed。测试中的一次初始失败来自 FP32 两次运算末位舍入超过默认 `1e-8` 绝对容差；将契约设为 `atol=1e-7` 后通过，不涉及随机性或算法修复。
