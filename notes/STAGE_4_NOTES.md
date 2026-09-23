# 阶段 4：极小数据过拟合测试

本阶段让完整 MiniViT 反复学习 64 张固定训练图片，验证数据、前向传播、交叉熵、反向传播和优化器更新能够形成真正可学习的闭环。

开始日期：2026-09-19  
当前状态：已完成  
当前位置：阶段 4 总验收通过

## 阶段概览

### 阶段目标

```text
固定 64 张训练图片
  → 重复前向传播
  → CrossEntropyLoss
  → backward
  → optimizer.step
  → loss 明显下降
  → 训练准确率接近 100%
```

Tiny overfit 验证的是“模型和训练链能否学会”，不是泛化能力。

### 阶段边界

本阶段涉及：

- `tests/test_tiny_overfit.py`：可重复执行的极小数据过拟合检查。
- 必要时复用或小幅扩展 `src/data.py` 的数据接口。
- `notes/STAGE_4_NOTES.md`：记录机制、配置、曲线与验收结果。

本阶段不做：

- 正式训练集/验证集长时间训练。
- 测试集评估与模型选择。
- 数据增强、Dropout、weight decay 等泛化手段的调优。
- 根据 64 张图片的结果宣称模型具有泛化能力。

Tiny overfit 不通过时，不进入正式训练阶段。

## 进度总览

- [x] 理解 tiny overfit 的目的、能力边界与成功现象
- [x] 构造固定 64 张、无随机增强的训练子集
- [x] 建立模型、loss 与优化器
- [x] 实现并理解单个训练 step
- [x] 实现重复训练并记录 loss、accuracy
- [x] 达到接近完全记忆的验收标准
- [x] 固化为可重复执行的诊断测试
- [x] 阶段 4 总验收

## 符号与配置速查

| 名称 | 含义 | 初始方案 |
|---|---|---:|
| `S` | tiny subset 样本数 | 64 |
| `B` | 每次训练使用的样本数 | 64（整个 tiny subset） |
| `K` | CIFAR-10 类别数 | 10 |
| `dropout` | 随机丢弃概率 | 0 |
| `weight_decay` | 权重衰减 | 0 |
| `seed` | 子集随机种子 | 42 |
| `steps` | 最大参数更新次数 | 实验后记录 |
| `lr` | 学习率 | `1e-3` |

首轮诊断模型采用 `embed_dim=64`、`depth=2`、`num_heads=4`、`mlp_dim=128`；优化器采用 `AdamW(lr=1e-3, weight_decay=0)`。最终 step 数和验收指标在实际实验后填写。

## 1. Tiny Overfit 的诊断意义

阶段 3 已证明：

- 模型前向 shape 正确；
- CrossEntropyLoss 可以计算；
- 所有关键参数具有梯度。

这些仍不能证明参数更新后 loss 会持续下降。Tiny overfit 把任务缩小到模型理应能够记住的 64 张图片，用来联合检查：

```text
data → model → logits → loss → gradients → optimizer → new parameters
```

通过意味着基本学习闭环有效；不通过时应先排查实现和优化设置，而不是直接开始 100 epoch 正式训练。

它不能证明验证集或测试集表现，也不能证明模型学到了可泛化的视觉规律。

## 2. 固定 64 张训练子集

### 2.1 必须同时固定什么

- 固定训练/验证划分 seed。
- 从训练索引中固定选择同一组 64 个索引。
- 用无随机裁剪、无随机翻转的 `eval_transform` 读取这些训练图片。
- 固定模型初始化及训练随机种子。

现有 `build_datasets()` 返回的训练 Dataset 使用 `RandomCrop` 和 `RandomHorizontalFlip`，每次访问同一索引仍可能得到不同图像，不适合第一轮 tiny overfit。

正确思路是：

```text
CIFAR10 官方训练图片
  + eval_transform
  + 与阶段 1 相同的固定训练划分
  + 训练索引中的固定前 64 个
```

这样数据仍来自训练划分，但每次读取的 Tensor 完全一致。

### 2.2 待验收

- [x] 子集长度严格为 64。
- [x] 连续两次读取同一索引，图片 Tensor 和标签完全一致。
- [x] 图片为 `[3,32,32]`、`float32` 且数值有限。
- [x] 标签为 0–9 的整数类别。
- [x] 64 个索引全部属于训练划分，不与验证索引重叠。

2026-09-19 验收结果：`tests/test_tiny_overfit.py` 的数据契约测试 `4 passed`。子集使用 `seed=42` 对应的固定训练索引前 64 项，底层 CIFAR-10 训练图片配合无随机增强的 `eval_transform`。

## 3. 模型、Loss 与优化器

第一轮诊断关闭会增加记忆难度的正则化：

```text
dropout = 0
weight_decay = 0
随机数据增强 = 关闭
```

训练组件：

```text
model.train()
criterion = CrossEntropyLoss()
optimizer = AdamW(lr=1e-3, weight_decay=0)
```

首轮采用较小但结构完整的 MiniViT：

```text
patch_size=4, embed_dim=64, depth=2
num_heads=4, mlp_dim=128, dropout=0
```

- 小模型能降低诊断耗时，但仍覆盖完整 ViT 计算链。
- `CrossEntropyLoss` 直接接收原始 logits 与整数类别标签，模型末尾不先做 softmax。
- `AdamW` 为每个参数维护一阶、二阶动量，首轮用 `1e-3` 提高快速记忆的可行性。
- `weight_decay=0`、`dropout=0`：关闭妨碍记忆的正则化因素。
- `torch.manual_seed(42)`：固定模型初始化；有 CUDA 时同步固定 CUDA seed。

## 4. 单个训练 Step

固定顺序：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

需要同步计算：

```text
predictions = logits.argmax(dim=-1)
accuracy = correct / sample_count
```

关键机制：

1. 图片、标签和模型必须位于同一设备。
2. `zero_grad(set_to_none=True)` 清除上一步梯度，避免默认的梯度累加。
3. 前向传播得到 `[64, 10]` 的原始 logits，交叉熵据此计算标量 loss。
4. `loss.backward()` 沿计算图计算参数梯度，但此时尚未修改参数。
5. `optimizer.step()` 读取梯度和 AdamW 状态，真正更新参数。

2026-09-19 单步验收结果：`1 passed`。分类头梯度存在且全部有限，执行 `optimizer.step()` 后其权重与更新前不同，证明前向、loss、反向传播和参数更新已经连通。

## 5. 重复训练与成功标准

必须同时观察：

- loss 是否持续显著下降；
- 64 张固定图片的准确率是否持续接近 100%。

只看单次波动或只看其中一个指标不足以验收。具体 step、loss 和 accuracy 记录将在运行后写入。

### 5.1 十步短期趋势验收

固定初始化和固定 64 张图片，使用整个子集作为一个 batch。此时：

```text
1 step = 处理全部 64 张图片一次 = 1 epoch
```

2026-09-19 实测结果：

```text
step 1  loss=2.4313  accuracy=4.69%
step 10 loss=1.3596  accuracy=51.56%
```

测试结果：`1 passed`。10 步内 loss 持续下降，accuracy 总体上升，证明模型确实在学习这批固定样本。

accuracy 曾从 `39.06%` 短暂回落到 `37.50%`，但同期 loss 仍然下降。这并不矛盾：交叉熵会衡量全部类别 logits 的概率质量变化，而 accuracy 只记录最大 logit 对应的类别是否正确，因此后者更离散，也不保证逐步单调。

趋势测试只断言最终 loss 低于初始值、最终 accuracy 高于初始值，不要求每一步严格改善。下一步需要继续训练，检验能否接近完全记忆。

### 5.2 完整 Tiny Overfit 验收

最终使用 60 steps，并要求：

```text
final accuracy >= 98%
final loss < 0.1
```

对 64 张图片而言，`63 / 64 = 98.4375%`，所以准确率阈值实际只允许最多错一张。loss 阈值进一步排除“虽然分类正确，但只勉强跨过分类边界”的情况。

2026-09-19 实测：

```text
首次达到 100% accuracy：step 31
step 60 loss：0.021674
step 60 accuracy：100%
测试结果：1 passed
```

第 30 步 loss 出现过一次局部上升，accuracy 在首次达到 100% 后也曾短暂回落至 `98.4375%`，随后恢复并稳定在 100%。局部波动不影响总体收敛结论。

## 6. 失败排查顺序

1. 标签是否位于 0–9，dtype 是否为 `torch.int64`。
2. 模型输出是否为原始 logits，是否错误提前 softmax。
3. 是否按顺序调用 `zero_grad → forward → loss → backward → step`。
4. 模型是否处于 `train()` 模式。
5. 数据是否确实固定并关闭随机增强。
6. 学习率是否过小或过大。
7. 关键参数梯度是否存在且有限。
8. Attention softmax、残差连接和位置编码是否仍符合契约。

## 7. 阶段总验收

- [x] 64 张固定样本的训练 loss 显著下降。
- [x] 训练准确率达到接近 100%，或持续接近完全记忆。
- [x] 诊断可通过明确命令重复执行。
- [x] 实际配置、运行时长和最终指标已记录。
- [x] 项目其他自动化测试仍全部通过。

2026-09-19 最终验收：

```text
tests/test_tiny_overfit.py：6 passed
项目全套测试：90 passed
```

阶段 4 通过，可以进入阶段 5 的正式训练闭环。
