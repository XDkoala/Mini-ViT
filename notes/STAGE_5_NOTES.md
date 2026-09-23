# 阶段 5：正式训练闭环

本阶段把已经验证能够学习的 `MiniViT` 接入可复用的训练、验证、日志与 checkpoint 系统，使后续 MiniViT 和 CNN 能共享同一套实验流程。

开始日期：2026-09-19  
完成日期：2026-09-22

当前状态：已完成
当前位置：MiniViT 100 epochs 正式基线已完成并归档

## 阶段概览

### 阶段目标

```text
DataLoader
  → train_one_epoch
  → evaluate
  → fit
  → history.csv
  → last.pt / best.pt
  → 从 last.pt 恢复训练
```

### 阶段边界

本阶段负责训练与验证流程、指标累计、日志、模型保存和断点续训；不负责下载数据、定义模型结构、绘制图表或使用测试集选择模型。

## 进度总览

- [x] 理解训练引擎的职责与 epoch 指标累计方式
- [x] 实现并测试 `train_one_epoch()`
- [x] 实现并测试 `evaluate()`
- [x] 实现 `fit()`，串联训练和验证
- [x] 写入 `history.csv`
- [x] 保存 `last.pt` 与 `best.pt`
- [x] 恢复模型、优化器和训练进度
- [x] 完成单 batch 冒烟测试
- [x] 完成 2 epoch 冒烟测试
- [x] 完成正式 MiniViT 基线训练（100 epochs）
- [x] 阶段 5 总验收

正式长时间训练不直接开始。必须依次通过单 batch、2 epoch、断点续训三个关卡。

## 1. Epoch 指标累计

`ClassificationMetrics` 按样本数累计 batch loss、正确数和总样本数：

```text
total_loss += batch_mean_loss × batch_size
epoch_loss = total_loss / total_samples

total_correct += batch_correct
epoch_accuracy = total_correct / total_samples
```

不能直接平均各 batch 的 loss 或 accuracy，因为最后一个 batch 可能更小，否则每个 batch 而不是每个样本会获得相同权重。

`update()` 使用 `@torch.no_grad()`，指标统计不进入反向传播图；`compute()` 在尚无样本时主动报错；`reset()` 清除全部历史统计。

2026-09-20 验收结果：`tests/test_engine.py` 中 3 项测试通过，覆盖不等长 batch 加权、空状态报错和 reset 行为。

## 2. 单个训练 Epoch

`train_one_epoch()` 接收外部创建的模型、DataLoader、loss、优化器和设备，因此 MiniViT 与后续 CNN 可以复用同一训练逻辑。

每个 batch 的固定顺序：

```text
移动 images / labels 到 device
→ zero_grad(set_to_none=True)
→ forward
→ loss
→ backward
→ optimizer.step
→ 累计指标
```

`model.train()` 只切换 Dropout、BatchNorm 等模块的训练行为，不会自动计算梯度或更新参数；真正计算梯度与修改参数的分别是 `loss.backward()` 和 `optimizer.step()`。

2026-09-20 验收结果：训练引擎共 `4 passed`。精确测试使用两个 batch、3 个样本和可手算的线性分类器，确认模型被切回训练模式、参数发生更新，并返回正确的样本加权指标。

## 3. 验证循环

`evaluate()` 同时使用：

```text
model.eval()       → 切换 Dropout / BatchNorm 等模块的行为
torch.no_grad()    → 不记录反向传播计算图
```

两者不能互相替代。验证循环不接收 optimizer，也不执行 `zero_grad()`、`backward()` 或 `step()`；它只进行前向传播、loss 计算和指标累计。

2026-09-20 验收结果：训练引擎共 `5 passed`。探针模型确认每个验证 batch 都在 `training=False` 和梯度关闭状态下前向传播，参数及 `.grad` 保持不变，loss 与 accuracy 和手算结果一致。

## 4. 多 Epoch 调度

纯内存版 `fit()` 每轮严格执行：

```text
train_one_epoch → evaluate → 生成一条 epoch record
```

history 使用 `list[dict]`，每个字典对应未来 CSV 的一行，字段为 `epoch`、`train_loss`、`train_accuracy`、`val_loss`、`val_accuracy`。`epochs <= 0` 被视为配置错误并主动拒绝。

2026-09-20 验收结果：训练引擎共 `8 passed`。两轮测试确认模式及梯度状态按 `[train, eval, train, eval]` 切换，history 从 epoch 1 开始且字段完整。

## 5. History CSV

每完成一个 epoch，`fit()` 都把当前完整 history 重写到：

```text
output_dir/history.csv
```

相比训练结束后一次性写入，这能在意外中断时保留所有已完成 epoch；相比盲目追加，完整重写能避免重复表头和重复 epoch。CSV 使用固定字段顺序，Windows 下以 `newline=""` 打开文件，避免产生额外空行。

`output_dir=None` 时只返回内存 history，不产生文件，便于单元测试和快速调试。

2026-09-20 验收结果：训练引擎与文件工具共 `10 passed`。两轮 `fit()` 写出的 CSV 已从磁盘读回，并与返回的 history 逐项一致。

## 6. Checkpoint 保存

checkpoint 保存模型状态、优化器状态、已完成 epoch、历史最佳验证准确率、history 和 config。只保存模型参数可以用于推理，但不足以无缝恢复 AdamW 的一阶/二阶动量与 step。

```text
last.pt → 每轮覆盖，表示最近完成的训练状态，用于续训
best.pt → 仅在 val accuracy 严格刷新时覆盖，用于最终模型选择
```

验证准确率持平时不覆盖 `best.pt`，因此保留最早达到最佳值的模型。测试中两轮验证准确率均为 50%，最终 `last.pt` 属于 epoch 2，`best.pt` 属于 epoch 1，且两份模型权重确实不同。

2026-09-20 验收结果：训练引擎与工具共 `11 passed`，checkpoint 六类状态均已验证可读取且内容一致。

## 7. Checkpoint 加载

恢复顺序必须是：先按原配置创建模型，再用其参数创建 optimizer，最后分别加载模型与优化器 state dict。加载会原地修改现有对象，并返回 `epoch`、`best_val_accuracy`、history 和 config。

```text
checkpoint epoch = 37
→ 前 37 轮已完成
→ start_epoch = 38
```

当前目标是功能性续训；由于尚未保存所有随机数生成器状态，中断恢复后的随机增强和 shuffle 不保证与从未中断的运行逐位一致。

2026-09-20 验收结果：工具测试 `4 passed`。保存到全新对象后，模型权重和 AdamW 状态逐项一致，且恢复后的 optimizer 能继续更新目标模型参数。

## 8. 断点续训

`fit()` 将 `epochs` 解释为整个实验的目标总轮数，并接收 `start_epoch`、`initial_history` 与历史 `best_val_accuracy`。恢复时必须满足：

```text
history 最后一轮 = start_epoch - 1
```

否则会造成 epoch 重复或日志缺失。传入 history 时复制外层 list，后续追加不会修改调用者持有的旧列表。

2026-09-20 验收结果：训练引擎与工具共 `17 passed`。真实中断流程从 epoch 1 的 `last.pt` 恢复并续训到 epoch 3，内存 history、CSV 与最终 checkpoint 均连续为 `[1, 2, 3]`；四种非法恢复组合均被拒绝。

## 9. YAML 配置

正式 MiniViT 配置按 `experiment`、`data`、`model`、`training` 分组，并单独记录 seed 与 device。配置只声明代码当前真正支持的能力，暂不提前写入 AMP、scheduler 等尚未实现的选项。

`load_config()` 使用 `yaml.safe_load()`，检查根对象为 mapping、必需顶层键完整且四个分组均为 mapping。更具体的数值关系继续由 DataLoader、MiniViT 和训练入口验证。

2026-09-21 验收结果：配置测试 `6 passed`，覆盖真实基线 YAML、空文件、列表根、缺失字段、错误 section 类型及文件不存在。

## 10. 设备与随机种子

`resolve_device()` 支持 `auto`、`cpu`、`cuda`。`auto` 可在 CUDA 不可用时回退 CPU；显式请求 `cuda` 时若不可用则报错，避免静默偏离用户意图。

`set_seed()` 同时设置 Python、NumPy、PyTorch 与可用 CUDA 设备的随机数生成器。它提高同环境复现能力，但没有保存全部 RNG 状态，因此不能承诺跨设备或中断续训后的逐位一致。

2026-09-21 验收结果：工具测试 `13 passed`，覆盖设备分支、错误设备名、三套随机序列复现及 seed 合法范围。

## 11. 训练入口与单 Batch 冒烟

训练入口已连接 YAML、seed、device、DataLoader、MiniViT、AdamW、可选 checkpoint 恢复与 `fit()`。命令行支持配置路径、总 epoch 临时覆盖和恢复路径；隔离测试不读取真实数据即可验证组装参数。

2026-09-21 回归结果：项目全套测试 `129 passed`。

随后使用正式基线配置在 CUDA 上执行一个真实 CIFAR-10 batch：

```text
images shape： [128, 3, 32, 32]
labels shape： [128]
模型参数量：   2,693,578
loss：         2.4356
accuracy：     12.50%
```

初始十分类模型接近随机猜测水平是正常现象；该关卡验证的是正式数据、完整 MiniViT、交叉熵、反向传播与 AdamW 更新能够共同运行。

## 12. 受限 Batch 与 2 Epoch Smoke

训练和验证循环支持可选 batch 上限；正式训练默认 `None`，仍遍历完整 DataLoader。停止判断放在已处理 batch 末尾，确保不会为了判断上限而额外读取、随机增强下一批数据。

命令行使用 `--max-batches` 时必须同时显式指定 `--output-dir`，避免受限训练结果覆盖正式基线。checkpoint config 会记录 `is_smoke_test` 及训练/验证上限。

2026-09-22 实际运行：

```text
python -m scripts.train --epochs 2 --max-batches 2 \
  --output-dir outputs/stage5_smoke
```

结果：

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---:|---:|---:|---:|---:|
| 1 | 2.3951 | 11.33% | 2.3286 | 19.14% |
| 2 | 2.3472 | 16.80% | 2.1376 | 17.58% |

产物验收：

```text
history.csv：epoch [1, 2]
last.pt：epoch 2
best.pt：epoch 1
best_val_accuracy：19.14%
runtime：smoke=True，train/val 各 2 batches
```

这是工程冒烟结果，不代表完整训练集或验证集性能。`best.pt` 正确保留第 1 轮，因为第 2 轮验证准确率没有刷新历史最佳值。

## 13. 真实断点恢复 Smoke

从 `outputs/stage5_smoke/last.pt`（epoch 2）恢复，目标总轮数设为 3，并写入独立目录：

```text
python -m scripts.train --epochs 3 --max-batches 2 \
  --output-dir outputs/stage5_resume_smoke \
  --resume outputs/stage5_smoke/last.pt
```

验收结果：

```text
history epochs：[1, 2, 3]
last.pt：epoch 3
best.pt：epoch 3
epoch 3 train loss：1.8701
epoch 3 train accuracy：30.86%
epoch 3 val loss：1.9873
epoch 3 val accuracy：28.52%
AdamW step：6
```

前两轮各 2 个训练 batch，共 4 次参数更新；恢复后再执行 2 次，optimizer step 最终为 6，证明 AdamW 状态从 checkpoint 连续恢复而非重新初始化。第 3 轮验证准确率刷新历史最佳，因此 `best.pt` 正确更新到 epoch 3。

## 14. 工程闭环验收

两个真实 smoke 目录均具备完整实验四件套：

```text
config.yaml
history.csv
last.pt
best.pt
```

`config.yaml` 保存经过命令行覆盖后的有效设置，并明确记录 smoke 标记与 batch 上限。测试集没有传入 `fit()`，没有参与训练、验证或最佳模型选择。

2026-09-22 工程关卡验收：

```text
单 batch 正式配置训练：通过
2 epoch 受限 smoke：通过
真实 last.pt 恢复到 epoch 3：通过
AdamW step 连续性：4 → 6
项目全套自动化测试：139 passed
```

训练引擎、日志和断点恢复已经具备正式训练条件。按照基础工作流，下一步还需完成正式 100 epoch MiniViT 基线训练并记录结果，之后才能将阶段 5 标记完成并进入阶段 6。

## 15. 逐 Epoch 进度与耗时

正式入口调用 `fit(verbose=True)`。每轮结束后输出 epoch 进度、训练/验证 loss、训练/验证 accuracy、耗时及是否刷新最佳验证准确率；`flush=True` 保证长时间训练时日志立即显示。

耗时使用单调高精度的 `time.perf_counter()` 测量，定义为本轮训练与验证的计算时间，不包含保存 CSV 和 checkpoint 的磁盘时间。该值同时写入 `history.csv` 的 `epoch_time_seconds` 列，便于训练后比较速度。

## 16. MiniViT 正式基线结果

正式实验使用 `configs/minivit.yaml`，随机种子 42，训练 100 epochs。测试集没有参与训练、模型选择或本阶段结果统计。

```text
最佳验证准确率：81.84%（epoch 80）
最佳轮训练准确率：94.08%
最佳轮验证 loss：0.7255
最低验证 loss：0.5879（epoch 40，val accuracy 80.30%）
最终训练准确率：95.20%
最终验证准确率：81.36%
最终验证 loss：0.7793
总计算耗时：2,563.73 s（约 42 min 44 s）
平均每轮耗时：25.64 s
```

训练前 40 轮验证性能持续提升；之后训练准确率继续上升，而验证准确率约停留在 81%，验证 loss 从最低点逐渐升高，表明模型后期出现过拟合并对部分错误预测变得更自信。该结果作为未经 scheduler、Mixup、CutMix 或 Label Smoothing 改进的原始 MiniViT 基线保留，不覆盖重训。

归档目录：`outputs/minivit_baseline/`

```text
best.pt     epoch 80，后续最终测试应加载此文件
last.pt     epoch 100，用于保留最终训练状态
history.csv 完整 100 epochs 指标与耗时
config.yaml 本次实验的有效配置快照
```

SHA-256：

```text
best.pt     250E2CC425C7AA080DA37BE036C734D31548ACC6D563D7DAF4B6FF72AA6726A4
last.pt     DB264FF20AFD289C7D34548F1D293EB0299B019A0838EB016342C8C444D4021C
history.csv 6B39CEF579FD870C5DEFA4B7DBA20C867CD7B7A1675E7EEB36A318C1A0B7757D
config.yaml EA2CF20ED784D6DB7224B99A6491F2738B46278EE41F46500577E9036879C0F7
```

阶段 5 验收结论：训练闭环、逐轮日志、最佳/最终 checkpoint、恢复机制与正式 100 epochs 实验全部完成。测试准确率留到阶段 8，统一加载 `best.pt` 后评估一次。
