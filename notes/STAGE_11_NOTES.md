# 阶段 11：优化训练栈

本阶段在阶段 10 的可追溯实验协议上，加入线性 warmup、cosine decay、梯度范数记录、梯度裁剪与 AMP，并验证它们的更新顺序和续训连续性。

开始日期：2026-09-25

当前状态：进行中

当前位置：O0–O3 实验前猜想已冻结，等待 clean commit 后开始正式训练

## 进度

- [x] 建立阶段 11 学习地图和实现顺序
- [x] 理解学习率调度与 optimizer update 的关系
- [x] 定义并测试 warmup + cosine 数学曲线
- [x] 封装并测试有状态 scheduler
- [x] 将 scheduler 接入训练循环与 history
- [x] 将 scheduler 接入 YAML、训练入口与断点恢复
- [x] 实现并测试只读梯度总范数函数
- [x] 记录更新前的梯度总范数
- [x] 加入梯度裁剪与裁剪触发比例
- [x] 理解 AMP、autocast 与 GradScaler 的分工
- [x] 定义并测试 AMP 设备策略与 FP32 回退
- [x] 接入 AMP，并支持 CPU/禁用回退
- [x] checkpoint 保存并恢复 scheduler/scaler
- [x] 验证中断续训与连续训练的 LR 一致
- [x] 完成单 batch 与 2 epoch smoke
- [x] 建立 O0–O3 优化消融配置
- [x] 完成全量回归测试
- [ ] 阶段 11 总验收

## 1. 本阶段到底优化什么

四项能力看起来都属于“训练优化”，但它们解决的问题不同。

| 能力 | 主要目的 | 是否改变模型学习轨迹 |
|---|---|---|
| Linear Warmup | 避免训练初期直接使用峰值学习率 | 是 |
| Cosine Decay | 训练后期逐渐减小更新步幅 | 是 |
| Gradient Clipping | 限制异常大的梯度更新 | 触发时会改变 |
| AMP | 用混合精度减少显存并提高 GPU 吞吐 | 理论目标相同，但有数值差异 |

因此实验时不能一次把四项全部打开，然后把准确率变化归因给其中某一项。

## 2. 两条主线

### 2.1 学习策略

~~~text
warmup
cosine decay
gradient clipping
~~~

这些功能会主动改变每次参数更新的大小或方向，应通过 validation 做消融比较。

### 2.2 执行效率

~~~text
AMP / mixed precision
~~~

AMP 主要研究速度、显存和数值稳定性。它使用同一份已选定的训练配方做开关对照，不与 scheduler 消融混成一个变量。

## 3. 当前 FP32 基线更新顺序

当前 `train_one_epoch()` 的核心流程是：

~~~text
optimizer.zero_grad()
        ↓
model(images)
        ↓
loss = criterion(logits, labels)
        ↓
loss.backward()
        ↓
optimizer.step()
~~~

含义：

1. 清除上一批梯度；
2. 前向计算预测与损失；
3. 反向传播，把梯度写入每个参数的 `.grad`；
4. optimizer 根据梯度更新参数。

阶段 11 会在这条链上插入 scheduler、梯度统计、裁剪和 AMP。插入位置不正确时，代码即使能运行，数学语义也可能错误。

## 4. 阶段 11 的目标更新顺序

最终需要形成两条等价职责的路径。

### 4.1 FP32 路径

~~~text
zero_grad
→ forward
→ loss
→ backward
→ 统计 gradient norm
→ clip gradients（若启用）
→ optimizer.step
→ scheduler.step
~~~

### 4.2 AMP 路径

~~~text
zero_grad
→ autocast 中 forward + loss
→ scaler.scale(loss).backward()
→ scaler.unscale_(optimizer)
→ 统计 gradient norm
→ clip gradients（若启用）
→ scaler.step(optimizer)
→ scaler.update()
→ scheduler.step（仅对应成功的 optimizer update）
~~~

最关键的顺序约束是：梯度统计和裁剪必须发生在 `unscale_()` 之后。否则看到的是人为放大后的梯度，不是真实梯度。

## 5. 为什么先做 scheduler

scheduler 的数学关系和 step 次数最容易独立测试，也是后续 history 与 checkpoint 的时间轴基础。

本项目选择按成功的 optimizer update 调度，而不是按 epoch 调度：

~~~text
total_steps = epochs × steps_per_epoch
warmup_steps = warmup_epochs × steps_per_epoch
~~~

其优势是学习率变化与真实参数更新一一对应。同一 epoch 内不同 batch 可以拥有不同学习率，warmup 也更加平滑。

必须始终满足：

~~~text
scheduler step 数 == 成功的 optimizer update 数
~~~

若 AMP 因梯度溢出而跳过 optimizer update，scheduler 也不应假装参数已经更新。该边界会在 AMP 微步骤单独实现和测试。

## 6. 实现顺序

1. 先用纯数学函数定义每个 update 对应的学习率倍率；
2. 测试 warmup 单调上升、cosine 平滑下降及最小 LR 下界；
3. 将 scheduler 接入训练循环，并把实际 LR 写入 history；
4. 先观察梯度范数，再加入裁剪；
5. 接入 AMP，并保留明确的 FP32 路径；
6. 扩展 checkpoint 调用，保存并恢复 scheduler/scaler；
7. 对比连续训练与中断续训的 LR 序列；
8. 通过 smoke 后再建立正式消融。

## 7. 计划消融

| 编号 | 配方 | 要回答的问题 |
|---|---|---|
| O0 | AdamW + 固定 LR | 现有基线 |
| O1 | AdamW + cosine | 单独衰减是否改善验证表现 |
| O2 | AdamW + warmup + cosine | warmup 是否让训练初期更稳定 |
| O3 | O2 + gradient clipping | 裁剪是否抑制异常更新并改善泛化 |

AMP 在 O3 配方确定后单独比较开/关，主要观察：

- epoch 耗时；
- GPU 显存；
- 是否出现非有限 loss/gradient；
- validation 表现是否保持在合理波动范围。

## 8. 本阶段不做什么

- 不加入 Label Smoothing、Mixup 或 CutMix；它们属于阶段 12；
- 不使用 test accuracy 选择 O0–O3；
- 不把 smoke 准确率当成实验结果；
- 不在理解 scheduler step 语义前直接堆叠全部训练技巧。

## 9. 下一微步

下一步只讲学习率调度的基础：

1. 学习率在参数更新公式中处于什么位置；
2. 为什么固定学习率可能不够；
3. “按 batch step”和“按 epoch step”究竟有什么区别；
4. 我们如何给第 0 次、第 1 次和最后一次更新编号。

先统一时间轴，再写 warmup 与 cosine 公式。

## 10. Warmup 与 cosine 的数学契约

### 10.1 Update 时间轴

总更新次数为 `total_steps=T`，编号固定为：

~~~text
0, 1, ..., T - 1
~~~

一个编号对应一次成功的 optimizer update，也对应一个实际使用的学习率。不能额外生成编号 `T`，否则会多出一个永远不应被参数更新使用的学习率位置。

### 10.2 Linear warmup

设 warmup 步数为 `W`，起始倍率为 `start_factor`。当 `W>1` 且 `t<W`：

~~~text
progress = t / (W - 1)
factor = start_factor + progress × (1 - start_factor)
~~~

因此：

~~~text
t = 0      → factor = start_factor
t = W - 1  → factor = 1，也就是 peak LR
~~~

边界定义：

- `W=0`：关闭 warmup；
- `W=1`：唯一 warmup 更新直接使用峰值；
- `W>=T`：拒绝，因为组合曲线将没有 cosine 阶段。

### 10.3 Cosine decay

对进度 `p∈[0,1]`：

~~~text
cosine_weight = 0.5 × (1 + cos(πp))
factor = min_factor + (1 - min_factor) × cosine_weight
~~~

性质：

~~~text
p = 0 → factor = 1
p = 1 → factor = min_factor
~~~

无 warmup 时，cosine 自己包含峰值与最小值：

~~~text
p = t / (T - 1)
~~~

有 warmup 时，warmup 最后一步已经使用峰值。为了不重复峰值，剩余 `C=T-W` 个 cosine 更新使用：

~~~text
cosine_index = t - W
p = (cosine_index + 1) / C
~~~

所以第一步 cosine 已开始下降，最后一步精确到达最小倍率。

### 10.4 为什么返回倍率

实现位于 `src/schedulers.py`：

~~~python
compute_warmup_cosine_multiplier(...)
~~~

它返回相对于峰值学习率的倍率，而不是绝对学习率：

~~~text
actual_lr = base_lr × multiplier
~~~

例如：

~~~text
base_lr = 0.0003
multiplier = 0.4
actual_lr = 0.00012
~~~

这样做保留了 optimizer 不同参数组之间的比例。若两个参数组峰值分别为 `0.0003` 和 `0.00003`，同乘 `0.4` 后仍保持十倍关系。

### 10.5 为什么先写纯函数

纯函数：

- 不修改 optimizer；
- 不保存内部 step；
- 不依赖 GPU；
- 相同输入始终返回相同结果。

因此可以独立验证数学公式。若曲线错误，问题一定在参数或公式；不会与 PyTorch scheduler 的内部计数、checkpoint 或调用顺序混在一起。

### 10.6 输入验证

当前函数明确拒绝：

- bool、浮点数形式的 step；
- `total_steps<=0`；
- 超出 `[0,total_steps)` 的 update index；
- 负 warmup 或 `warmup_steps>=total_steps`；
- 超出 `[0,1]` 的 start/min factor。

Python 中 `bool` 是 `int` 的子类，因此必须单独拒绝；否则 YAML 中误写的 `true` 可能被当成数字 1。

### 10.7 测试结果

2026-09-25：

~~~text
tests/test_schedulers.py：15 passed
项目全量测试：242 passed
~~~

测试锁定：

- warmup 首尾端点；
- warmup 单调上升；
- cosine 不重复峰值并单调下降；
- 最后一步精确等于最小倍率；
- 无 warmup 时 cosine 自己包含两个端点；
- 单步训练和单步 warmup 的明确含义；
- 全部倍率不越界；
- 非法输入尽早失败。

Windows `.pytest_cache` 的已存在警告不影响结果。

## 11. 下一微步

下一步把纯函数封装成有状态 scheduler，重点不是重复公式，而是理解三个状态：

~~~text
base_lrs          每个参数组的峰值学习率
completed_updates 已成功完成多少次参数更新
current_lrs       下一次 optimizer update 将使用什么学习率
~~~

我们会先确定 scheduler 初始化时如何给第 0 次更新设置 LR，以及 `optimizer.step()` 后的 `scheduler.step()` 为什么是在准备下一次学习率。

## 12. 有状态 WarmupCosineScheduler

### 12.1 三种状态

`WarmupCosineScheduler` 保存：

~~~text
base_lrs
  每个 optimizer 参数组配置的峰值学习率

completed_updates
  已经成功完成多少次 optimizer update

optimizer.param_groups[*]["lr"]
  下一次 optimizer update 将实际使用的学习率
~~~

“已经完成多少次”和“下一次要使用哪个编号”在训练未结束时数值相同。例如：

~~~text
completed_updates = 4
→ update 0、1、2、3 已完成
→ optimizer 当前 LR 为 update 4 准备
~~~

### 12.2 初始化为何立即设置 update 0

构造 scheduler 前，optimizer 中保存的是峰值 LR。若等第一次 `scheduler.step()` 才降低学习率，那么第一次参数更新会错误地直接使用峰值，warmup 的第 0 个值永远不会被使用。

因此构造顺序为：

~~~text
读取 optimizer 中的 base_lrs
→ 计算 update 0 的 multiplier
→ 立即写回各参数组 lr
→ completed_updates = 0
~~~

此时尚未发生参数更新，只是为第一次更新做好准备。

### 12.3 每次 step 的语义

调用顺序：

~~~text
读取当前 LR
→ optimizer.step() 成功更新参数
→ scheduler.step()
~~~

`scheduler.step()` 做两件事：

1. `completed_updates += 1`，承认刚才那次更新已经完成；
2. 若训练尚未结束，为下一个 update index 设置 LR。

最后一次参数更新后仍调用一次 `scheduler.step()`，使计数最终等于 `total_steps`；但不再计算不存在的下一次 LR。

如果之后又多调用一次，scheduler 会明确报错。这样能尽早发现“scheduler 比 optimizer 多走一步”的时间轴错误。

### 12.4 多参数组为何使用倍率

若 optimizer 有两个参数组：

~~~text
group 0 base_lr = 0.00030
group 1 base_lr = 0.00003
~~~

当倍率为 `0.1`：

~~~text
group 0 current_lr = 0.000030
group 1 current_lr = 0.000003
~~~

两组仍保持十倍比例。scheduler 不会把它们粗暴改成同一个绝对学习率。

### 12.5 Scheduler 无法自行判断 optimizer 是否成功

普通 Python 调用中，scheduler 无法知道前面的 optimizer 是否真的修改了参数。因此它的 `step()` 契约是：

~~~text
只有成功 optimizer update 后，调用者才允许 scheduler.step()
~~~

FP32 中一般每次 `optimizer.step()` 都成功。AMP 可能因为梯度溢出跳过更新，届时必须由训练循环检测是否跳过，再决定 scheduler 是否前进。

### 12.6 state_dict 保存什么

~~~text
total_steps
warmup_steps
start_factor
min_factor
base_lrs
completed_updates
~~~

加载时不仅恢复计数，还会重新计算并写入“下一次更新应使用的 LR”。这比只保存 `completed_updates` 更容易验证恢复后的即时状态。

加载器要求当前配置与保存配置一致。例如用 `total_steps=9` 构造的新 scheduler 不能静默加载 `total_steps=8` 的状态，否则同一个 update index 会对应不同曲线位置。

### 12.7 Factor 验证补强

纯函数现在要求 `start_factor` 和 `min_factor`：

- 是 int/float，但不能是 bool；
- 是有限数，拒绝 NaN 和正负无穷；
- 位于闭区间 `[0,1]`。

### 12.8 测试结果

2026-09-25：

~~~text
tests/test_schedulers.py：19 passed
项目全量测试：246 passed
~~~

新增测试证明：

- 初始化立即准备 update 0；
- 多参数组的 LR 比例保持不变；
- 每个 step 精确准备下一个曲线位置；
- 总 step 数不会超过计划更新数；
- state round-trip 恢复相同的下一步 LR；
- 恢复后继续 step 仍与不中断轨迹一致；
- 不匹配的调度配置会被拒绝。

Windows `.pytest_cache` 的已存在警告不影响结果。

## 13. 下一微步

下一步把 scheduler 接入 `train_one_epoch()` 和 `fit()`。在动代码前先明确两个记录问题：

1. 每个 batch 应在什么时候读取“本次实际使用的 LR”；
2. history 每个 epoch 只有一个 `learning_rate` 单元格，应该保存首个、末个还是平均 LR。

本项目将记录该 epoch 最后一次成功参数更新实际使用的 LR，因为它能直接表示训练在 epoch 边界处走到曲线的哪个位置；完整曲线仍由配置和 update 公式精确重建。

## 14. Scheduler 训练循环接线

### 14.1 每个 batch 的顺序

当前 FP32 更新链变为：

~~~text
zero_grad
→ forward
→ loss
→ backward
→ 读取 optimizer 当前 LR
→ optimizer.step
→ scheduler.step（若启用）
~~~

LR 必须在 `optimizer.step()` 前读取。原因是当前 optimizer 中的 LR 属于本次更新，而随后的 `scheduler.step()` 会把它改成下一次更新的值。

若把读取放在 scheduler 后，history 记录的会是“下一 batch 将使用的 LR”，与本轮最后一次真实更新错位一格。

### 14.2 Epoch 记录哪个 LR

一个 epoch 内可能有许多不同 LR，但当前 history 每轮只有一个 `learning_rate` 单元格。本项目记录：

~~~text
该 epoch 最后一次成功 optimizer update 实际使用的第一个参数组 LR
~~~

这样每个 epoch 的记录表示调度曲线在 epoch 边界走到的位置。若将来需要逐 update 分析，可从有效配置与 scheduler 公式重建完整序列，或新增 batch 级日志，而不把多个含义塞进一个字段。

第一参数组作为主 LR；其他参数组通过共同 multiplier 保持相对比例。

### 14.3 固定 LR 也会记录

即使 `scheduler=None`，训练循环仍读取 optimizer 当前 LR。因此 O0 固定学习率基线也会在每个 epoch 明确记录同一个值，而不是留下空白。

历史阶段 0–10 的旧 CSV 不会被改写，其可选 LR 列可以缺失或为空；新运行开始产生真实值。

### 14.4 为什么 scheduler 是可选参数

~~~python
train_one_epoch(..., scheduler=None)
fit(..., scheduler=None)
~~~

这保留了现有固定 LR 行为，使 O0 与历史入口不需要伪造 scheduler。提供 scheduler 时，训练函数还会验证它管理的正是同一个 optimizer，避免一个 scheduler 修改 A optimizer、训练却更新 B optimizer。

### 14.5 Fit 与 checkpoint

`fit()` 将 scheduler 继续传给：

- 每轮 `train_one_epoch()`；
- `last.pt` 保存；
- 验证表现刷新时的 `best.pt` 保存。

因此 checkpoint 中的 scheduler 状态与模型、optimizer、history 属于同一个 epoch 边界。`best.pt` 保留最佳 epoch 当时的调度位置，`last.pt` 保留最近 epoch 的位置。

### 14.6 终端输出

逐 epoch 输出新增：

~~~text
lr=<该轮最后实际使用的学习率>
~~~

它与 history 使用完全相同的值，避免终端显示与持久化记录采用不同定义。

### 14.7 测试结果

2026-09-25：

~~~text
tests/test_engine.py：26 passed
项目全量测试：248 passed
~~~

测试证明：

- scheduler 每个训练 batch 前进一次；
- epoch 返回最后实际使用的 LR，而非下一步 LR；
- 固定 LR 路径记录稳定值；
- history 与 CSV 保存 learning_rate；
- checkpoint 保存对应的 completed_updates；
- evaluate 保持纯评估，不接触训练 LR 状态；
- 原有续训、batch 限制和模型选择行为不变。

Windows `.pytest_cache` 的已存在警告不影响结果。

## 15. 下一微步

下一步建立 scheduler 配置并接入 `scripts/train.py`：

1. 定义 `training.scheduler` 的关闭与 warmup-cosine schema；
2. 根据真实 `len(train_loader)` 和 `max_train_batches` 计算每轮有效更新数；
3. 创建 scheduler 后再处理 checkpoint 恢复；
4. 将 scheduler 传给 `load_checkpoint()` 与 `fit()`；
5. 保证旧配置没有 scheduler 字段时仍走固定 LR。

## 16. Scheduler 配置与训练入口

### 16.1 YAML 契约

启用 warmup + cosine：

~~~yaml
training:
  epochs: 100
  learning_rate: 0.0003
  scheduler:
    name: warmup_cosine
    warmup_epochs: 5
    start_factor: 0.1
    min_factor: 0.1
~~~

含义：峰值 LR 仍由 `training.learning_rate` 定义；第一次更新使用其 `0.1` 倍，前 5 个 epoch 线性升温，之后用 cosine 下降，最后一次更新到峰值的 `0.1` 倍。

以下两种写法都保留固定学习率：

~~~yaml
# 兼容旧配置：完全没有 scheduler 字段
~~~

~~~yaml
scheduler:
  name: none
~~~

### 16.2 Epoch 必须换算成 Update

配置对人使用 epoch 更直观，但 scheduler 按 optimizer update 工作：

~~~text
steps_per_epoch = min(len(train_loader), max_train_batches)
total_steps = epochs × steps_per_epoch
warmup_steps = warmup_epochs × steps_per_epoch
~~~

若没有 `max_train_batches`，每轮使用完整的 `len(train_loader)`。这样 smoke run 的短时间轴与实际执行的 batch 数一致，不会创建一条大部分永远走不到的曲线。

### 16.3 创建与恢复顺序

入口顺序固定为：

~~~text
创建 optimizer
→ 用 optimizer 的峰值 LR 创建 scheduler
→ 加载 checkpoint，同时恢复 model / optimizer / scheduler
→ 把同一个 scheduler 传给 fit
~~~

scheduler 必须先创建，`load_checkpoint()` 才有对象可以接收状态；又必须晚于 optimizer 创建，因为 scheduler 需要读取其参数组与基础 LR。

恢复时采取严格配对：若当前运行启用了 scheduler，而旧 checkpoint 没有 `scheduler_state`，程序会拒绝静默续训。否则模型和 optimizer 已走到中途，学习率曲线却从第 0 步重启，表面上叫“resume”，实际训练轨迹已经断裂。固定 LR 的旧配置与旧 checkpoint 仍可正常恢复。

### 16.4 配置验证

入口会尽早拒绝：

- `training.scheduler` 不是映射；
- 未知 scheduler 名称；
- 缺少 warmup/factor 字段；
- `epochs` 不是正整数；
- `warmup_epochs` 不在 `[0, epochs)`；
- train loader 没有有效 batch。

factor 的数值范围与有限性继续由 `WarmupCosineScheduler` 自身验证，避免入口与核心实现维护两套不一致规则。

### 16.5 测试结果

2026-09-25 定向测试：

~~~text
tests/test_train_script.py
tests/test_utils.py
tests/test_engine.py
tests/test_schedulers.py

92 passed
~~~

测试覆盖 scheduler 工厂、有效 update 数、关闭兼容、非法配置、入口传递、checkpoint 严格配对以及训练循环时序。

## 17. 下一微步

下一步先不裁剪梯度，只观察并记录更新前的梯度总范数：

1. 什么是单个参数张量的梯度范数；
2. 如何把全模型的许多 `.grad` 合成为一个总范数；
3. 为什么必须在 `backward()` 后、`optimizer.step()` 前测量；
4. 梯度范数能告诉我们什么，又不能告诉我们什么。

先获得诊断仪表，再决定何时以及如何裁剪。

## 18. 只读梯度总范数

### 18.1 数学定义

把所有已有参数梯度看成一个长向量，其全局 L2 范数为：

~~~text
total_norm = sqrt(sum(||grad_p||²))
~~~

例如两个参数张量的梯度分别为 `[3, 4]` 与 `[0, 12]`：

~~~text
total_norm = sqrt(3² + 4² + 0² + 12²) = 13
~~~

实现不需要真的拼接所有梯度。先求每个梯度张量的 L2 范数，再对这些范数求一次 L2，数学上完全等价，同时避免创建巨大的临时向量。

### 18.2 函数契约

~~~python
compute_gradient_norm(model.parameters())
~~~

函数位于 `src/gradients.py`，只读取 `.grad` 并返回 Python `float`：

- `grad is None`：参数被冻结或未参与当前计算图，忽略；
- 没有任何梯度：返回 `0.0`；
- FP16/BF16 梯度：转成 FP32 后求范数，降低平方时溢出或下溢风险；
- 稀疏梯度：只统计实际存储的非零 values；
- 出现 NaN/Inf：抛出 `FloatingPointError`，不让异常静默进入日志；
- 不缩放、不清空、不裁剪原始梯度。

`detach()` 的含义不是删除原梯度，而是明确表示统计操作不应进入 autograd 计算图。

### 18.3 与参数更新量的区别

梯度总范数描述 `backward()` 产生的信号规模，不等于参数实际移动距离。AdamW 还会使用动量、二阶矩估计、学习率和 weight decay。因此它是诊断仪表，不能单凭“越大/越小”判断训练好坏。

### 18.4 测试结果

2026-09-25：

~~~text
tests/test_gradients.py：7 passed
~~~

测试锁定全局 L2 公式、generator 输入、无梯度参数、空梯度、只读性、NaN/Inf 与稀疏梯度行为。

## 19. 下一微步

下一步将这个只读函数接到 `train_one_epoch()`：

1. 在 `backward()` 后、`optimizer.step()` 前测量；
2. 明确一个 epoch 有多个 batch 时 history 记录什么；
3. 当前先记录每轮 batch gradient norm 的平均值；
4. 仍不启用裁剪，保证这一微步只增加观测、不改变训练轨迹。

## 20. 梯度范数训练接线

### 20.1 Batch 内时序

每个训练 batch 现在执行：

~~~text
zero_grad
→ forward
→ loss
→ backward
→ compute_gradient_norm
→ 读取本次 LR
→ optimizer.step
→ scheduler.step
~~~

测量必须位于 `backward()` 后，因为此前新梯度尚未产生；必须位于 `optimizer.step()` 前，因为我们要记录本次更新准备使用的梯度。当前只读取梯度，参数更新结果与未加入监测时相同。

### 20.2 Epoch 为什么记录 batch 平均值

一轮有多个梯度范数，history 的 `gradient_norm` 定义为：

~~~text
sum(batch_gradient_norms) / optimizer_update_count
~~~

这是按 batch/update 次数做算术平均，不按样本数加权。原因是梯度范数描述的是一次参数更新事件；最后一个较小 batch 仍然产生一次真实更新。

这与 loss/accuracy 不同：loss 和准确率要描述整轮样本，因此必须按样本数加权。两个平均方式回答的问题不同。

平均梯度范数描述典型规模，但可能掩盖少数尖峰。后续加入 clipping 时，会另行记录裁剪触发比例；不把“平均水平”和“极端事件”混进同一个字段。

### 20.3 持久化与输出

每轮的平均梯度范数会写入：

- 内存 history 的 `gradient_norm`；
- `history.csv`；
- `last.pt` 与 `best.pt` 内保存的 history；
- 终端逐 epoch 输出 `grad_norm=...`。

### 20.4 测试结果

2026-09-25：

~~~text
相关定向测试：60 passed
项目全量测试：265 passed
~~~

测试用事件序列锁定：

~~~text
measure → optimizer.step → measure → optimizer.step
~~~

并验证两批模拟范数 `2.0`、`4.0` 的 epoch 结果为 `3.0`，以及 history、CSV 和终端输出保持一致。

## 21. 下一微步

下一步学习梯度裁剪：

1. `max_norm` 究竟限制什么；
2. 全局范数超过阈值时如何按同一比例缩放全部梯度；
3. 为什么裁剪改变大小但保持全局梯度方向；
4. 为什么要记录裁剪前范数和裁剪触发比例；
5. 裁剪为何是保险丝，而不是修复所有训练问题的万能方法。

## 22. 全局梯度范数裁剪

### 22.1 配置语义

启用：

~~~yaml
training:
  gradient_clip_norm: 1.0
~~~

缺少字段或写为 `null` 表示关闭。阈值必须是有限正数；bool、字符串、零、负数、NaN 与无穷都会在训练前被拒绝。

### 22.2 全局统一缩放

若原始全局范数为 `G`，阈值为 `M` 且 `G>M`，所有参数梯度使用同一个近似倍率：

~~~text
scale = M / G
grad_p = grad_p × scale
~~~

因此裁剪改变完整梯度向量的长度，但保持方向。它不同于逐元素 clamp；后者会以不同方式截断各分量，可能旋转梯度方向。

实现使用 `torch.nn.utils.clip_grad_norm_()`。末尾 `_` 表示它会原地修改 `.grad`，其返回值是裁剪前的全局范数。因此启用裁剪时无需先独立求一次 norm，再让 PyTorch 重复扫描梯度。

### 22.3 训练顺序

~~~text
backward
→ clip_grad_norm_（返回并记录裁剪前 norm）
→ optimizer.step
→ scheduler.step
~~~

AdamW 读取的是裁剪后的梯度，所以裁剪会影响当前更新及后续动量状态，属于真正改变学习轨迹的实验变量。

### 22.4 触发比例

~~~text
gradient_clip_fraction
= 裁剪触发的 update 数 / 本 epoch 总 update 数
~~~

history 保留两种不同状态：

- 字段为空：裁剪未启用；
- `0.0`：裁剪已启用，但本轮没有梯度超过阈值。

大于零时记录真实触发比例，终端以百分比输出。平均 `gradient_norm` 始终保存裁剪前的范数，因此仍能观察模型自然产生的梯度规模。

### 22.5 测试结果

2026-09-25：

~~~text
入口到训练循环定向测试：68 passed
项目全量测试：278 passed
~~~

测试覆盖阈值验证、`13→2` 的全局缩放、方向保持、阈值内不修改、裁剪前时序、触发比例、关闭语义以及 YAML 到 `fit()` 的完整接线。

## 23. 下一微步

下一步先理解 AMP，不立即实现：

1. FP32、FP16 与 BF16 在表示范围和精度上的区别；
2. `autocast` 为什么只让适合的运算降精度；
3. 小梯度为何可能在 FP16 中下溢为零；
4. `GradScaler` 为什么放大 loss，以及它如何判断是否跳过更新；
5. 为什么必须先 `unscale_()`，再统计或裁剪梯度。

## 24. AMP 运行时策略

### 24.1 三个不同概念

`src/amp.py` 中的 `AmpRuntime` 同时保存：

~~~text
requested
  用户是否在配置中请求 AMP

enabled
  当前设备上是否真的执行 AMP

fallback_reason
  请求与实际不一致时的明确原因
~~~

因此 CPU 回退可以表示为：

~~~text
requested = true
enabled = false
fallback_reason = "...resolved device is cpu..."
~~~

不能只记录配置中的 `amp: true`，否则实验报告会把实际 FP32 运行误写成 AMP 运行。

### 24.2 本项目的设备策略

~~~text
amp: false 或缺省
→ 保持原 FP32 路径，不创建 scaler

amp: true + CUDA
→ FP16 autocast + CUDA GradScaler

amp: true + CPU
→ 明确回退 FP32，保存回退原因
~~~

虽然 PyTorch 支持部分 CPU BF16 autocast，但它与 CUDA FP16 在数值范围、是否需要 loss scaling 和性能目标上不同。阶段 11 的 AMP 对照固定研究 CUDA FP16，避免同一个 `amp: true` 在不同设备上代表不同实验。

### 24.3 为什么封装 context

训练循环以后只需：

~~~python
with amp_runtime.autocast():
    logits = model(images)
    loss = criterion(logits, labels)
~~~

当 AMP 关闭或回退时，这个 context 不改变运算精度；启用时才让 PyTorch 自动选择适合 FP16 的算子。训练循环不需要散落多处分支判断。

### 24.4 Metadata

`to_metadata()` 返回可写入 YAML/JSON 的字段：

~~~text
requested
enabled
device_type
dtype
uses_grad_scaler
fallback_reason
~~~

这里只保存字符串、布尔值和 null，不把 `torch.dtype` 或 scaler 对象直接塞进配置文件。

### 24.5 当前环境与测试

2026-09-25 当前环境：

~~~text
PyTorch 2.14.0+cu130
CUDA available: true
~~~

验收结果：

~~~text
tests/test_amp.py：9 passed
项目全量测试：287 passed
~~~

测试覆盖关闭 FP32、CPU 显式回退、CUDA FP16 + scaler、可序列化 metadata、非法布尔配置与内部不一致的 CUDA 请求。

## 25. 下一微步

下一步将 `AmpRuntime` 接入 `train_one_epoch()`：

1. 只把 forward 与 loss 放入 autocast；
2. FP32 保留普通 `loss.backward()` 与 `optimizer.step()`；
3. AMP 使用 scaled backward 与 `unscale_()`；
4. unscale 后再测量或裁剪梯度；
5. 判断 scaler 是否跳过 optimizer update；
6. 只有成功更新时才推进 scheduler。

## 26. AMP 训练循环接线

### 26.1 两条执行路径

FP32 或 AMP 回退路径保持：

~~~text
forward + loss
→ loss.backward
→ gradient norm / clipping
→ optimizer.step
→ scheduler.step
~~~

CUDA AMP 路径为：

~~~text
autocast(forward + loss)
→ scaler.scale(loss).backward
→ scaler.unscale_(optimizer)
→ gradient norm / clipping
→ scaler.step(optimizer)
→ scaler.update
→ 仅成功更新时 scheduler.step
~~~

autocast 只包住 forward 与 loss。数据搬运、backward、梯度统计和 optimizer 控制不放进 autocast context。

### 26.2 如何判断 AMP 是否真的更新

AdamW/SGD 的 `optimizer.step()` 通常返回 `None`，因此不能用 `scaler.step()` 的返回值判断是否跳过。

当前运行时比较 `scaler.update()` 前后的 scale：

~~~text
scale_after < scale_before
→ 检测到 Inf/NaN，optimizer update 被跳过

scale_after >= scale_before
→ update 成功；scale 保持或按 growth 规则增大
~~~

只有第二种情况才推进 scheduler，从而保持：

~~~text
scheduler completed_updates
== 成功 optimizer update 数
~~~

### 26.3 非有限梯度的不同处理

FP32 发现 NaN/Inf 时仍立即抛错，因为没有 scaler 帮助恢复。

AMP 发现非有限梯度时：

1. 不把该范数计入 epoch 的平均梯度范数；
2. 不执行梯度裁剪，避免 `Inf × 0` 产生更多 NaN；
3. 交给 GradScaler 跳过 optimizer update 并降低 scale；
4. 不推进 scheduler；
5. 下一 batch 的 `zero_grad(set_to_none=True)` 清除异常梯度。

如果整轮没有任何成功 update，则明确报错，而不是伪造 LR 或梯度平均值。

### 26.4 Checkpoint 与有效配置

`last.pt` 和 `best.pt` 现在会随 scheduler 一起保存实际 GradScaler 状态。resume 时入口把同一个 scaler 交给 `load_checkpoint()`，避免 loss scale 从初始值重新开始。

有效配置的 `runtime.amp` 记录 requested、enabled、dtype、是否使用 scaler 与 fallback reason。CPU 回退会在终端明确打印原因。

### 26.5 测试结果

2026-09-25：

~~~text
AMP 真实 CUDA 与跳步测试：13 passed
AMP 全链路定向测试：106 passed
项目全量测试：291 passed
~~~

测试既使用可控替身模拟“第一批溢出、第二批成功”，证明 scheduler 只走一步；也在当前 CUDA GPU 上真实执行一次 FP16 autocast + GradScaler 更新。

## 27. 下一微步

下一步验证续训的数值时间轴：

1. 构造一条不中断的 scheduler/scaler 运行轨迹；
2. 在中间保存 checkpoint；
3. 重建 model、optimizer、scheduler、scaler 后加载；
4. 比较恢复瞬间的下一步 LR 与 scale；
5. 继续运行，确认 LR 序列与成功更新计数一致。

这个测试关注状态连续性，不要求 AMP 的浮点权重逐 bit 相同。

## 28. Scheduler 与 GradScaler 续训连续性

### 28.1 为什么普通 round-trip 还不够

只验证 `state_dict()` 保存后能原样加载，仍不能证明恢复后的对象会继续正确演化。例如计数虽然被加载，但 optimizer 当前 LR 没有同步，就会在下一次更新发生错位。

因此测试比较两条真实 CUDA 轨迹：

~~~text
连续路径：update 1 → 2 → 3 → 4

恢复路径：update 1 → 2 → 保存 checkpoint
                         ↓
          重建 model / optimizer / scheduler / scaler
                         ↓
                  加载 → update 3 → 4
~~~

### 28.2 Scaler 的可观察演化

测试设置：

~~~text
init_scale = 1024
growth_interval = 2
~~~

无溢出时：

~~~text
成功 update 2 后：scale = 2048
成功 update 4 后：scale = 4096
~~~

新建 scaler 原本会从 1024 开始。加载中断 checkpoint 后，它立即恢复为 2048；继续两步后到 4096，与连续路径一致。这同时证明 `scale` 和内部 growth tracker 都被恢复，而不只是表面数值巧合。

### 28.3 Scheduler 的可观察演化

恢复瞬间验证：

~~~text
completed_updates = 2
optimizer 当前 LR = update 2 将使用的 LR
~~~

继续训练后，两条路径的四个 epoch 边界 LR 序列一致，最终：

~~~text
continuous.completed_updates = 4
resumed.completed_updates = 4
~~~

这证明 checkpoint 恢复的是“下一步该从哪里继续”，不会重复或跳过一个学习率位置。

### 28.4 为什么不要求权重逐 bit 相同

本测试的目标是状态机时间轴。CUDA FP16 算子可能存在微小数值差异，因此不把逐 bit 权重一致作为 scheduler/scaler 接线的必要条件。后续可复现性实验关注指标是否在合理误差内，而不是把所有 GPU 浮点归约强制为位级相同。

### 28.5 测试结果

2026-09-25：

~~~text
真实 CUDA 连续/恢复对照：1 passed
项目全量测试：292 passed
~~~

## 29. 下一微步

下一步建立阶段 11 的 2-epoch smoke 配置，使用少量 batch 检查：

1. warmup + cosine 的真实 LR 输出；
2. gradient norm 与 clip fraction 写入 CSV；
3. CUDA AMP metadata 与 scaler state 写入产物；
4. `run_status.json` 正确标记 smoke；
5. 产物不会进入正式实验聚合。

## 30. 阶段 11 Smoke 验收

### 30.1 配置

配置文件：`configs/advanced/stage11_smoke.yaml`

核心设置：

~~~yaml
training:
  epochs: 2
  learning_rate: 0.0003
  scheduler:
    name: warmup_cosine
    warmup_epochs: 1
    start_factor: 0.1
    min_factor: 0.1
  gradient_clip_norm: 1.0
  amp: true
~~~

使用较小但结构完整的 MiniViT，只验证训练栈，不比较模型效果。

### 30.2 两批次时间轴 smoke

命令限制每轮 2 个 train/val batch，共 4 次计划更新：

~~~text
epoch 1 末：lr = 0.0003
epoch 2 末：lr = 0.00003
last scheduler completed/total = 4/4
~~~

`best.pt` 保留 epoch 1 边界，scheduler completed 为 2；`last.pt` 保留 epoch 2 边界，completed 为 4。两者都保存非空 scaler state。

### 30.3 单批次边界 smoke

每轮限制 1 个 batch，共 2 次更新，覆盖 `warmup_steps=1`：

~~~text
唯一 warmup update：直接使用 peak LR = 0.0003
最后 cosine update：使用 min LR = 0.00003
last scheduler completed/total = 2/2
~~~

这符合此前对 `W=1` 的数学定义：只有一个 warmup 位置时，它就是峰值，不额外制造起始倍率位置。

### 30.4 产物验收

两次运行均满足：

- `history.csv` 含 learning_rate、gradient_norm、gradient_clip_fraction；
- 有效 `config.yaml` 记录 `amp.enabled=true`、`dtype=float16` 和 CUDA metadata；
- `last.pt`/`best.pt` 含 scheduler 与 scaler state；
- `run_status.json` 为 completed；
- `is_smoke_test=true`；
- `aggregation_eligible=false`，不会进入正式结果聚合。

两轮 `clip_fraction` 均为 100%，说明阈值 1.0 下裁剪路径确实执行。smoke batch 极少，不能据此认定正式训练也会 100% 触发，更不能用 smoke accuracy 比较模型效果；它提示 O3 必须与 O2 独立对照并观察完整训练分布。

### 30.5 输出目录

~~~text
outputs/advanced/smoke/stage11_optimization/split_42__train_123
outputs/advanced/smoke/stage11_single_batch/split_42__train_123
~~~

## 31. 下一微步

下一步建立 O0–O3 正式消融配置：

~~~text
O0：固定 LR
O1：cosine
O2：warmup + cosine
O3：warmup + cosine + gradient clipping
~~~

AMP 不混入 O0–O3 的学习策略比较；先用统一 FP32 路径比较 validation，再用选定的 O3 配方单独做 AMP 开关效率对照。

## 32. O0–O3 正式消融矩阵

### 32.1 冻结的共同条件

四组配置共同使用：

~~~text
model       MiniViT p4 / d6 / embed 192 / heads 6 / MLP 768
data        CIFAR-10 + basic augmentation
epochs      100
optimizer   AdamW
peak LR     0.0003
weight decay 0.05
batch size  128
split seed  42
train seed  42
AMP         false
~~~

这是基础项目中验证表现最好的结构平衡点。阶段 11 研究优化策略，因此不同时修改 patch、depth、增强或随机种子。

### 32.2 配置矩阵

| 编号 | Scheduler | Warmup | Min LR | Clip | 回答的问题 |
|---|---|---:|---:|---:|---|
| O0 | none | — | 固定 3e-4 | off | 当前固定 LR 基线 |
| O1 | cosine | 0 | 3e-5 | off | 单独衰减是否改善后期收敛 |
| O2 | cosine | 5 epochs | 3e-5 | off | warmup 是否改善训练初期稳定性 |
| O3 | cosine | 5 epochs | 3e-5 | 1.0 | clipping 是否在 O2 上提供额外收益 |

对应文件：

~~~text
configs/advanced/stage11/o0_fixed_lr.yaml
configs/advanced/stage11/o1_cosine.yaml
configs/advanced/stage11/o2_warmup_cosine.yaml
configs/advanced/stage11/o3_warmup_cosine_clip.yaml
~~~

### 32.3 为什么 O1 仍使用 warmup_cosine 实现

当前 scheduler 是一个统一实现。O1 写成：

~~~yaml
scheduler:
  name: warmup_cosine
  warmup_epochs: 0
  start_factor: 1.0
  min_factor: 0.1
~~~

`warmup_epochs: 0` 会直接走纯 cosine 公式；`start_factor` 在该路径不参与计算，写为 1.0 是为了表达“不从较小倍率升温”，并满足统一 schema。

### 32.4 为什么 O0 必须重新运行

基础项目旧 baseline 可以作为历史锚点，但不能替代当前 O0。O0–O3 的正式比较必须来自：

- 同一版代码；
- 同一个 Git commit；
- 同一套 seed 拆分协议；
- 同样的 history/checkpoint metadata；
- 同样的梯度监测开销。

重新运行 O0 才能把差异合理归因于优化策略。

### 32.5 配置契约测试

`tests/test_stage11_configs.py` 会验证：

- 四组模型、数据、seed 与基础 optimizer 完全一致；
- O0→O1 只加入 cosine；
- O1→O2 只加入 warmup；
- O2→O3 只加入 clipping；
- AMP 在四组中全部关闭；
- experiment ID 与输出目录互不重复。

2026-09-25 验收：

~~~text
矩阵相关定向测试：55 passed
项目全量测试：295 passed
~~~

## 33. 正式训练前的下一步

在启动 100 epoch 训练前：

1. 写下 O0–O3 的实验前猜想，避免看到结果后倒推理由；
2. 提交当前实现与配置，使工作区 clean；
3. 确认四个正式输出目录均不存在；
4. 依次运行 O0、O1、O2、O3；
5. 只根据 validation 指标选出阶段 11 配方，不查看 test。

若在 dirty worktree 下运行，协议会将 `aggregation_eligible` 标记为 false，因此不能直接开始正式训练。

## 34. O0–O3 实验前猜想

记录日期：2026-09-26

记录边界：写入时四个正式输出目录均不存在，用户尚未运行或查看 O0–O3 的任何正式结果。因此以下内容没有根据实验结果反向修改。

本轮是单 train seed 筛选。“接近”的操作性定义为最佳验证准确率相差不超过 `0.5` 个百分点；它只用于整理本轮猜想，不代表统计显著性。

### 34.1 O0 vs O1：Cosine

用户预测：

> 最佳验证准确率为 `O1 > O0`，但差别可能不大。

原因：O1 的 cosine 在训练后期减小更新步幅，更适合做精细调整，有可能得到更好的准确率。

用户同时预测：

- O1 的最佳 epoch 晚于 O0；
- O1 的后期 validation 曲线更稳定。

推理是：O1 后期较小的学习率使训练继续缓慢改善，并可能减小模型在较优参数区域附近的来回震荡，因此可能在更晚 epoch 达到最佳，validation loss 也可能更加平稳。

### 34.2 O1 vs O2：Warmup

用户预测：

> O1 与 O2 的最佳验证准确率接近。

同时预测：

- 前 5 epoch 的训练 loss 下降速度为 `O1 > O2`；
- O1 的早期梯度或 loss 波动大于 O2。

原因：O2 能使训练初期更加平稳并降低异常更新风险，但当前模型规模与峰值学习率不算激进，因此最终结果可能不会产生明显差距。

精确化说明：在相同初始化、相同首批数据下，warmup 不会改变第一次 `backward()` 产生的梯度；两组第一次反向传播理论上相同。差异从第一次参数更新开始：O1 用更大的 LR 修改参数，随后几个 batch/epoch 的 loss 与梯度轨迹才可能比 O2 波动更大。

### 34.3 O2 vs O3：Gradient Clipping

用户预测：

> O3 的裁剪触发比例初期更高，随后下降；O2 与 O3 的最佳验证准确率接近。

用户认为 clipping 在本实验中更像偶尔触发的保险丝：主要阻止少量异常梯度尖峰，而不是长期介入每次更新，因此对最佳准确率的影响应该不大。

该猜想的关键反证信号是：若正式训练中 `gradient_clip_fraction` 长期接近 100%，则 clipping 实际上不是“偶尔保险”，而是持续改变优化轨迹。届时需要在实验后反思阈值 1.0 是否过低，以及 O3 与 O2 的性能差异是否来自过度约束。

### 34.4 总体预测

用户预测最佳验证准确率排序：

~~~text
O3（适度触发）≈ O2 ≈ O1 > O0
~~~

预测最佳方案为 O2 或适度触发的 O3。

最大不确定性是 O1、O2、O3 之间的最佳验证准确率排序：cosine 的主要收益相对明确，但 warmup 在当前保守峰值 LR 下可能收益有限；clipping 的效果则高度依赖实际触发频率。

### 34.5 结果揭晓后的复盘顺序

正式实验完成后按以下顺序复盘，不能只看最终排名：

1. 比较 O0/O1 的最佳验证准确率、最佳 epoch 与后期曲线；
2. 比较 O1/O2 前 5 epoch 的 loss、gradient norm 与最终验证表现；
3. 检查 O3 每轮 clip fraction 是否从高到低，以及是否长期高频触发；
4. 比较最终训练/验证差距，判断收益来自优化还是更强拟合；
5. 对照上述每条预测，分别标记成立、部分成立或不成立并解释原因。

下一步先提交当前代码、测试、配置和本猜想，使四组正式运行共享同一个 clean Git commit。
