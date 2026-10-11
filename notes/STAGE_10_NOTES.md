# 阶段 10：进阶实验协议与基础设施

本阶段在加入 AMP、scheduler 或高级增强之前，先升级实验身份、随机种子、日志、checkpoint 和聚合协议。目标是让后续每个结论都公平、可恢复、可追溯。

开始日期：2026-09-25

当前状态：已完成

当前位置：阶段 10 总验收通过；后续阶段 11–18 也已完成。本笔记保留当时逐微步的学习顺序，当前全项目进度见 [STAGE_18_NOTES.md](STAGE_18_NOTES.md)。

## 进度

- [x] 审计当前 seed、数据划分、DataLoader 与全局随机数的数据流
- [x] 定义 split_seed 与 train_seed 的职责
- [x] 确定旧配置的向后兼容原则
- [x] 实现 v1/v2 seed 解析契约
- [x] 解耦数据划分与训练 DataLoader generator
- [x] 更新训练、评估和可视化入口
- [x] 增加实验 format_version 与运行元数据
- [x] 扩展 checkpoint，为 scheduler/scaler 预留兼容字段
- [x] 扩展 history 可选字段
- [x] 实现正式实验状态生命周期
- [x] 实现正式实验聚合契约与脚本
- [x] 完成定向测试
- [x] 完成 2 epoch 隔离 smoke
- [x] 完成全量回归测试
- [x] 阶段 10 总验收

## 1. 不可变基础锚点

阶段 0–9 的正式结果不覆盖、不重训、不改名：

~~~text
MiniViT baseline
  best val accuracy = 81.84%
  test accuracy = 80.65%
  output = outputs/minivit_baseline/

SimpleCNN baseline
  best val accuracy = 89.58%
  test accuracy = 88.29%
  output = outputs/cnn_baseline/
~~~

旧 YAML、旧 best.pt/last.pt 和旧评估入口必须继续可读。进阶实验只写入 configs/advanced、outputs/advanced 和 assets/advanced。

## 2. 当前 seed 数据流

当前 YAML 只有一个顶层字段：

~~~yaml
seed: 42
~~~

train.py 读取它后同时做三件事：

~~~text
seed
├── set_seed(seed)
│   ├── Python random
│   ├── NumPy
│   ├── torch CPU
│   └── torch CUDA
├── split_train_val_indices(..., seed)
│   └── 决定 train/validation 成员
└── DataLoader(generator=manual_seed(seed))
    └── 决定训练集初始 shuffle 顺序
~~~

此外，RandomCrop、RandomHorizontalFlip、dropout 和参数初始化也受全局随机状态影响。

这意味着将 seed 从 42 改成 123 时，不仅模型初始化和训练顺序改变，train/validation 的成员也改变。因此现有单一 seed 不适合直接做“只测训练随机性”的多 seed 实验。

## 3. v2 随机性配置

进阶配置采用独立顶层 section：

~~~yaml
randomness:
  split_seed: 42
  train_seed: 42
~~~

不使用 runtime.train_seed。原因是当前 train.py 会写入派生运行状态：

~~~python
config["runtime"] = {
    "is_smoke_test": ...,
    "max_train_batches": ...,
    "max_val_batches": ...,
}
~~~

若把用户配置写进 runtime，现有代码会整体覆盖它。runtime 应继续表示本次执行产生的状态，而不是用户声明的实验随机性。

### 3.1 split_seed

只负责：

- 50,000 张官方训练图片如何拆成 45,000 train 与 5,000 validation；
- 保证所有模型和训练配方看到相同的数据成员。

正式进阶实验固定：

~~~text
split_seed = 42
~~~

### 3.2 train_seed

负责：

- 模型参数初始化；
- DataLoader shuffle；
- RandomCrop / RandomHorizontalFlip；
- dropout；
- Mixup/CutMix 等未来训练随机事件；
- 其他 Python、NumPy、PyTorch 随机操作。

正式统计计划：

~~~text
train_seed = 42, 123, 2026
~~~

## 4. 向后兼容原则

### 4.1 v1 旧配置

~~~yaml
seed: 42
~~~

解析为：

~~~text
split_seed = 42
train_seed = 42
~~~

因此旧实验行为不变。

### 4.2 v2 进阶配置

~~~yaml
randomness:
  split_seed: 42
  train_seed: 123
~~~

v2 配置必须显式提供两个 seed，避免缺一个字段时做出隐蔽假设。

第一版实现不修改已有 baseline/ablation YAML，只让加载层同时理解 v1 和 v2。

### 4.3 冲突处理

若配置同时包含 seed 和 randomness，应主动报错，而不是猜测哪个优先。一个配置只能使用一种 schema。

## 5. 计划中的接口变化

### 5.1 数据层

当前：

~~~python
build_datasets(..., seed)
build_dataloaders(..., seed)
~~~

计划：

~~~python
build_datasets(..., split_seed)

build_dataloaders(
    ...,
    split_seed,
    train_seed,
)
~~~

- Dataset 划分只接收 split_seed；
- 训练 DataLoader generator 只接收 train_seed；
- validation/test 仍然 shuffle=False；
- 函数命名直接表达随机数用途。

### 5.2 训练入口

计划数据流：

~~~text
resolve_random_seeds(config)
  → split_seed, train_seed
  → set_seed(train_seed)
  → build_dataloaders(split_seed, train_seed)
~~~

有效配置快照必须写入两个解析后的 seed，确保运行事实清楚。

### 5.3 评估与可视化

评估重建 validation/test 数据时只需要 split_seed。为了保证所有潜在随机操作一致，仍可调用 set_seed(train_seed)，但 test loader 不 shuffle、eval transform 不随机，因此预测顺序不依赖 train_seed。

可视化按 checkpoint 内嵌配置解析种子，不依赖后来修改的仓库 YAML。

## 6. config / checkpoint / history 草案

### 6.1 配置版本

进阶有效配置计划增加：

~~~yaml
format_version: 2
~~~

旧配置没有该字段时按 v1 解释。

### 6.2 checkpoint

v1 当前字段：

~~~text
epoch
model_state
optimizer_state
best_val_accuracy
history
config
~~~

v2 最终计划：

~~~text
format_version
epoch
model_state
optimizer_state
scheduler_state
scaler_state
best_val_accuracy
history
config
run_metadata
~~~

阶段 10 先建立兼容契约。scheduler_state 和 scaler_state 在相应功能尚未启用时允许为 null。

### 6.3 history

当前固定字段：

~~~text
epoch
train_loss
train_accuracy
val_loss
val_accuracy
epoch_time_seconds
~~~

计划支持：

~~~text
learning_rate
gradient_norm
gradient_clip_fraction
~~~

新增字段应由阶段 11 真正产生；阶段 10 只设计向后兼容方式，不能给旧历史伪造数据。

### 6.4 run metadata

至少计划记录：

~~~text
experiment_id
status
git_commit
created_at
python_version
torch_version
cuda_version
device_name
split_seed
train_seed
is_smoke_test
~~~

## 7. 计划测试

### 7.1 配置解析

- v1 seed 解析成相同 split/train seed；
- v2 randomness 分别解析；
- 同时出现两套 schema 时拒绝；
- seed 越界时拒绝；
- v2 缺字段时拒绝；
- 保存后的有效配置可 round-trip。

### 7.2 数据层

- 相同 split_seed、不同 train_seed 的 train/val 索引完全相同；
- 相同 train_seed 的首轮 shuffle 一致；
- 不同 train_seed 的训练顺序不同；
- validation/test 顺序不受 train_seed 影响。

### 7.3 兼容性

- 现有 configs/minivit.yaml 与 configs/cnn.yaml 仍可加载；
- v1 checkpoint 仍可恢复；
- v2 可选状态为 null 时可恢复；
- smoke 不被正式聚合。

## 8. 验收标准

- [x] 数据划分与训练随机性已真正解耦。
- [x] 旧配置和旧 checkpoint 行为不变。
- [x] v2 配置没有模糊默认值。
- [x] 有效配置保存解析后的两个 seed。
- [x] smoke 与正式实验物理隔离。
- [x] run 能追溯到配置、commit、环境和 seed。
- [x] 定向测试与全量回归通过。
- [x] 2 epoch smoke 能训练、保存和续训。

以上完成状态依据本笔记末尾阶段总验收更新；中途 smoke 的 dirty 状态及排除原因仍按原记录保留。

## 9. 第一微步之后做什么

下一微步只实现“随机性配置解析契约”及其单元测试，不立刻修改 DataLoader，也不加入 scheduler、AMP 或新增强。

这样先得到一个明确、可测试的输入接口，再让数据层和训练入口逐步迁移。

## 10. v1/v2 配置解析实现

`src/config.py` 新增 `resolve_random_seeds(config)`，返回：

~~~python
(split_seed, train_seed)
~~~

解析规则：

~~~text
只有 seed
  → v1
  → (seed, seed)

只有 randomness
  → v2
  → (randomness.split_seed, randomness.train_seed)

两者同时存在
  → 配置含义冲突，拒绝

两者都不存在
  → 缺少随机性配置，拒绝
~~~

`load_config()` 会调用解析器完成早期校验，但不会把 v1 配置偷偷改写成 v2。原因是：

- 读取函数应验证用户输入，而不是静默改变其 schema；
- 旧配置保存后再读取应保持内容一致；
- 有效配置若要标准化，应由明确的运行配置组装步骤完成；
- 调用者可以看出自己使用的是 legacy 还是 advanced schema。

### 10.1 seed 数值校验

两个 schema 共用 `_validate_seed()`：

- 必须是 Python `int`；
- 显式拒绝 `bool`，因为 `bool` 虽是 `int` 子类，但通常代表 YAML 笔误；
- 合法范围为 `[0, 2**32)`，兼容当前 NumPy 等随机数入口；
- 错误信息包含完整字段名，例如 `randomness.train_seed`。

### 10.2 为什么 v2 不提供隐式默认值

如果 v2 只写 `train_seed`，程序不能默认为 `split_seed = train_seed`，否则会重新制造本阶段要消除的耦合。反过来也一样。因此 v2 必须显式提供两个字段。

### 10.3 测试结果

2026-09-25：

~~~text
tests/test_config.py：22 passed
项目全量测试：193 passed
~~~

新增测试覆盖：

- 真实旧 MiniViT 配置解析为 `(42, 42)`；
- v2 `(42, 123)` 保持原结构且正确解析；
- 双 schema与无 schema 被拒绝；
- v2 缺任一字段被拒绝；
- bool、float、负数和 `2**32` 被拒绝；
- 非 mapping 的 randomness 被拒绝。

pytest 仍有一个 Windows `.pytest_cache` 已存在警告，不影响 193 项测试结果。

## 11. 下一微步

下一步迁移 `src/data.py`：

1. `build_datasets()` 只接收 `split_seed`；
2. `build_dataloaders()` 分别接收 `split_seed` 与 `train_seed`；
3. 证明不同 train_seed 不改变 train/validation 成员；
4. 证明不同 train_seed 会改变训练 DataLoader 的样本顺序；
5. 暂不修改训练入口，避免一次跨越两个职责层。

## 12. 数据层 seed 解耦实现

`src/data.py` 现在支持两套互斥调用方式。

### 12.1 旧接口

~~~python
build_dataloaders(
    ...,
    seed=42,
)
~~~

兼容解释为：

~~~text
split_seed = 42
train_seed = 42
~~~

因此现有 `train.py`、`evaluate.py`、`visualize.py` 和旧测试不需要在同一微步全部修改。

### 12.2 新接口

~~~python
build_dataloaders(
    ...,
    split_seed=42,
    train_seed=123,
)
~~~

内部职责：

~~~text
split_seed
  → split_train_val_indices()
  → 决定 Subset.indices

train_seed
  → torch.Generator().manual_seed()
  → 传给训练 RandomSampler
  → 决定训练样本访问顺序
~~~

`build_datasets()` 只需要 `split_seed`，因为 Dataset 本身不负责训练批次的 shuffle。

### 12.3 为什么保留兼容层

如果立刻删除 `seed` 参数，所有旧脚本会同时损坏，难以判断错误来自数据层还是调用入口。当前做法把迁移分为两步：

1. 数据层先支持新语义，同时维持旧行为；
2. 下一微步再让各入口显式使用 v2 解析结果。

兼容层不是让新代码继续偷懒。以下调用会被拒绝：

- 同时传 `seed` 和显式 seeds；
- 只传 `split_seed`；
- 只传 `train_seed`；
- 完全不传 seed。

### 12.4 “成员相同”与“顺序不同”

固定 `split_seed=42` 后，两个训练集的 `Subset.indices` 必须逐项相同。这意味着两次训练使用的是同一批 45,000 张图片。

改变 `train_seed` 后，RandomSampler 生成的索引排列不同。这意味着模型仍看相同样本，但第一轮看到它们的先后顺序不同。

这才是多训练 seed 实验要研究的随机性之一：

~~~text
相同数据问题 + 不同训练随机轨迹
~~~

### 12.5 当前边界

RandomCrop、RandomHorizontalFlip、dropout 和参数初始化仍由全局 `set_seed(train_seed)` 控制。数据层只负责划分与 DataLoader generator；全局随机状态将在入口迁移时接上 `train_seed`。

因此现在只能说“数据层具备真正解耦的接口”，还不能说正式训练入口已经启用 v2。

### 12.6 测试结果

2026-09-25：

~~~text
tests/test_data.py：19 passed
项目全量测试：199 passed
~~~

新增测试证明：

- 同一 split_seed、不同 train_seed 的 train/validation 成员完全相同；
- 不同 train_seed 的 RandomSampler 顺序不同；
- 相同 train_seed 的首轮采样顺序完全一致；
- 两种 schema 混用或显式字段不完整时会报错；
- 所有旧入口仍通过全量回归。

## 13. 下一微步

下一步迁移三个程序入口：

1. `train.py` 用 `resolve_random_seeds()`；
2. `set_seed(train_seed)` 控制初始化和训练随机事件；
3. DataLoader 显式接收 split_seed/train_seed；
4. `evaluate.py` 和 `visualize.py` 从 checkpoint 配置解析同一 schema；
5. 用脚本测试证明 v1 调用参数不变、v2 参数正确传递。

## 14. 程序入口迁移

### 14.1 训练入口

`train.py` 现在先统一解析配置：

~~~python
split_seed, train_seed = resolve_random_seeds(config)
set_seed(train_seed)
~~~

随后把两个值显式交给数据层：

~~~python
build_dataloaders(
    ...,
    split_seed=split_seed,
    train_seed=train_seed,
)
~~~

这使两种职责在入口处就清晰分开：

- `split_seed` 固定训练集与验证集分别有哪些样本；
- `train_seed` 控制参数初始化、随机增强、dropout 和训练样本顺序等随机轨迹。

### 14.2 评估与可视化入口

`evaluate.py` 和 `visualize.py` 不读取当前仓库中的 YAML，而是解析 checkpoint 内嵌的配置。因此：

- 历史 checkpoint 中的 `seed: 42` 仍可正常使用；
- 新 checkpoint 中的 `randomness` 也能正确解析；
- 即使日后修改仓库配置，旧实验仍按训练当时的随机性契约复现。

可视化只重建测试 Dataset，因此只把 `split_seed` 传给 `build_datasets()`；它仍用 `train_seed` 设置全局随机状态，使模型重建过程遵守原实验配置。

### 14.3 为什么评估也传 train_seed

当前 `build_dataloaders()` 会一次性构建 train、validation 和 test 三个 loader。评估最终只使用 test loader，但完整构造过程仍要求明确的 `train_seed`。这不会改变测试集内容或测试顺序，只是保持统一、无歧义的接口。

后续若单独实现 `build_test_dataloader()`，评估入口便可只构建测试 loader；这不是当前微步的必要改动。

### 14.4 兼容性测试

2026-09-25：

~~~text
入口脚本定向测试：24 passed
项目全量测试：201 passed
~~~

测试分别用旧式和新式配置运行被隔离的 `train.main()` 与 `evaluate.main()`，证明：

- v1 的 `seed: 42` 会把 `42` 同时传给两个职责；
- v2 的 `split_seed: 17`、`train_seed: 29` 不会被混淆；
- `set_seed()` 接收的是 `train_seed`；
- DataLoader 同时收到明确的 `split_seed` 与 `train_seed`；
- 阶段 0–9 的其余功能没有发生测试回归。

pytest 仍报告一个 Windows `.pytest_cache` 已存在警告；它不代表测试失败，也不影响上述 201 项结果。

## 15. 下一微步

下一步设计并实现实验格式版本与运行元数据。先回答两个问题：

1. 一个输出目录中的文件属于哪一版实验协议；
2. 除 YAML 超参数外，还要保存哪些运行环境信息，才能日后解释和复现实验。

这一微步暂不修改 checkpoint 的训练状态字段；checkpoint 兼容层将在元数据契约确定后单独实现。

## 16. 实验格式版本与运行元数据

### 16.1 `format_version` 标记什么

`format_version` 标记的是本次生成的实验产物遵循哪一版读写协议，而不是模型版本，也不是 Python 包版本。

当前契约：

~~~text
旧 YAML 没有 format_version
  → 读取时解释为 v1
  → 不修改源文件

新训练生成有效配置
  → 显式写入 format_version: 2
~~~

这一区分很重要：兼容旧格式不等于继续生成旧格式。读取器可以理解历史产物，写入器则应生成当前格式。

显式版本必须是受支持的整数。`true`、字符串 `"2"`、0 和未知的未来版本都会被拒绝，程序不会猜测其含义。

### 16.2 元数据保存在哪里

训练入口把元数据放入有效配置的：

~~~yaml
format_version: 2
runtime:
  is_smoke_test: false
  max_train_batches: null
  max_val_batches: null
  metadata:
    ...
~~~

该有效配置会在 `fit()` 前保存为输出目录中的 `config.yaml`，也会随当前 checkpoint 的内嵌 `config` 一起保存。因此即使原始 YAML 后来被修改，实验产物仍保留运行当时的事实。

### 16.3 当前记录的事实

~~~text
created_at_utc   创建时间，带 UTC 时区
python_version   Python 版本
torch_version    PyTorch 版本
cuda_version     PyTorch 对应的 CUDA 版本；CPU 构建可为 null
platform         操作系统与平台描述
device_type      cpu 或 cuda
device_name      实际设备名称
git_commit       当前 Git commit；不可取得时为 null
split_seed       数据划分种子
train_seed       训练随机种子
is_smoke_test    是否为受限诊断运行
~~~

所有值都使用字符串、整数、布尔值或 `null`，不直接保存 `torch.device`、`Path` 等运行时对象，因此能够安全写入 YAML/JSON。

### 16.4 为什么时间统一使用 UTC

本地时间若不带时区，无法判断来自上海、伦敦还是云服务器。UTC 时间带明确的 `+00:00` 偏移，跨机器排序与比较不会产生歧义；显示报告时再转换成本地时区即可。

### 16.5 Git commit 能保证什么

commit 哈希能指出运行时检出的代码版本，但它不能证明工作区没有未提交修改。因此它是实验身份的重要组成部分，却不是完整证明。

当前微步在 Git 不可用或目录不是仓库时记录 `null`，而不是中止训练。后续建立正式实验状态和聚合协议时，还应记录工作区是否 dirty，并决定 dirty run 能否进入正式汇总。

### 16.6 为什么暂不写 `status`

简单地在训练前写 `status: running` 不足以构成可靠状态系统：正常完成、Python 异常、断电和用户强制终止需要不同处理。如果没有原子写入和异常收尾规则，状态字段反而可能误导聚合脚本。

因此本微步只记录事实型元数据；`running / completed / failed` 将与输出目录和聚合规则一起实现。

### 16.7 测试结果

2026-09-25：

~~~text
配置、工具与训练入口定向测试：54 passed
项目全量测试：208 passed
~~~

测试证明：

- 缺失版本号的历史配置按 v1 读取；
- 当前 v2 可显式读取，非法和未知版本会被拒绝；
- 元数据正确记录环境、设备、seed 和 smoke 标记；
- Git commit 不可取得时训练仍可继续；
- 新训练的有效配置被提升到 v2；
- 旧训练、评估与 checkpoint 测试没有回归。

Windows `.pytest_cache` 的已存在警告仍不影响测试结果。

## 17. 下一微步

下一步升级 checkpoint 兼容契约：

1. 新 checkpoint 显式携带 `format_version` 与 `run_metadata`；
2. 为阶段 11 的 scheduler 和 AMP scaler 预留可为空的状态字段；
3. 加载旧 v1 checkpoint 时继续恢复模型与 optimizer；
4. 加载新 v2 checkpoint 时验证版本和新增字段；
5. 用 round-trip 与旧格式测试证明续训基础状态没有损坏。

## 18. v2 checkpoint 与向后兼容

### 18.1 新 checkpoint 的结构

新写入的 checkpoint 一律采用 v2：

~~~text
format_version
epoch
model_state
optimizer_state
scheduler_state
scaler_state
best_val_accuracy
history
config
run_metadata
~~~

其中：

- `model_state` 保存模型参数和 buffer；
- `optimizer_state` 保存 AdamW 动量、步数和参数组；
- `scheduler_state` 保存学习率调度进度；
- `scaler_state` 保存 AMP 动态缩放状态；
- `run_metadata` 从有效配置的 `runtime.metadata` 提取，方便不展开整份配置也能定位运行环境。

### 18.2 为什么未启用的功能也保存 `null`

阶段 10 尚未真正启用 scheduler 和 AMP，因此当前值是：

~~~python
"scheduler_state": None,
"scaler_state": None,
~~~

字段存在但值为 `null` 表示“这是 v2 checkpoint，且本次运行没有该组件”。字段完全缺失则表示文件不完整或不符合它声明的 v2 契约。

这两种情况不能混为一谈：

~~~text
字段存在，值为 null  → 明确没有状态
v2 中字段缺失         → 产物结构损坏或写入不完整
~~~

### 18.3 旧 checkpoint 如何兼容

历史 v1 checkpoint 没有 `format_version`，也没有三个扩展字段。加载器将它解释为 v1，并在返回给训练入口时归一化：

~~~python
{
    "format_version": 1,
    ...,
    "scheduler_state": None,
    "scaler_state": None,
    "run_metadata": None,
}
~~~

因此上层续训代码面对新旧文件时能使用相同字段，不必到处判断 key 是否存在；模型、optimizer、epoch、history 和最佳验证准确率仍按旧方式恢复。

### 18.4 为什么不能静默忽略扩展状态

假设 checkpoint 已保存训练到第 50 轮的 scheduler 状态，但恢复时没有创建 scheduler。如果加载器只恢复模型和 optimizer，学习率计划会从错误位置开始，表面上“成功续训”，实际上已经不是同一次实验轨迹。

因此当前规则是：

- checkpoint 状态为 `null`：无需提供对应对象；
- checkpoint 状态非空：必须提供 scheduler/scaler 对象；
- 缺少对象时明确报错。

而且该检查发生在模型和 optimizer 被修改之前，防止加载失败后留下“只恢复了一半”的内存状态。

### 18.5 评估为什么也检查 checkpoint 版本

评估不需要恢复 optimizer、scheduler 或 scaler，但仍需要验证文件结构。否则一个自称 v2、实际缺字段的 checkpoint 可能被错误当作完整正式产物。

评估入口因此会：

1. 将无版本号文件解释为 v1；
2. 对 v2 检查扩展字段是否完整；
3. 拒绝未知未来版本，而不是猜测读取。

### 18.6 测试结果

2026-09-25：

~~~text
checkpoint、engine、评估与训练入口定向测试：63 passed
项目全量测试：211 passed
~~~

新增测试证明：

- 新 checkpoint 写入完整 v2 字段；
- 尚未启用的 scheduler/scaler 状态为 `null`；
- 手工构造的历史 v1 checkpoint 仍能恢复；
- 非空 scheduler/scaler 状态能完整 round-trip；
- 缺少对应恢复对象时会明确失败；
- v2 缺字段以及未知版本会被评估入口拒绝；
- 既有训练、续训、评估和可视化功能没有回归。

Windows `.pytest_cache` 的已存在警告不影响测试结果。

## 19. 下一微步

下一步扩展 history 的可选字段契约，为阶段 11 预留：

~~~text
learning_rate
gradient_norm
gradient_clip_fraction
~~~

重点不是现在伪造这些指标，而是让 CSV 写入器同时兼容：

- 阶段 0–10 的基础 history；
- 阶段 11 真正产生高级指标后的扩展 history；
- 续训时旧记录与新记录可能拥有不同字段的过渡情况。

## 20. history 的基础字段与可选字段

### 20.1 两类字段

每个 epoch 都必须存在的基础字段：

~~~text
epoch
train_loss
train_accuracy
val_loss
val_accuracy
epoch_time_seconds
~~~

阶段 11 才可能产生的可选字段：

~~~text
learning_rate
gradient_norm
gradient_clip_fraction
~~~

代码中分别使用 `BASE_HISTORY_FIELDS` 和 `OPTIONAL_HISTORY_FIELDS` 声明，最终 CSV 列顺序由二者拼接得到。这样“哪些字段必需、哪些字段允许缺失”不再只是口头约定。

### 20.2 旧记录和新记录如何共存

假设一个历史实验前两轮没有记录梯度范数，升级后第三轮开始记录：

~~~text
epoch, ..., learning_rate, gradient_norm, gradient_clip_fraction
1,     ...,              ,              ,
2,     ...,              ,              ,
3,     ..., 0.0003       , 1.25         , 0.10
~~~

CSV 的空单元格表示“这一轮没有记录该指标”。它不等于：

- 梯度范数为 0；
- 学习率为 0；
- 指标计算结果为 NaN。

如果伪造为 0，后续绘图和统计会把“缺失信息”错误解释成真实观测值。

### 20.3 为什么 CSV 现在总有扩展列

新写入器使用统一 v2 表头，即使当前基础训练尚未产生高级指标，也会把可选列写成空白。这样后续工具不必根据每个文件动态猜测列结构，续训时也无需中途改变表头。

历史磁盘上的旧 CSV 不会被批量改写；它们仍可由按字段名读取的现有绘图代码使用。

### 20.4 写入前完整验证

写入器会先检查全部记录，然后才创建文件：

- 缺少任何基础字段时拒绝写入；
- 出现未声明字段时拒绝写入；
- 缺少可选字段时允许写入空白。

“先验证、后写盘”可以避免程序写到一半才报错，留下看似存在、实际不完整的 `history.csv`。

拒绝未知字段不是限制实验扩展，而是要求新增指标先进入正式 schema。否则拼写错误，例如 `gradient_nrom`，可能被当成一个新指标长期保存。

### 20.5 当前尚未产生高级指标

阶段 10 只建立存储契约，没有伪造 `learning_rate`、`gradient_norm` 或 `gradient_clip_fraction`。这些值将在阶段 11 真正接入 scheduler、梯度统计与裁剪后，由训练循环计算并写入。

### 20.6 测试结果

2026-09-25：

~~~text
history、engine 与可视化定向测试：52 passed
项目全量测试：214 passed
~~~

新增测试证明：

- 纯基础 history 会生成空白的可选列；
- 旧式基础行和扩展行可以出现在同一 CSV；
- 扩展值按固定列正确保存；
- 缺少基础字段时在写盘前失败；
- 未声明字段不会被静默接受；
- 既有训练循环、续训和绘图不受影响。

Windows `.pytest_cache` 的已存在警告不影响测试结果。

## 21. 下一微步

下一步设计正式实验状态与聚合契约，重点区分：

~~~text
running    正在运行或尚未正常收尾
completed  已完成全部计划训练并通过产物检查
failed     已捕获异常并留下失败信息
~~~

同时规定 smoke、失败运行和 dirty Git 工作区产生的结果是否允许进入正式聚合。先把状态文件及其原子写入规则实现清楚，再编写 `aggregate.py`。

## 22. 实验状态生命周期

### 22.1 为什么单独使用 `run_status.json`

`config.yaml` 是实验设置与运行环境的快照，原则上写入后不应不断变化；状态则会经历：

~~~text
running → completed
running → failed
~~~

因此每个新运行目录使用独立的 `run_status.json`。这个文件体积小，适合频繁、原子地替换，也让聚合器不必读取整份配置来判断运行是否完成。

### 22.2 状态文件结构

~~~text
format_version
experiment_id
status
updated_at_utc
target_epochs
completed_epochs
is_smoke_test
git_commit
git_is_dirty
aggregation_eligible
error
~~~

其中 `error` 在非失败状态为 `null`；失败时记录异常类型和消息。它用于快速诊断，不替代完整 traceback。

### 22.3 三种状态的含义

`running`：入口已经建立运行身份，但尚未正常完成。程序正在运行、机器断电或进程被强制终止，都可能让文件停留在此状态。

`completed`：`fit()` 正常返回，且最后 epoch 等于计划的 `target_epochs`。仅仅存在 `last.pt` 不能代替这个判断。

`failed`：程序捕获到异常或中断，在重新抛出原异常前记录错误信息。因此调用者仍能看到原始报错，不会被状态系统吞掉。

### 22.4 状态转换覆盖的范围

`running` 在构建 DataLoader、模型和 optimizer 之前写入。之后的数据构建、模型构建、checkpoint 恢复、配置保存和训练异常都会进入 `failed` 分支。

异常分支捕获后会重新抛出原异常。如果连失败状态本身也无法写入，附加说明会加入原异常，但原始训练错误仍保持为主错误。

### 22.5 为什么要原子替换

直接覆盖 JSON 时，如果恰好断电，文件可能只写了一半。当前写法是：

~~~text
完整写入 run_status.json.tmp
            ↓
写入成功后 replace
            ↓
run_status.json 一次性切换到新状态
~~~

读取者看到的应当是旧的完整状态或新的完整状态，而不是半段 JSON。

### 22.6 Git dirty 的含义

`git_commit` 只能确定最近一次提交。若工作区还有未提交修改，真实运行代码并不完全等于该 commit。

因此元数据新增：

~~~text
git_is_dirty = false  工作区与 commit 一致
git_is_dirty = true   存在未提交或未跟踪改动
git_is_dirty = null   无法取得 Git 状态
~~~

dirty run 仍可用于调试和探索，但不能自动进入要求精确追溯的正式聚合。

### 22.7 当前正式聚合资格

只有同时满足以下条件，状态文件中的 `aggregation_eligible` 才为 `true`：

1. `status == completed`；
2. 不是 smoke test；
3. 能取得 `git_commit`；
4. `git_is_dirty is false`。

因此下列运行都不会混入正式结果：

- 尚未完成或异常失败；
- 只跑少量 batch 的 smoke；
- 无法追溯代码提交；
- 使用未提交代码产生的 dirty run。

`aggregation_eligible` 是写入时的快速判断。下一微步的聚合器仍会重新核验原始字段，不能只盲信这个布尔值。

### 22.8 测试结果

2026-09-25：

~~~text
状态工具与训练生命周期定向测试：38 passed
项目全量测试：220 passed
~~~

测试覆盖：

- running 状态被 completed 原子替换；
- completed 的 epoch 数必须等于计划轮数；
- failed 必须包含异常类型；
- smoke、dirty 和 failed 均不具备聚合资格；
- 训练成功产生 running → completed；
- 训练异常产生 running → failed，并继续抛出原异常；
- Git 不可用时使用 `null` 而不是阻止诊断运行；
- 全部旧功能保持通过。

Windows `.pytest_cache` 的已存在警告不影响测试结果。

## 23. 下一微步

下一步新增 `scripts/aggregate.py`，但先只完成最小、严格的聚合行为：

1. 扫描指定输出根目录下的 `run_status.json`；
2. 重新验证格式版本、状态、smoke、Git 和 epoch 完整性；
3. 只读取合格运行的配置与 history；
4. 输出结构化汇总 CSV；
5. 对跳过的运行给出明确原因，而不是静默消失。

## 24. 严格实验聚合器

### 24.1 两份输出而不是一份

`scripts/aggregate.py` 会生成：

~~~text
summary.csv          通过全部检查的正式运行
summary_skipped.csv  被跳过的运行及具体原因
~~~

只输出成功表会让“不见了的实验”难以解释。单独保存跳过表，可以区分实验尚未完成、属于 smoke、Git dirty、缺文件或 history 损坏。

### 24.2 发现哪些运行目录

聚合器递归扫描输入根目录。只要目录中存在以下任一顶层实验产物，就把它视为候选运行：

~~~text
run_status.json
config.yaml
history.csv
best.pt
last.pt
~~~

因此缺少 `run_status.json` 的旧目录也不会被静默忽略，而会进入跳过表并标明原因。

### 24.3 为什么不能只信 `aggregation_eligible`

状态文件可能被手动编辑、来自旧代码或写入不完整。聚合器会从原始字段重新计算资格：

1. 格式版本受支持；
2. 状态确实是 `completed`；
3. `is_smoke_test is false`；
4. Git commit 非空；
5. `git_is_dirty is false`；
6. 完成轮数等于合法的目标轮数；
7. `aggregation_eligible` 与这些事实一致。

布尔标记是缓存的结论，不是证据本身。

### 24.4 产物完整性检查

通过状态检查后，还必须存在：

~~~text
config.yaml
history.csv
best.pt
last.pt
~~~

配置中的 `experiment.name` 必须与状态文件的 `experiment_id` 相同，seed 必须能通过统一解析器读取。

history 还需满足：

- 包含全部基础字段；
- epoch 精确构成 `1..target_epochs`，无缺失、重复或跳号；
- 必需数值可以解析；
- 至少包含一轮记录。

当前只检查 checkpoint 文件存在，不在聚合时加载大体积权重。更深的 checkpoint 哈希与内容验证可在最终归档阶段执行。

### 24.5 汇总字段

每个合格运行输出：

~~~text
experiment_id
run_dir
format_version
git_commit
split_seed
train_seed
target_epochs
best_epoch
best_val_accuracy
final_train_accuracy
final_val_accuracy
total_epoch_time_seconds
~~~

最佳 epoch 只根据 validation accuracy 选择，不读取 test 指标，继续遵守测试集隔离原则。

### 24.6 原子写入和空表

两份汇总 CSV 都先写临时文件，再原子替换目标文件。即使没有合格运行，`summary.csv` 仍会保留表头；这明确表示“扫描成功但结果为空”，不同于脚本根本没有运行。

### 24.7 命令行

默认扫描进阶实验目录：

~~~powershell
python -m scripts.aggregate
~~~

等价于扫描：

~~~text
outputs/advanced
~~~

也可显式指定隔离目录与输出文件：

~~~powershell
python -m scripts.aggregate `
  --input-root outputs/advanced `
  --output outputs/advanced/summary.csv `
  --skipped-output outputs/advanced/summary_skipped.csv
~~~

### 24.8 测试结果

2026-09-25：

~~~text
聚合器定向测试：7 passed
项目全量测试：227 passed
~~~

测试覆盖：

- 合格运行的 seed、最佳验证结果、最终结果和耗时汇总；
- running、smoke、dirty 和无 commit 运行的明确排除；
- 缺失 checkpoint 的完整性拒绝；
- 合格表和跳过表同时写出；
- 临时 CSV 不会在成功写入后残留；
- 既有训练、评估、可视化和续训测试全部通过。

Windows `.pytest_cache` 的已存在警告不影响测试结果。

## 25. 下一微步

下一步先固定 `outputs/advanced` 的目录命名约定，再创建一个 v2 smoke 配置并运行 2 epoch 隔离测试。该 smoke 需要实际生成：

~~~text
config.yaml
history.csv
best.pt
last.pt
run_status.json
~~~

随后运行聚合器，验证 smoke 会出现在 `summary_skipped.csv`，且明确以 smoke 原因被排除。

## 26. 输出命名与 2 epoch 集成 smoke

### 26.1 advanced 输出命名

进阶实验采用：

~~~text
outputs/advanced/<阶段或研究>/<实验名>/split_<split_seed>__train_<train_seed>/
~~~

例如本次 smoke：

~~~text
outputs/advanced/
└── smoke/
    └── stage10_protocol/
        └── split_42__train_123/
~~~

路径让人能够快速浏览研究主题和 seed；结构化配置与 `run_status.json` 才是程序判断实验身份和资格的依据，聚合器不会只解析目录名。

### 26.2 专用配置

创建：

~~~text
configs/advanced/stage10_smoke.yaml
~~~

它使用完整但缩小的 MiniViT：

~~~text
format_version = 2
split_seed = 42
train_seed = 123
embed_dim = 48
depth = 2
num_heads = 4
epochs = 2
~~~

缩小模型只为更快验证基础设施，不能用此次准确率评价正式 MiniViT。

### 26.3 正确的启动方式

首次尝试直接执行：

~~~powershell
python scripts/train.py ...
~~~

当前项目布局下，Python 会把 `scripts/` 作为脚本导入起点，导致同级 `src/` 无法导入。正确方式是从项目根目录按模块启动：

~~~powershell
python -m scripts.train `
  --config configs/advanced/stage10_smoke.yaml `
  --epochs 2 `
  --max-batches 2 `
  --output-dir outputs/advanced/smoke/stage10_protocol/split_42__train_123
~~~

同理，聚合器使用：

~~~powershell
python -m scripts.aggregate
~~~

### 26.4 训练结果

2026-09-25，CUDA：

~~~text
Epoch 1：train_acc=7.81%，val_acc=14.06%，time=0.4s
Epoch 2：train_acc=12.50%，val_acc=15.62%，time<0.1s
~~~

每轮只使用 2 个训练 batch 和 2 个验证 batch。这些准确率不具备实验结论价值，只证明前向传播、反向传播、验证与保存链路能运行。

### 26.5 产物验收

输出目录包含：

~~~text
config.yaml
history.csv
best.pt
last.pt
run_status.json
~~~

checkpoint 验收：

~~~text
format_version = 2
scheduler_state = null
scaler_state = null
run_metadata.split_seed = 42
run_metadata.train_seed = 123
~~~

history 共 2 行，epoch 连续；高级指标列存在但为空，符合阶段 10 契约。

状态验收：

~~~text
status = completed
target_epochs = 2
completed_epochs = 2
is_smoke_test = true
git_is_dirty = true
aggregation_eligible = false
~~~

即使工作区是 clean，这次运行也会因为 smoke 标记而被排除。当前 dirty 状态进一步说明代码尚未提交，因此也不能当作正式可追溯结果。

### 26.6 聚合器集成验收

运行：

~~~powershell
python -m scripts.aggregate
~~~

结果：

~~~text
eligible = 0
skipped = 1
~~~

`summary.csv` 只有表头；`summary_skipped.csv` 包含本次运行，原因是：

~~~text
run is a smoke test or has an invalid smoke flag
~~~

这证明聚合器没有因为目录中存在完整 checkpoint 就误收诊断结果。

## 27. 下一微步

下一步进行阶段 10 总验收：

1. 创建 `notes/ADVANCED_EXPERIMENT_RESULTS.md`，作为阶段 11 以后正式实验的统一结果总表；
2. 核对阶段 10 所有验收项和实际文件；
3. 再运行最终全量测试；
4. 将阶段状态改为已完成，然后进入阶段 11。

## 28. 阶段 10 总验收

验收日期：2026-09-25

### 28.1 协议与兼容性

- [x] `split_seed` 与 `train_seed` 职责分离；
- [x] 同一 split seed 保持相同数据成员；
- [x] 不同 train seed 改变训练采样随机轨迹；
- [x] 旧 `seed` YAML 继续按原行为解释；
- [x] 旧 v1 checkpoint 继续恢复模型、optimizer 和历史；
- [x] 新配置与 checkpoint 显式使用 format v2；
- [x] scheduler/scaler 状态槽可为空且支持非空 round-trip；
- [x] 实际启用后的 scheduler/scaler 数值连续性明确留到阶段 11 验证。

### 28.2 可追溯性与数据契约

- [x] 有效配置记录 Python、PyTorch、CUDA、设备、Git 和 seed；
- [x] 记录 Git dirty 状态；
- [x] history 区分基础字段与可选高级字段；
- [x] 缺失高级指标保持空白，不伪造数值；
- [x] checkpoint 内嵌有效配置和运行元数据；
- [x] advanced 输出目录命名约定已经确定。

### 28.3 状态与聚合

- [x] `run_status.json` 支持 running、completed、failed；
- [x] 状态文件采用临时文件原子替换；
- [x] smoke、failed、dirty 和不可追溯运行不会进入正式聚合；
- [x] 聚合器重新验证原始字段和必要产物；
- [x] 合格结果与跳过原因分别保存；
- [x] `ADVANCED_EXPERIMENT_RESULTS.md` 已建立。

### 28.4 集成运行

完成两次互不覆盖的真实 CUDA smoke：

~~~text
单 batch smoke
  epochs = 1
  max_train_batches = 1
  max_val_batches = 1
  status = completed

2 epoch smoke
  epochs = 2
  max_train_batches = 2
  max_val_batches = 2
  status = completed
~~~

最终聚合结果：

~~~text
eligible = 0
skipped = 2
~~~

两次均因 `is_smoke_test=true` 被正确排除。

### 28.5 最终测试

~~~text
pytest：227 passed
警告：1 个 Windows .pytest_cache 已存在警告
~~~

该警告不影响任何测试结论。

### 28.6 阶段结论

阶段 10 已把“可以训练的项目”升级为“能够识别、追溯、恢复、筛选和汇总实验的项目”。后续阶段可以在不混淆数据划分、训练随机性、诊断运行和正式结果的前提下，引入训练技巧。

下一阶段：阶段 11——优化训练栈，包括 warmup、cosine decay、梯度范数、梯度裁剪与 AMP。
