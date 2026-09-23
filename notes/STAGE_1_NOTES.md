# 阶段 1：数据最小闭环

本笔记记录 `Mini ViT` 项目阶段 1 的操作、原理、测试与验收结果。

## 进度

- [x] 明确数据模块的输入、输出与职责边界
- [x] 下载 CIFAR-10 并读取单个样本
- [x] 定义训练集与验证/测试集变换
- [x] 使用固定随机种子划分训练集和验证集
- [x] 组装训练、验证与测试 Dataset
- [x] 创建训练、验证与测试 DataLoader
- [x] 检查数据数量、shape、dtype、标签范围与有限性
- [x] 验证训练/验证索引无重叠且划分可复现
- [x] 可视化反归一化样本并检查颜色
- [x] 编写并通过 `tests/test_data.py`
- [x] 完成阶段 1 总验收

## 1. 数据模块接口与职责

统一入口：

```python
def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    ...
```

输入分别控制数据目录、批量大小、读取进程数和可复现划分的随机种子；输出依次为训练、验证、测试 DataLoader 和 CIFAR-10 类别名称。

`src/data.py` 负责下载和读取数据、定义变换、固定划分、创建 Dataset/DataLoader；不负责模型、loss、优化器、GPU 搬运、训练循环或绘图。Dataset 按索引提供单个样本，DataLoader 负责打乱、并行读取和组成 batch。

接口最初用 `NotImplementedError` 标记尚未实现，避免像 `pass` 一样静默返回 `None`；后续步骤逐层完成其内部实现。

验收命令：

```powershell
python -m py_compile src\data.py
python -c "from src.data import build_dataloaders; print(build_dataloaders.__name__)"
```

2026-09-08 验收通过：语法正确，函数签名、类型标注和 docstring 完整，模块可导入，并能按预期抛出 `NotImplementedError`。

### 关键概念

- 训练集用于计算梯度和更新参数；验证集不更新参数，用于选择配置、epoch 和最佳 checkpoint；测试集只在方案确定后做最终评估，不能继续据其调参。
- Dataset 提供单个样本，DataLoader 将样本组成 mini-batch。一次迭代处理一个 batch；一个 epoch 是完整遍历一次训练集，通常包含许多迭代。
- mini-batch 在单样本的高噪声、低并行效率与全量数据的高内存开销之间取得平衡。

## 2. 下载并检查原始 CIFAR-10

CIFAR-10 包含 60,000 张 `32×32` RGB 图片，官方划分为 50,000 张训练图片和 10,000 张测试图片，共 10 类。本项目稍后再从官方训练集中固定划出 5,000 张作为验证集。

```python
from torchvision.datasets import CIFAR10

train_dataset = CIFAR10(root="data", train=True, download=True)
test_dataset = CIFAR10(root="data", train=False, download=True)
image, label = train_dataset[0]
```

未设置 transform 时，单张图片是 PIL Image，尺寸顺序为 `(width, height)`；转成 NumPy 后采用 HWC 排列。后续 `ToTensor()` 会转换为 PyTorch 常用的 CHW 排列。

2026-09-08 验收结果：

| 检查项 | 结果 |
|---|---|
| 压缩包 MD5 | `c58f30108f718f92721af3b95e74349a` |
| 官方训练集 | `50,000` |
| 官方测试集 | `10,000` |
| 类别数 | `10` |
| 原始图片 | PIL Image，`32×32`，RGB |
| NumPy 表示 | shape `(32, 32, 3)`，`uint8`，像素范围 0–255 |
| 第一个样本 | 标签 `6`，类别 `frog` |

下载过程中遇到 HTTPS 中断，最终通过可续传下载和官方 MD5 校验完成。临时分片与日志已清理，仅保留正式压缩包和解压目录；`data/` 仍被 Git 忽略。

## 3. 训练与评估变换

```text
训练：RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → Normalize
评估：ToTensor → Normalize
```

训练增强制造位置和方向变化；验证/测试不使用随机增强，以保证指标稳定可比。`ToTensor` 将 HWC、`uint8`、0–255 转为 CHW、`float32`、0–1；`Normalize` 再按通道执行 `(x - mean) / std`。

归一化统计量只由官方训练集的全部像素计算，并统一用于训练、验证和测试：

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
```

2026-09-09 验收通过：变换顺序正确；输出 shape 为 `(3, 32, 32)`、dtype 为 `torch.float32` 且数值有限；评估变换结果一致，不同随机状态下的训练变换结果不同。发现 4 处无功能影响的行尾空格，留待代码整理时清除。

## 4. 固定训练/验证索引

`split_train_val_indices(dataset_size, val_size, seed)` 使用局部 `torch.Generator` 和 `torch.randperm` 生成无重复的完整索引排列：前 5,000 个作为验证集，其余 45,000 个作为训练集。局部生成器只控制本次划分，不污染模型初始化、增强等全局随机状态。

```text
官方训练集 50,000
├── train_indices 45,000
└── val_indices    5,000
```

划分必须跨实验固定，以保证模型和消融比较公平；训练 shuffle 和随机增强则应在一个 run 的不同 epoch 中继续变化。

2026-09-09 验收通过：两组索引内部唯一、交集为 0、并集完整覆盖 0–49,999；相同 seed 完全一致，不同 seed 产生不同划分；全局 PyTorch RNG 不受影响，非法大小均抛出 `ValueError`。

## 5. 组装三个 Dataset

从官方训练部分建立两个数据源：训练数据源使用随机增强，验证数据源使用确定性评估变换；再用固定索引分别包装为 `Subset`。官方测试部分使用 `train=False` 和评估变换。

```text
官方训练部分 50,000 ── train transform ── Subset 45,000（训练）
                    └─ eval transform  ── Subset  5,000（验证）
官方测试部分 10,000 ── eval transform  ───────── 10,000（测试）
```

两份训练数据源读取相同的本地数据，但允许各自持有不同的 transform；`Subset` 保存数据源和索引关系，不复制磁盘图片。`download=True` 会校验并复用完整的本地文件。

2026-09-09 验收通过：三个 Dataset 数量为 45,000 / 5,000 / 10,000；类别数为 10；样本为 `(3, 32, 32)`、`float32`、数值有限且标签在 0–9；训练/验证索引交集为 0，验证样本重复读取结果一致。

## 6. 创建 DataLoader

训练 DataLoader 使用 `shuffle=True` 和带 seed 的局部生成器；验证、测试使用固定顺序。三者均保留不足一个完整 batch 的末批次，并仅在 CUDA 可用时启用 `pin_memory`。

以 `batch_size=128` 为例：

| 数据 | 样本数 | batch 数 | sampler |
|---|---:|---:|---|
| 训练 | 45,000 | 352 | `RandomSampler` |
| 验证 | 5,000 | 40 | `SequentialSampler` |
| 测试 | 10,000 | 79 | `SequentialSampler` |

2026-09-09 验收通过：三种 loader 均输出 `[128, 3, 32, 32]` 的 `float32` 图片和 `[128]` 的 `int64` 标签；数值有限、标签位于 0–9；相同 seed 的初始训练顺序一致；非法 `batch_size` 和 `num_workers` 均抛出 `ValueError`。

## 7. 反归一化与样本可视化

显示前按 `image * std + mean` 撤销归一化，限制到 `[0, 1]`，再将图片从 CHW 转为 Matplotlib 使用的 HWC。网格标题由数值标签映射到类别名称。

2026-09-10 验收通过：随机图片经历归一化和逆变换后的最大误差约为 `5.96e-08`；非法单图 shape 会抛出 `ValueError`；成功生成 4×4 验证样本图。肉眼检查颜色自然、方向和通道顺序正确，标签与内容整体匹配。

验收产物：`outputs/stage1_val_samples.png`（运行输出，不纳入 Git）。

## 8. 自动化测试与阶段总验收

`tests/test_data.py` 使用 module 级 fixture 复用 DataLoader，并检查固定划分、非法参数、数据规模、采样方式、类别顺序、batch 格式、数值有限性和验证变换确定性。参数化用例会展开为多项独立测试。

2026-09-10 执行 `python -m pytest tests/test_data.py -v`：`11 passed`。pytest 仅提示缓存目录写入警告，不影响测试结果；`.pytest_cache/` 已被 Git 忽略。

阶段 1 总验收通过：三个 loader 均可独立输出 `[128, 3, 32, 32]` 图片和 `[128]` 标签；数据数量、固定划分、索引隔离、变换职责、反归一化显示及自动化测试全部符合工作流要求。`data/`、`outputs/`、`.venv/` 和测试缓存均未纳入 Git。

代码仍有 7 处无功能影响的行尾空格，可在后续整理时统一清除。
