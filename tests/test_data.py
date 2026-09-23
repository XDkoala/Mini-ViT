"""Tests for the CIFAR-10 data pipeline."""

# Stage 1, Step 8

from pathlib import Path

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from torchvision import transforms

from src.data import (
    build_dataloaders,
    build_transforms,
    split_train_val_indices,
)

# __file__ 当前文件；.resolve() 转换为绝对路径；.parents[1] 向上跳两级
# 最终得到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


"""pytest 是测试框架，主要帮我们：
- 自动发现测试函数（以 test_ 开头的函数）
- 运行所有 assert
- 提供 fixture
- 提供参数化测试
- 检查预期异常
- 输出清晰的失败位置
"""

# fixture: 只构建一次 Dataloader，供所有测试使用（避免每个测试都加载一遍）
"""scope 决定 fixture 的生命周期：
    function    每个测试函数创建一次，默认值
    class       每个测试类创建一次
    module	    每个测试文件创建一次
    session	    整次 pytest 运行创建一次
"""
@pytest.fixture(scope="module")
def dataloaders():
    """Build the DataLoaders once for all tests in this module."""
    return build_dataloaders(
        data_dir=DATA_DIR,
        batch_size=128,
        num_workers=0,
        seed=42,
    )


# 测试1：划分是否正确且可复现
def test_split_is_reproducible_and_non_overlapping():
    # 用相同 seed 调用两次，目的是确认相同输入能产生相同划分
    train_a, val_a = split_train_val_indices(
        dataset_size=50_000,
        val_size=5_000,
        seed=42,
    )
    train_b, val_b = split_train_val_indices(
        dataset_size=50_000,
        val_size=5_000,
        seed=42,
    )

    # assert：断言后面的条件必须为真，否则测试失败。
    assert train_a == train_b
    assert val_a == val_b

    assert len(train_a) == 45_000
    assert len(val_a) == 5_000

    # 使用 set 是为了消除重复元素，这里是为了检测是否有重复索引
    assert len(set(train_a)) == 45_000
    assert len(set(val_a)) == 5_000
    # isdisjoint() 检查两组数据是否完全没有共同元素。
    assert set(train_a).isdisjoint(val_a)
    # | 是集合并集运算符
    assert set(train_a) | set(val_a) == set(range(50_000))


# 测试2：split_train_val_indices() 函数是否会拒绝不合理的参数
@pytest.mark.parametrize(
    # 让同一个测试函数运行四次
    ("dataset_size", "val_size"),   #参数名
    [   # 参数值
        (0, 0),             # 第一次：数据集本身为空
        (50_000, 0),        # 第二次：验证集大小为 0
        (50_000, 50_000),   # 第三次：验证集占据整个数据集
        (50_000, 60_000),   # 第四次：验证集比数据集还大
    ],
)
def test_split_rejects_invalid_sizes(dataset_size, val_size):
    # 这里表达的是：运行缩进范围内的代码时，必须出现 ValueError。
    with pytest.raises(ValueError):
        split_train_val_indices(
            dataset_size=dataset_size,
            val_size=val_size,
            seed=42,
        )


# 测试3：检测 DataLoader 的结构
# 这里的 dataloaders 是 pytest 自动传入的 fixture 结果。
def test_dataloader_sizes_and_sampling(dataloaders):
    train_loader, val_loader, test_loader, class_names = dataloaders

    # 检查 Dataset 样本数
    assert len(train_loader.dataset) == 45_000
    assert len(val_loader.dataset) == 5_000
    assert len(test_loader.dataset) == 10_000

    # 检查 DataLoader batch 数
    assert len(train_loader) == 352
    assert len(val_loader) == 40
    assert len(test_loader) == 79

    # isinstance(对象, 类型) 用于判断对象是否属于某个类型
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)

    # 检查类别表
    assert class_names == CIFAR10_CLASSES


# 测试4：取出一个 batch 进行 images 和 labels 的各项检测
def test_training_batch_shape_dtype_and_values(dataloaders):
    train_loader, _, _, _ = dataloaders
    # iter() 获取迭代器，next() 获取指向的元素，这里获取的是第一个 batch
    images, labels = next(iter(train_loader))

    assert images.shape == (128, 3, 32, 32)
    assert labels.shape == (128,)   # (128,) 末尾的逗号表示单元素元组

    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    # torch.isfinite(images) 会逐元素判断：普通有限数字 → True
    # .all() 要求所有元素都为 True
    assert torch.isfinite(images).all()
    
    assert 0 <= labels.min().item()
    assert labels.max().item() <= 9

# 测试5：验证变换必须稳定
def test_validation_transform_is_deterministic(dataloaders):
    _, val_loader, _, _ = dataloaders

    # 读取两次验证集中的同一个样本。没有随机操作，所以两次结果必须完全相同。
    image_a, label_a = val_loader.dataset[0]
    image_b, label_b = val_loader.dataset[0]

    assert torch.equal(image_a, image_b)
    assert label_a == label_b


# 测试6：检测 DataLoader 是否拒绝错误参数
@pytest.mark.parametrize(
    ("batch_size", "num_workers"),
    [
        (0, 0),
        (-1, 0),
        (1, -1),
    ],
)
def test_dataloader_rejects_invalid_arguments(batch_size, num_workers):
    with pytest.raises(ValueError):
        build_dataloaders(
            data_dir=DATA_DIR,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=42,
        )


# basic 与 none 必须只改变训练增强；验证变换始终保持确定性。
def test_build_transforms_supports_basic_and_none_augmentation():
    """Switch training augmentation without changing evaluation preprocessing."""

    basic_train, basic_eval = build_transforms("basic")
    plain_train, plain_eval = build_transforms("none")

    assert [type(operation) for operation in basic_train.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert [type(operation) for operation in plain_train.transforms] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert [type(operation) for operation in basic_eval.transforms] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert [type(operation) for operation in plain_eval.transforms] == [
        transforms.ToTensor,
        transforms.Normalize,
    ]


# 拼写错误不能静默退回默认增强，否则会污染消融实验结论。
def test_build_transforms_rejects_unknown_augmentation():
    """Reject unknown augmentation names instead of silently falling back."""

    with pytest.raises(
        ValueError,
        match="basic, none",
    ):
        build_transforms("strong")
