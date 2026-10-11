"""CIFAR-10 dataset and DataLoader construction."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Stage 1, Step 3
# 通道均值与通道标准差（官方统计）
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def _validate_data_seed(
    value: object,
    field_name: str,
) -> int:
    """Validate a seed passed directly to the data-layer API."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")

    if not 0 <= value < 2**32:
        raise ValueError(
            f"{field_name} must be in the range [0, 2**32)."
        )

    return value


def _resolve_dataset_split_seed(
    legacy_seed: int | None,
    split_seed: int | None,
) -> int:
    """Resolve a legacy dataset seed or one explicit split seed."""

    if legacy_seed is not None and split_seed is not None:
        raise ValueError(
            "Use either legacy seed or split_seed, not both."
        )

    selected_seed = (
        split_seed
        if split_seed is not None
        else legacy_seed
    )
    if selected_seed is None:
        raise ValueError("split_seed is required.")

    return _validate_data_seed(selected_seed, "split_seed")


def _resolve_dataloader_seeds(
    legacy_seed: int | None,
    split_seed: int | None,
    train_seed: int | None,
) -> tuple[int, int]:
    """Resolve legacy coupled seeds or explicit split/training seeds."""

    has_legacy_seed = legacy_seed is not None
    has_explicit_seed = (
        split_seed is not None
        or train_seed is not None
    )

    if has_legacy_seed and has_explicit_seed:
        raise ValueError(
            "Use either legacy seed or explicit "
            "split_seed/train_seed, not both."
        )

    if has_legacy_seed:
        resolved_seed = _validate_data_seed(
            legacy_seed,
            "seed",
        )
        return resolved_seed, resolved_seed

    if split_seed is None or train_seed is None:
        raise ValueError(
            "split_seed and train_seed are both required."
        )

    return (
        _validate_data_seed(split_seed, "split_seed"),
        _validate_data_seed(train_seed, "train_seed"),
    )

# 定义数据集的图像变换
def build_transforms(
    augmentation: str = "basic",
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build configured training transforms and deterministic evaluation transforms."""

    augmentation_name = augmentation.strip().lower()

    if augmentation_name == "basic":
        # 基础增强只作用于训练集，为同一图片制造合理的位置与朝向变化。
        train_operations = [
            # 随机裁剪：图片四周先补 4 个像素，再随机裁剪回 32×32
            # 相当于让物体发生轻微位置移动，降低模型对固定位置的依赖
            transforms.RandomCrop(32, padding=4),

            # 随机水平翻转：默认概率 50%
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation_name == "none":
        # 无增强实验不执行随机操作，但仍保留必要的张量转换和归一化。
        train_operations = []
    else:
        raise ValueError(
            "augmentation must be one of: basic, none."
        )

    common_operations = [
            # PIL/NumPy [H, W, C] uint8 0–255
            # → Tensor [C, H, W] float32 0.0–1.0
            transforms.ToTensor(),

            # 归一化
            # 归一化值 = (原值 - 通道均值) / 通道标准差
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    train_transform = transforms.Compose(
        [*train_operations, *common_operations]
    )

    # 验证集和测试集的图像变换
    # 不使用随机增强的原因：验证准确率可能变化，导致不同 epoch 无法公平比较
    eval_transform = transforms.Compose(
        common_operations
    )

    return train_transform, eval_transform

# Stage 1, Step 4
CIFAR10_VAL_SIZE = 5_000    # 定义验证集 5,000，那么训练集 45,000

# 由于官方样本没有拆分训练集和验证集，我们这里需要主动拆分
def split_train_val_indices(
    dataset_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create reproducible, non-overlapping training and validation indices."""
    if dataset_size <= 0:
        raise ValueError("dataset_size must be greater than 0.")

    if not 0 < val_size < dataset_size:
        raise ValueError("val_size must be between 0 and dataset_size.")

    # 局部随机数生成器
    generator = torch.Generator().manual_seed(seed)
    # 生成随机排列
    shuffled_indices = torch.randperm(
        dataset_size,
        generator=generator,
    ).tolist()

    # 前 5,000 个 → val；后 45,000 个 → train
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    return train_indices, val_indices

# Stage 1, Step 5
def build_datasets(
    data_dir: str | Path,
    seed: int | None = None,
    augmentation: str = "basic",
    *,
    split_seed: int | None = None,
) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    """Build datasets using a legacy seed or one explicit split seed."""
    resolved_split_seed = _resolve_dataset_split_seed(
        legacy_seed=seed,
        split_seed=split_seed,
    )

    # 获取两套变换
    train_transform, eval_transform = build_transforms(
        augmentation=augmentation,
    )

    # 训练/验证 的 CIFAR-10 对象（前 datasets）
    # train_source 和 val_source 用同一批训练图片，但预处理不同
    train_source = CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True,
    )

    val_source = CIFAR10(
        root=data_dir,
        train=True,
        transform=eval_transform,
        download=True,
    )
    
    # 测试的 datasets
    test_dataset = CIFAR10(
        root=data_dir,
        train=False,
        transform=eval_transform,
        download=True,
    )

    # 产生固定索引
    train_indices, val_indices = split_train_val_indices(
        dataset_size=len(train_source),
        val_size=CIFAR10_VAL_SIZE,
        seed=resolved_split_seed,
    )

    # 用索引建立子集
    # Subset 不会再复制一份图片文件，只保存原 Dataset 和索引关系。
    train_dataset = Subset(train_source, train_indices)
    val_dataset = Subset(val_source, val_indices)
    
    # ['airplane', 'automobile', ... , 'truck']
    class_names = list(train_source.classes)

    return train_dataset, val_dataset, test_dataset, class_names

# Stage 1, Step 1
def build_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int | None = None,
    augmentation: str = "basic",
    *,
    split_seed: int | None = None,
    train_seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build the CIFAR-10 training, validation, and test DataLoaders.

    Args:
        data_dir: Directory used to download or read CIFAR-10.
        batch_size: Number of samples in each batch.
        num_workers: Number of worker processes used for data loading.
        seed: Legacy seed controlling both the split and training shuffle.
        augmentation: Training augmentation mode: ``basic`` or ``none``.
        split_seed: Explicit seed used only for train/validation membership.
        train_seed: Explicit seed used for the training DataLoader shuffle.

    Returns:
        A tuple containing the training DataLoader, validation DataLoader,
        test DataLoader, and CIFAR-10 class names.
    """
    # Stage 1, Step 6
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    resolved_split_seed, resolved_train_seed = (
        _resolve_dataloader_seeds(
            legacy_seed=seed,
            split_seed=split_seed,
            train_seed=train_seed,
        )
    )

    # 获取 datasets
    train_dataset, val_dataset, test_dataset, class_names = build_datasets(
        data_dir=data_dir,
        augmentation=augmentation,
        split_seed=resolved_split_seed,
    )

    # 训练顺序只由 train_seed 控制，不再影响数据集成员划分。
    train_generator = torch.Generator().manual_seed(
        resolved_train_seed
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,  # 是否使用 CUDA
        drop_last=False,        # 是否丢弃最后一组 batch
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_names
