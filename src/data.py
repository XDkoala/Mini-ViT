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

# 定义数据集的图像变换
def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Build the training and evaluation transforms for CIFAR-10."""
    # 训练集的图像变换
    train_transform = transforms.Compose(
        [
            # 随机裁剪：图片四周先补 4 个像素，再随机裁剪回 32×32
            # 相当于让物体发生轻微位置移动，降低模型对固定位置的依赖
            transforms.RandomCrop(32, padding=4),
            
            # 随机水平翻转：默认概率 50%
            transforms.RandomHorizontalFlip(),
            
            # 转成 Tensor
            #    PIL/NumPy：[H, W, C]，uint8，  0–255
            # ➡ Tensor：   [C, H, W]，float32，0.0–1.0
            transforms.ToTensor(),
            
            # 归一化
            # 归一化值 = (原值 - 通道均值) / 通道标准差
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    
    # 验证集和测试集的图像变换
    # 不使用随机增强的原因：验证准确率可能变化，导致不同 epoch 无法公平比较
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
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
    seed: int,
) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    """Build CIFAR-10 training, validation, and test datasets."""
    # 获取两套变换
    train_transform, eval_transform = build_transforms()

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
        seed=seed,
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
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build the CIFAR-10 training, validation, and test DataLoaders.

    Args:
        data_dir: Directory used to download or read CIFAR-10.
        batch_size: Number of samples in each batch.
        num_workers: Number of worker processes used for data loading.
        seed: Random seed used for the train/validation split.

    Returns:
        A tuple containing the training DataLoader, validation DataLoader,
        test DataLoader, and CIFAR-10 class names.
    """
    # Stage 1, Step 6
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    # 获取 datasets
    train_dataset, val_dataset, test_dataset, class_names = build_datasets(
        data_dir=data_dir,
        seed=seed,
    )

    # 局部随机数生成器
    train_generator = torch.Generator().manual_seed(seed)

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