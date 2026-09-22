"""Tiny-overfit diagnostics for the MiniViT training pipeline."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from src.models.minivit import MiniViT

from src.data import (
    CIFAR10_VAL_SIZE,
    build_transforms,
    split_train_val_indices,
)


TINY_OVERFIT_SIZE = 64
TINY_OVERFIT_SEED = 42
DATA_DIR = Path("data")


def build_tiny_overfit_dataset(
    data_dir: str | Path,
    seed: int,
) -> Subset:
    """Build a fixed, augmentation-free subset of training samples."""

    # Tiny overfit 必须反复看到完全相同的图片，因此使用无随机增强的变换。
    _, eval_transform = build_transforms()

    # 仍然读取 CIFAR-10 官方训练图片，而不是验证集或测试集。
    train_source = CIFAR10(
        root=data_dir,
        train=True,
        transform=eval_transform,
        download=False,
    )

    # 复用阶段 1 的固定划分，确保 tiny subset 属于训练划分。
    train_indices, _ = split_train_val_indices(
        dataset_size=len(train_source),
        val_size=CIFAR10_VAL_SIZE,
        seed=seed,
    )

    # 固定选择训练索引中的前 64 个。
    tiny_indices = train_indices[:TINY_OVERFIT_SIZE]

    return Subset(
        train_source,
        tiny_indices,
    )


def train_for_steps(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
) -> tuple[list[float], list[float]]:
    """Train a model repeatedly on one fixed batch and record its metrics."""
    losses: list[float] = []
    accuracies: list[float] = []

    model.train()

    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # 记录本次更新前的 loss 和预测准确率。
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()

        losses.append(loss.item())
        accuracies.append(accuracy.item())

    return losses, accuracies


@pytest.fixture(scope="module")
def tiny_dataset() -> Subset:
    """Build the deterministic tiny subset once for this test module."""
    return build_tiny_overfit_dataset(
        data_dir=DATA_DIR,
        seed=TINY_OVERFIT_SEED,
    )


# Tiny-overfit 任务必须固定为工作流规定的 64 张图片。
def test_tiny_dataset_has_expected_size(tiny_dataset):
    """Check that the diagnostic subset contains exactly 64 samples."""
    assert len(tiny_dataset) == TINY_OVERFIT_SIZE


# 无随机增强时，同一索引的图片 Tensor 和标签必须完全一致。
def test_tiny_dataset_repeats_identical_sample(tiny_dataset):
    """Read one index twice and require identical image and label values."""
    image_a, label_a = tiny_dataset[0]
    image_b, label_b = tiny_dataset[0]

    assert torch.equal(image_a, image_b)
    assert label_a == label_b


# 检查 64 张图片的 shape、dtype、有限性和标签范围。
def test_tiny_dataset_sample_contract(tiny_dataset):
    """Validate image tensors and class labels across the entire subset."""
    for image, label in tiny_dataset:
        assert image.shape == (3, 32, 32)
        assert image.dtype == torch.float32
        assert torch.isfinite(image).all()
        assert isinstance(label, int)
        assert 0 <= label < 10


# Tiny subset 必须来自固定训练索引，且不能与验证索引重叠。
def test_tiny_dataset_uses_only_training_indices(tiny_dataset):
    """Verify exact subset selection and separation from validation indices."""
    train_indices, val_indices = split_train_val_indices(
        dataset_size=len(tiny_dataset.dataset),
        val_size=CIFAR10_VAL_SIZE,
        seed=TINY_OVERFIT_SEED,
    )

    assert list(tiny_dataset.indices) == train_indices[:TINY_OVERFIT_SIZE]
    assert set(tiny_dataset.indices).isdisjoint(val_indices)


# 执行一次完整训练 step，确认 loss、梯度和参数更新链路全部生效。
def test_single_tiny_overfit_step_updates_parameters(tiny_dataset):
    """Run one training step and verify that model parameters change."""

    seed = TINY_OVERFIT_SEED
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 64 张图片构成唯一一个完整 batch。
    data_loader = DataLoader(
        tiny_dataset,
        batch_size=TINY_OVERFIT_SIZE,
        shuffle=False,
    )
    images, labels = next(iter(data_loader))

    images = images.to(device)
    labels = labels.to(device)

    model = MiniViT(
        image_size=32,
        in_channels=3,
        num_classes=10,
        patch_size=4,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_dim=128,
        dropout=0.0,
    ).to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    # 保存更新前的独立副本，稍后与更新后的参数比较。
    classifier_weight_before = (
        model.classifier.weight
        .detach()
        .clone()
    )

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()

    # step 前先确认分类头已经获得合法梯度。
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(
        model.classifier.weight.grad
    ).all()

    optimizer.step()

    classifier_weight_after = (
        model.classifier.weight
        .detach()
        .clone()
    )

    assert logits.shape == (TINY_OVERFIT_SIZE, 10)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert not torch.equal(
        classifier_weight_before,
        classifier_weight_after,
    )
    
    
# 重复训练固定 batch，确认模型能够近乎完全记住 64 张图片。
def test_tiny_overfit_reaches_near_perfect_accuracy(tiny_dataset):
    """Verify that MiniViT can memorize the fixed tiny dataset."""
    torch.manual_seed(TINY_OVERFIT_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TINY_OVERFIT_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data_loader = DataLoader(
        tiny_dataset,
        batch_size=TINY_OVERFIT_SIZE,
        shuffle=False,
    )
    images, labels = next(iter(data_loader))

    images = images.to(device)
    labels = labels.to(device)

    model = MiniViT(
        image_size=32,
        in_channels=3,
        num_classes=10,
        patch_size=4,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_dim=128,
        dropout=0.0,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    losses, accuracies = train_for_steps(
        model=model,
        images=images,
        labels=labels,
        criterion=criterion,
        optimizer=optimizer,
        num_steps=60,
    )

    print(f"losses: {losses}")
    print(f"accuracies: {accuracies}")

    assert len(losses) == 60
    assert len(accuracies) == 60

    # 所有记录值都必须有效，不能出现 NaN 或无穷大。
    assert torch.isfinite(torch.tensor(losses)).all()
    assert torch.isfinite(torch.tensor(accuracies)).all()

    # 准确率必须始终位于合法范围内。
    assert all(
        0.0 <= accuracy <= 1.0
        for accuracy in accuracies
    )

    # 完整训练后，固定 batch 上的 loss 应低于初始 loss。
    assert losses[-1] < losses[0]

    # 63/64 = 98.4375%，因此 0.98 表示最多只允许错一张。
    assert accuracies[-1] >= 0.98

    # 除了分类正确，还要求模型对正确类别具有较高置信度。
    assert losses[-1] < 0.1
