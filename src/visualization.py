"""Visualization helpers for images and model results."""

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib.figure import Figure
from torch import Tensor

from .data import CIFAR10_MEAN, CIFAR10_STD

# Stage 1, Step 7.1
# 反归一化函数：将归一化结果重新还原
def denormalize_image(image: Tensor) -> Tensor:
    """Reverse CIFAR-10 normalization for one CHW image."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, height, width].")

    # Tensor.new_tensor(List) 创建新的内容为 List 的同性质张量
    # Tensor.view() 改变张量形状，这样 PyTorch 可以通过广播机制分别处理三个通道
    mean = image.new_tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = image.new_tensor(CIFAR10_STD).view(3, 1, 1)

    # original = normalized * std + mean
    # .clamp(0.0, 1.0) 把极小的浮点误差限制回合法图片范围
    return (image * std + mean).clamp(0.0, 1.0)

# Stage 1, Step 7.2
# 图片网格函数
def plot_image_grid(
    images: Tensor,
    labels: Tensor,
    class_names: Sequence[str],
    max_images: int = 16,
) -> Figure:
    """Plot a grid of normalized images with their class names."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [batch, 3, height, width].")

    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("labels must have shape [batch].")

    if max_images <= 0:
        raise ValueError("max_images must be greater than 0.")

    image_count = min(max_images, len(images))
    columns = min(4, image_count)
    rows = math.ceil(image_count / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns, 2.5 * rows),
        squeeze=False,
    )

    for index, axis in enumerate(axes.flat):
        axis.axis("off")

        if index >= image_count:
            continue

        image = denormalize_image(images[index])
        # detach() 切断梯度记录；cpu()：Matplotlib 不能直接读取 CUDA Tensor；permute(1, 2, 0)：从 CHW 变成 HWC。
        image = image.detach().cpu().permute(1, 2, 0).numpy()

        label = int(labels[index].item())

        if not 0 <= label < len(class_names):
            raise ValueError(f"label {label} is outside class_names.")

        axis.imshow(image)
        axis.set_title(class_names[label])

    figure.tight_layout()
    return figure


# Stage 8
def plot_training_history(
    history: Sequence[Mapping[str, int | float]],
    best_epoch: int,
    title: str,
) -> Figure:
    """Plot train/validation loss and accuracy for one experiment."""
    if not history:
        raise ValueError("history must not be empty.")

    required_fields = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    }
    if any(not required_fields.issubset(row) for row in history):
        raise ValueError("history rows are missing required fields.")

    epochs = [int(row["epoch"]) for row in history]
    if best_epoch not in epochs:
        raise ValueError("best_epoch must exist in history.")

    figure, (loss_axis, accuracy_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
    )

    # Loss 和 accuracy 分开画，避免二者数值范围不同而互相压缩。
    loss_axis.plot(
        epochs,
        [float(row["train_loss"]) for row in history],
        label="Train",
    )
    loss_axis.plot(
        epochs,
        [float(row["val_loss"]) for row in history],
        label="Validation",
    )
    loss_axis.axvline(
        best_epoch,
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        label=f"Best epoch ({best_epoch})",
    )
    loss_axis.set(
        title="Cross-Entropy Loss",
        xlabel="Epoch",
        ylabel="Loss",
    )
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()

    accuracy_axis.plot(
        epochs,
        [100.0 * float(row["train_accuracy"]) for row in history],
        label="Train",
    )
    accuracy_axis.plot(
        epochs,
        [100.0 * float(row["val_accuracy"]) for row in history],
        label="Validation",
    )
    accuracy_axis.axvline(
        best_epoch,
        color="tab:red",
        linestyle="--",
        alpha=0.8,
        label=f"Best epoch ({best_epoch})",
    )
    accuracy_axis.set(
        title="Top-1 Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy (%)",
    )
    accuracy_axis.grid(alpha=0.25)
    accuracy_axis.legend()

    figure.suptitle(title)
    figure.tight_layout()
    return figure


def build_confusion_matrix(
    records: Sequence[Mapping[str, object]],
    num_classes: int,
) -> np.ndarray:
    """Count predictions using rows=true labels and columns=predictions."""
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")

    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for record in records:
        true_label = int(record["true_label"])
        predicted_label = int(record["predicted_label"])

        if not 0 <= true_label < num_classes:
            raise ValueError(f"true label {true_label} is outside the matrix.")
        if not 0 <= predicted_label < num_classes:
            raise ValueError(
                f"predicted label {predicted_label} is outside the matrix."
            )

        matrix[true_label, predicted_label] += 1

    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    title: str,
) -> Figure:
    """Plot a row-normalized confusion matrix annotated with percentages."""
    expected_shape = (len(class_names), len(class_names))
    if matrix.shape != expected_shape:
        raise ValueError(f"matrix must have shape {expected_shape}.")

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, label="Fraction of true class")

    axis.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axis.tick_params(axis="x", rotation=45)

    # 每格同时写百分比和样本数：颜色便于快速比较，数字便于精确引用。
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = normalized[row, column]
            text_color = "white" if value >= 0.5 else "black"
            axis.text(
                column,
                row,
                f"{100.0 * value:.1f}%\n({matrix[row, column]})",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    figure.tight_layout()
    return figure


def select_error_cases(
    records: Sequence[Mapping[str, object]],
    max_cases: int,
) -> list[Mapping[str, object]]:
    """Select confident errors while first preserving confusion-pair variety."""
    if max_cases <= 0:
        raise ValueError("max_cases must be greater than zero.")

    errors = [record for record in records if not bool(record["correct"])]
    errors.sort(key=lambda row: float(row["confidence"]), reverse=True)

    selected: list[Mapping[str, object]] = []
    selected_ids: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()

    # 第一轮优先覆盖不同的“真实类 → 预测类”错误类型。
    for record in errors:
        pair = (
            int(record["true_label"]),
            int(record["predicted_label"]),
        )
        if pair in seen_pairs:
            continue

        selected.append(record)
        selected_ids.add(int(record["sample_index"]))
        seen_pairs.add(pair)
        if len(selected) == max_cases:
            return selected

    # 若错误类型不足，再按置信度补满，但不重复样本。
    for record in errors:
        sample_index = int(record["sample_index"])
        if sample_index in selected_ids:
            continue

        selected.append(record)
        if len(selected) == max_cases:
            break

    return selected


def plot_error_cases(
    images: Tensor,
    records: Sequence[Mapping[str, object]],
    title: str,
) -> Figure:
    """Plot normalized CIFAR-10 images with truth, prediction, confidence."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [batch, 3, height, width].")
    if len(images) != len(records) or not records:
        raise ValueError("images and non-empty records must have equal length.")

    columns = min(4, len(records))
    rows = math.ceil(len(records) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.1 * columns, 3.0 * rows),
        squeeze=False,
    )

    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= len(records):
            continue

        image = denormalize_image(images[index])
        axis.imshow(image.detach().cpu().permute(1, 2, 0).numpy())
        record = records[index]
        axis.set_title(
            f"True: {record['true_class']}\n"
            f"Pred: {record['predicted_class']} "
            f"({float(record['confidence']):.1%})",
            color="firebrick",
            fontsize=9,
        )

    figure.suptitle(title)
    figure.tight_layout()
    return figure


def attention_to_heatmaps(
    attention: Tensor,
    image_size: int,
) -> Tensor:
    """Convert final-layer CLS-to-patch attention into image-sized maps."""
    if attention.ndim != 4:
        raise ValueError("attention must have shape [batch, heads, tokens, tokens].")
    if attention.shape[-1] != attention.shape[-2]:
        raise ValueError("attention token dimensions must be square.")
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero.")

    patch_count = attention.shape[-1] - 1
    grid_size = math.isqrt(patch_count)
    if grid_size * grid_size != patch_count:
        raise ValueError("patch token count must form a square image grid.")

    # token 0 是 CLS；列 1: 是所有图像 patch。先对多个 head 求平均，
    # 得到每张图一个 patch 网格，再放大到原图分辨率。
    cls_to_patch = attention[:, :, 0, 1:].mean(dim=1)
    patch_maps = cls_to_patch.reshape(-1, 1, grid_size, grid_size)
    heatmaps = functional.interpolate(
        patch_maps,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    # 每张图独立缩放到 [0, 1]，使叠加色彩表达相对关注强弱。
    flat_maps = heatmaps.flatten(start_dim=1)
    minimum = flat_maps.min(dim=1).values[:, None, None]
    maximum = flat_maps.max(dim=1).values[:, None, None]
    return (heatmaps - minimum) / (maximum - minimum).clamp_min(1e-12)


def plot_attention_overlays(
    images: Tensor,
    heatmaps: Tensor,
    records: Sequence[Mapping[str, object]],
    title: str,
    alpha: float = 0.55,
) -> Figure:
    """Show each original image beside its final-layer attention overlay."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [batch, 3, height, width].")
    if heatmaps.ndim != 3 or heatmaps.shape[0] != images.shape[0]:
        raise ValueError("heatmaps must have shape [batch, height, width].")
    if len(records) != len(images) or not records:
        raise ValueError("images and non-empty records must have equal length.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    figure, axes = plt.subplots(
        len(records),
        2,
        figsize=(6.5, 2.8 * len(records)),
        squeeze=False,
    )

    for index, record in enumerate(records):
        image = denormalize_image(images[index])
        image_array = image.detach().cpu().permute(1, 2, 0).numpy()
        heatmap = heatmaps[index].detach().cpu().numpy()
        status = "correct" if bool(record["correct"]) else "wrong"
        description = (
            f"True: {record['true_class']} | "
            f"Pred: {record['predicted_class']} "
            f"({float(record['confidence']):.1%}, {status})"
        )

        axes[index, 0].imshow(image_array)
        axes[index, 0].set_title(description, fontsize=9)
        axes[index, 0].axis("off")

        axes[index, 1].imshow(image_array)
        axes[index, 1].imshow(
            heatmap,
            cmap="jet",
            alpha=alpha,
            vmin=0.0,
            vmax=1.0,
        )
        axes[index, 1].set_title("Final-block CLS attention", fontsize=9)
        axes[index, 1].axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    return figure
