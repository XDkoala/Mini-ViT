"""Visualization helpers for images and model results."""

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
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