"""Gradient measurement helpers used by the training engine."""

from collections.abc import Iterable
import math
from numbers import Real

import torch
from torch import Tensor


def validate_gradient_clip_norm(value: object) -> float | None:
    """Normalize an optional positive finite global gradient-norm limit."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "gradient_clip_norm must be a positive finite number or null."
        )

    normalized_value = float(value)
    if not math.isfinite(normalized_value) or normalized_value <= 0.0:
        raise ValueError(
            "gradient_clip_norm must be a positive finite number or null."
        )

    return normalized_value


def compute_gradient_norm(
    parameters: Iterable[Tensor],
    *,
    error_if_nonfinite: bool = True,
) -> float:
    """Return the global L2 norm of all available parameter gradients."""

    gradient_norms: list[Tensor] = []

    for parameter in parameters:
        gradient = parameter.grad

        # frozen、未参与本次计算图的参数没有梯度，不应被当成零梯度报错。
        if gradient is None:
            continue

        # detach 表明这里只读取统计量，不让“测量梯度”本身进入计算图。
        detached_gradient = gradient.detach()

        # 稀疏梯度未存储的位置本来就是零，只需统计实际存储的 values。
        if detached_gradient.is_sparse:
            detached_gradient = detached_gradient.coalesce().values()

        # 先转为 FP32 再求范数，避免 FP16/BF16 的平方更容易溢出或下溢。
        gradient_norms.append(
            torch.linalg.vector_norm(
                detached_gradient.float(),
                ord=2,
            )
        )

    if not gradient_norms:
        return 0.0

    # 各参数梯度范数再做一次 L2：
    # sqrt(||g1||² + ||g2||² + ...) 等价于拼接全部梯度后求 L2，
    # 但不需要真的创建一个巨大的拼接张量。
    reference_device = gradient_norms[0].device
    total_norm = torch.linalg.vector_norm(
        torch.stack([
            norm.to(reference_device)
            for norm in gradient_norms
        ]),
        ord=2,
    )

    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise FloatingPointError(
            "Gradient norm is not finite; gradients contain NaN or infinity."
        )

    return float(total_norm.item())


def clip_gradient_norm(
    parameters: Iterable[Tensor],
    max_norm: float,
) -> tuple[float, bool]:
    """Clip one global L2 gradient vector and report its original norm."""

    normalized_max_norm = validate_gradient_clip_norm(max_norm)
    if normalized_max_norm is None:  # pragma: no cover - excluded by type/API
        raise ValueError("max_norm cannot be null when clipping gradients.")

    # clip_grad_norm_ 原地缩放 .grad，并返回裁剪前的全局范数。
    original_norm = torch.nn.utils.clip_grad_norm_(
        list(parameters),
        max_norm=normalized_max_norm,
        norm_type=2.0,
        error_if_nonfinite=True,
    )
    original_norm_value = float(original_norm.item())

    return (
        original_norm_value,
        original_norm_value > normalized_max_norm,
    )
