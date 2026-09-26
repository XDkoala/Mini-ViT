"""Tests for gradient measurement helpers."""

import pytest
import torch
from torch import nn

from src.gradients import (
    clip_gradient_norm,
    compute_gradient_norm,
    validate_gradient_clip_norm,
)


# 所有参数梯度应被视为一个长向量，而不是分别取平均值。
def test_compute_gradient_norm_matches_global_l2_definition():
    """Combine separate gradient tensors using the global L2 definition."""

    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([3.0, 4.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([0.0, 12.0])

    total_norm = compute_gradient_norm([first, second])

    assert total_norm == pytest.approx(13.0)


# 未参与当前计算图或被冻结的参数可能没有 .grad，应直接忽略。
def test_compute_gradient_norm_ignores_parameters_without_gradients():
    """Ignore parameters whose gradients were not produced by backward."""

    with_gradient = nn.Parameter(torch.tensor([1.0]))
    without_gradient = nn.Parameter(torch.tensor([2.0]))
    with_gradient.grad = torch.tensor([5.0])

    assert compute_gradient_norm(
        parameter
        for parameter in [with_gradient, without_gradient]
    ) == pytest.approx(5.0)


# backward 尚未发生时没有可统计梯度，定义总范数为 0，而不是制造异常。
def test_compute_gradient_norm_returns_zero_when_no_gradients_exist():
    """Return zero for an empty gradient collection."""

    parameters = [nn.Parameter(torch.ones(2)) for _ in range(2)]

    assert compute_gradient_norm(parameters) == 0.0


# 诊断函数只能读取梯度，不能原地缩放、裁剪或清空它们。
def test_compute_gradient_norm_does_not_modify_gradients():
    """Measure gradients without changing their values or graph metadata."""

    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    original_gradient = parameter.grad.clone()

    compute_gradient_norm([parameter])

    assert torch.equal(parameter.grad, original_gradient)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_compute_gradient_norm_rejects_non_finite_gradients(invalid_value):
    """Fail immediately when NaN or infinity reaches the gradient stream."""

    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([1.0, invalid_value])

    with pytest.raises(FloatingPointError, match="not finite"):
        compute_gradient_norm([parameter])


# 稀疏梯度只存非零 values，但它们仍应参与同一个全局 L2 范数。
def test_compute_gradient_norm_supports_sparse_gradients():
    """Measure stored values of a sparse parameter gradient."""

    embedding = nn.Embedding(5, 2, sparse=True)
    embedding(torch.tensor([1, 3])).sum().backward()

    assert embedding.weight.grad.is_sparse
    assert compute_gradient_norm(
        embedding.parameters()
    ) == pytest.approx(2.0)


@pytest.mark.parametrize("value", [None, 0.5, 2])
def test_validate_gradient_clip_norm_accepts_disabled_or_positive(value):
    """Normalize valid clipping configuration without changing its meaning."""

    expected = None if value is None else float(value)

    assert validate_gradient_clip_norm(value) == expected


@pytest.mark.parametrize(
    "value",
    [True, "1.0", 0.0, -1.0, float("nan"), float("inf")],
)
def test_validate_gradient_clip_norm_rejects_invalid_values(value):
    """Reject ambiguous, non-positive, and non-finite clipping thresholds."""

    with pytest.raises(ValueError, match="positive finite"):
        validate_gradient_clip_norm(value)


# 全局 norm clipping 应统一缩放全部梯度，保持完整梯度向量方向。
def test_clip_gradient_norm_limits_global_norm_and_preserves_direction():
    """Scale one global gradient vector to max_norm without rotating it."""

    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([3.0, 4.0]))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([0.0, 12.0])
    original_gradients = torch.cat([
        first.grad.clone(),
        second.grad.clone(),
    ])

    original_norm, was_clipped = clip_gradient_norm(
        [first, second],
        max_norm=2.0,
    )
    clipped_gradients = torch.cat([first.grad, second.grad])

    assert original_norm == pytest.approx(13.0)
    assert was_clipped is True
    assert torch.linalg.vector_norm(clipped_gradients).item() == pytest.approx(
        2.0,
        abs=1e-5,
    )
    assert clipped_gradients == pytest.approx(
        original_gradients * (2.0 / 13.0),
        abs=1e-5,
    )


def test_clip_gradient_norm_leaves_below_threshold_gradients_unchanged():
    """Avoid modifying a global gradient already within the safety limit."""

    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])
    original_gradient = parameter.grad.clone()

    original_norm, was_clipped = clip_gradient_norm(
        [parameter],
        max_norm=10.0,
    )

    assert original_norm == pytest.approx(5.0)
    assert was_clipped is False
    assert torch.equal(parameter.grad, original_gradient)
