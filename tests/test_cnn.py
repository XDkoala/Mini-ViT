"""Tests for the convolutional CIFAR-10 baseline."""

import pytest
import torch
from torch import nn

from src.models.cnn import ConvBlock, SimpleCNN


# 单个卷积块应增加通道数，并通过 2x2 最大池化把空间尺寸减半。
def test_conv_block_changes_channels_and_halves_spatial_size():
    """Expand feature channels while halving height and width."""

    block = ConvBlock(in_channels=3, out_channels=16)
    inputs = torch.randn(2, 3, 32, 32)

    outputs = block(inputs)

    assert outputs.shape == (2, 16, 16, 16)
    assert torch.isfinite(outputs).all()


# 默认 CNN 必须遵守分类模型的公共契约：[B, 3, 32, 32] -> [B, 10]。
def test_simple_cnn_returns_one_logit_vector_per_image():
    """Return finite ten-class logits for every image in the batch."""

    model = SimpleCNN()
    images = torch.randn(2, 3, 32, 32)

    logits = model(images)

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


# AdaptiveAvgPool 使分类头不依赖固定空间尺寸，较大图片也能共用同一模型。
def test_simple_cnn_accepts_larger_spatial_resolution():
    """Use adaptive pooling to classify a larger valid image size."""

    model = SimpleCNN(channels=(8, 16))

    logits = model(torch.randn(3, 3, 40, 48))

    assert logits.shape == (3, 10)


# ModuleList 中的全部卷积块与最终分类头都必须位于反向传播路径上。
def test_simple_cnn_parameters_receive_finite_gradients():
    """Backpropagate through every convolution block and the classifier."""

    model = SimpleCNN(channels=(8, 16), dropout=0.0)
    images = torch.randn(4, 3, 16, 16)
    labels = torch.tensor([0, 1, 2, 3])

    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters
    assert all(parameter.grad is not None for parameter in trainable_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in trainable_parameters
    )


# 默认宽度经过选择后应与 MiniViT 同属约 2–3M 参数量级，但不宣称完全相等。
def test_default_simple_cnn_has_expected_parameter_scale():
    """Keep the baseline in the intended two-to-three-million parameter range."""

    parameter_count = sum(
        parameter.numel()
        for parameter in SimpleCNN().parameters()
    )

    assert 2_000_000 <= parameter_count <= 3_000_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0},
        {"num_classes": 0},
        {"channels": ()},
        {"channels": (16, 0)},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_simple_cnn_rejects_invalid_configuration(kwargs):
    """Reject invalid channel, class, and dropout configuration values."""

    with pytest.raises(ValueError):
        SimpleCNN(**kwargs)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((3, 32, 32), "expected input shape"),
        ((2, 1, 32, 32), "expected 3 input channels"),
        ((2, 3, 8, 32), "at least 16"),
    ],
)
def test_simple_cnn_rejects_invalid_image_shape(shape, message):
    """Reject malformed or spatially undersized image tensors."""

    model = SimpleCNN()

    with pytest.raises(ValueError, match=message):
        model(torch.randn(*shape))
