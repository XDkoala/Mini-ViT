"""Tests for MiniViT model components."""

import pytest
import torch

from src.models.minivit import (
    MLP,
    MiniViT,
    MultiHeadSelfAttention,
    PatchEmbedding,
    TransformerEncoderBlock,
)


# 检查默认 MiniViT 配置下的 token 数量、特征维度和数值稳定性。
def test_patch_embedding_default_shape_and_finite_values():
    """Check token count, feature dimension, and numerical stability with defaults."""
    module = PatchEmbedding()
    images = torch.randn(2, 3, 32, 32)

    tokens = module(images)

    assert tokens.shape == (2, 64, 192)
    assert torch.isfinite(tokens).all()


# 使用灰度、非正方形图片和非默认配置，防止实现把 3、4、64、192 写死。
def test_patch_embedding_custom_configuration():
    """Use a grayscale, non-square input and custom values to catch hard-coded sizes."""
    batch_size = 2
    in_channels = 1
    height = 6
    width = 10
    patch_size = 2
    embed_dim = 8

    module = PatchEmbedding(
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
    )
    images = torch.randn(
        batch_size,
        in_channels,
        height,
        width,
    )

    tokens = module(images)

    patch_rows = height // patch_size
    patch_columns = width // patch_size
    patch_count = patch_rows * patch_columns

    assert tokens.shape == (
        batch_size,
        patch_count,
        embed_dim,
    )


# 构造参数必须为正数；参数化会将下面六组错误配置分别执行一次。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"in_channels": 0},
        {"in_channels": -1},
        {"embed_dim": 0},
        {"embed_dim": -1},
        {"patch_size": 0},
        {"patch_size": -1},
    ],
)
def test_patch_embedding_rejects_non_positive_configuration(kwargs):
    """Constructor arguments must be positive; parametrization runs all six cases."""
    with pytest.raises(ValueError):
        PatchEmbedding(**kwargs)


# forward 必须拒绝结构不合法的图片，并给出能指出原因的错误信息。
@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((3, 32, 32), "expected input shape"),          # 缺少 batch 维
        ((2, 1, 32, 32), "expected 3 input channels"),  # 通道数错误
        ((2, 3, 2, 32), "at least patch_size"),         # 小于一个 patch
        ((2, 3, 30, 32), "divisible by patch_size"),    # 边缘无法完整切分
    ],
)
def test_patch_embedding_rejects_invalid_image_shape(shape, message):
    """Forward must reject malformed images with an error that identifies the cause."""
    module = PatchEmbedding()
    images = torch.randn(*shape)

    with pytest.raises(ValueError, match=message):
        module(images)


# 从 token 构造标量并反向传播，确认卷积权重和偏置确实位于梯度通路上。
def test_patch_embedding_projection_receives_finite_gradients():
    """Backpropagate a scalar from tokens to verify the projection's gradient path."""
    module = PatchEmbedding()
    images = torch.randn(2, 3, 32, 32)

    tokens = module(images)
    loss = tokens.square().mean()
    loss.backward()

    for parameter in (
        module.projection.weight,
        module.projection.bias,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# 检查默认配置的输出接口、注意力矩阵 shape 和数值稳定性。
def test_attention_default_shapes_and_finite_values():
    """Check default output interfaces, tensor shapes, and numerical stability."""
    module = MultiHeadSelfAttention(dropout=0.0)
    tokens = torch.randn(2, 65, 192)

    output = module(tokens)
    output_with_attention, attention_weights = module(
        tokens,
        return_attention=True,
    )

    assert output.shape == (2, 65, 192)
    assert output_with_attention.shape == (2, 65, 192)
    assert attention_weights.shape == (2, 6, 65, 65)
    assert torch.allclose(output, output_with_attention)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention_weights).all()


# 每个头中的每个 Query 都应把分配给所有 Key 的权重归一化为 1。
def test_attention_weights_sum_to_one():
    """Verify that every Query distribution is normalized across all Keys."""
    module = MultiHeadSelfAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )
    tokens = torch.randn(3, 5, 8)

    _, attention_weights = module(
        tokens,
        return_attention=True,
    )
    weight_sums = attention_weights.sum(dim=-1)

    assert attention_weights.shape == (3, 2, 5, 5)
    assert torch.allclose(
        weight_sums,
        torch.ones_like(weight_sums),
        atol=1e-6,
    )


# 非默认维度和 token 数用于防止实现把 192、6、65 等默认值写死。
def test_attention_custom_configuration():
    """Use custom dimensions and token counts to catch hard-coded default values."""
    module = MultiHeadSelfAttention(
        embed_dim=12,
        num_heads=3,
        dropout=0.0,
    )
    tokens = torch.randn(4, 7, 12)

    output = module(tokens)

    assert module.head_dim == 4
    assert output.shape == (4, 7, 12)


# 检查正数、整除关系和 dropout 概率三个构造契约。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dim": 0},
        {"embed_dim": -1},
        {"num_heads": 0},
        {"num_heads": -1},
        {"embed_dim": 10, "num_heads": 3},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_attention_rejects_invalid_configuration(kwargs):
    """Reject non-positive dimensions, uneven head splits, and invalid dropout."""
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(**kwargs)


# forward 只接受 [B, N, D]，且 D 必须与模块的 embed_dim 一致。
@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((5, 192), "expected input shape"),
        ((2, 5, 191), "expected embedding dimension 192"),
    ],
)
def test_attention_rejects_invalid_token_shape(shape, message):
    """Reject malformed token tensors with an error that identifies the cause."""
    module = MultiHeadSelfAttention()
    tokens = torch.randn(*shape)

    with pytest.raises(ValueError, match=message):
        module(tokens)


# 从最终输出反向传播，确认 QKV 和输出投影都位于有效梯度通路上。
def test_attention_projections_receive_finite_gradients():
    """Backpropagate from the output through both registered projections."""
    module = MultiHeadSelfAttention(
        embed_dim=8,
        num_heads=2,
        dropout=0.0,
    )
    tokens = torch.randn(2, 5, 8)

    output = module(tokens)
    loss = output.square().mean()
    loss.backward()

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# 检查默认配置下 MLP 保持 batch、token 和最终 embedding shape。
def test_mlp_default_shape_and_finite_values():
    """Check the default output shape and numerical stability."""
    module = MLP(dropout=0.0)
    tokens = torch.randn(2, 65, 192)

    output = module(tokens)

    assert output.shape == (2, 65, 192)
    assert torch.isfinite(output).all()


# 使用非默认且不成整数倍的隐藏维度，防止实现写死 192 或 768。
def test_mlp_custom_configuration():
    """Use custom dimensions to catch hard-coded embedding and hidden sizes."""
    module = MLP(
        embed_dim=8,
        hidden_dim=13,
        dropout=0.0,
    )
    tokens = torch.randn(3, 5, 8)

    output = module(tokens)

    assert output.shape == (3, 5, 8)


# 构造维度必须为正数，dropout 必须是合法概率。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dim": 0},
        {"embed_dim": -1},
        {"hidden_dim": 0},
        {"hidden_dim": -1},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_mlp_rejects_invalid_configuration(kwargs):
    """Reject non-positive dimensions and invalid dropout probabilities."""
    with pytest.raises(ValueError):
        MLP(**kwargs)


# forward 只接受 [B, N, D]，且 D 必须与第一层 Linear 的输入一致。
@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((5, 192), "expected input shape"),
        ((2, 5, 191), "expected embedding dimension 192"),
    ],
)
def test_mlp_rejects_invalid_token_shape(shape, message):
    """Reject malformed token tensors with an error that identifies the cause."""
    module = MLP()
    tokens = torch.randn(*shape)

    with pytest.raises(ValueError, match=message):
        module(tokens)


# 从最终输出反向传播，确认两个 Linear 都位于有效梯度通路上。
def test_mlp_linear_layers_receive_finite_gradients():
    """Backpropagate from the output through both Linear layers."""
    module = MLP(
        embed_dim=8,
        hidden_dim=13,
        dropout=0.0,
    )
    tokens = torch.randn(2, 5, 8)

    output = module(tokens)
    loss = output.square().mean()
    loss.backward()

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# 两个 Dropout 都必须注册为子模块并跟随父模块切换训练状态。
def test_mlp_dropout_layers_follow_module_mode():
    """Verify that both Dropout layers follow the parent module mode."""
    module = MLP()

    module.eval()
    assert not module.dropout1.training
    assert not module.dropout2.training

    module.train()
    assert module.dropout1.training
    assert module.dropout2.training


# 检查默认 Block 的输出接口、注意力矩阵 shape 和数值稳定性。
def test_encoder_block_default_shapes_and_finite_values():
    """Check default output interfaces, tensor shapes, and numerical stability."""
    block = TransformerEncoderBlock(dropout=0.0)
    tokens = torch.randn(2, 65, 192)

    output = block(tokens)
    output_with_attention, attention_weights = block(
        tokens,
        return_attention=True,
    )

    assert output.shape == (2, 65, 192)
    assert output_with_attention.shape == (2, 65, 192)
    assert attention_weights.shape == (2, 6, 65, 65)
    assert torch.allclose(output, output_with_attention)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention_weights).all()
    assert torch.allclose(
        attention_weights.sum(dim=-1),
        torch.ones_like(attention_weights.sum(dim=-1)),
        atol=1e-6,
    )


# 手工展开两条残差路径，防止 MLP 错误地读取 Block 最初的旧输入。
def test_encoder_block_matches_sequential_pre_norm_computation():
    """Match the exact sequential pre-norm attention and MLP computation."""
    block = TransformerEncoderBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=13,
        dropout=0.0,
    )
    tokens = torch.randn(3, 5, 8)

    output = block(tokens)

    attention_output = block.attention(block.norm1(tokens))
    after_attention = tokens + attention_output
    mlp_output = block.mlp(block.norm2(after_attention))
    expected = after_attention + mlp_output

    assert output.shape == (3, 5, 8)
    assert torch.allclose(output, expected)


# norm1 和 norm2 结构相同但服务不同分支，不能共享模块或参数。
def test_encoder_block_uses_independent_layer_norms():
    """Verify that the two LayerNorm stages own independent parameters."""
    block = TransformerEncoderBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=13,
    )

    assert block.norm1 is not block.norm2
    assert block.norm1.weight is not block.norm2.weight
    assert block.norm1.bias is not block.norm2.bias


# Block 在公开入口检查维度、head 整除关系和 dropout 概率。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dim": 0},
        {"embed_dim": -1},
        {"num_heads": 0},
        {"num_heads": -1},
        {"embed_dim": 10, "num_heads": 3},
        {"mlp_dim": 0},
        {"mlp_dim": -1},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_encoder_block_rejects_invalid_configuration(kwargs):
    """Reject invalid dimensions, uneven head splits, and invalid dropout."""
    with pytest.raises(ValueError):
        TransformerEncoderBlock(**kwargs)


# forward 只接受 [B, N, D]，且 D 必须与 Block 的 embed_dim 一致。
@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((5, 192), "expected input shape"),
        ((2, 5, 191), "expected embedding dimension 192"),
    ],
)
def test_encoder_block_rejects_invalid_token_shape(shape, message):
    """Reject malformed token tensors with an error that identifies the cause."""
    block = TransformerEncoderBlock()
    tokens = torch.randn(*shape)

    with pytest.raises(ValueError, match=message):
        block(tokens)


# 从最终输出反向传播，确认归一化、Attention 和 MLP 全部位于梯度通路。
def test_encoder_block_all_parameters_receive_finite_gradients():
    """Backpropagate through both residual branches and all registered parameters."""
    block = TransformerEncoderBlock(
        embed_dim=8,
        num_heads=2,
        mlp_dim=13,
        dropout=0.0,
    )
    tokens = torch.randn(2, 5, 8)

    output = block(tokens)
    loss = output.square().mean()
    loss.backward()

    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# 检查默认 MiniViT 的分类输出、最后一层 Attention 和数值稳定性。
def test_minivit_default_shapes_and_finite_values():
    """Check default logits, final attention, and numerical stability."""
    model = MiniViT(dropout=0.0)
    images = torch.randn(2, 3, 32, 32)

    logits = model(images)
    logits_with_attention, attention_weights = model(
        images,
        return_attention=True,
    )

    assert logits.shape == (2, 10)
    assert logits_with_attention.shape == (2, 10)
    assert attention_weights.shape == (2, 6, 65, 65)
    assert torch.allclose(logits, logits_with_attention)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention_weights).all()
    assert torch.allclose(
        attention_weights.sum(dim=-1),
        torch.ones_like(attention_weights.sum(dim=-1)),
        atol=1e-6,
    )


# 使用灰度小图片和非默认宽度，证明完整模型没有写死默认配置。
def test_minivit_custom_configuration():
    """Use a compact custom model to catch hard-coded default values."""
    model = MiniViT(
        image_size=8,
        in_channels=1,
        num_classes=5,
        patch_size=2,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_dim=13,
        dropout=0.0,
    )
    images = torch.randn(3, 1, 8, 8)

    logits, attention_weights = model(
        images,
        return_attention=True,
    )

    assert model.num_patches == 16
    assert model.num_tokens == 17
    assert logits.shape == (3, 5)
    assert attention_weights.shape == (3, 2, 17, 17)


# CLS、位置编码必须注册为参数，各 Encoder Block 必须是独立实例。
def test_minivit_registers_special_parameters_and_independent_blocks():
    """Verify special parameter registration and independent encoder layers."""
    model = MiniViT(
        image_size=8,
        patch_size=2,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_dim=13,
    )
    named_parameters = dict(model.named_parameters())

    assert named_parameters["cls_token"] is model.cls_token
    assert (
        named_parameters["position_embedding"]
        is model.position_embedding
    )
    assert model.cls_token.shape == (1, 1, 8)
    assert model.position_embedding.shape == (1, 17, 8)
    assert model.cls_token.abs().max() <= 0.04
    assert model.position_embedding.abs().max() <= 0.04

    assert model.blocks[0] is not model.blocks[1]
    assert (
        model.blocks[0].attention.qkv_projection.weight
        is not model.blocks[1].attention.qkv_projection.weight
    )


# 完整模型在公开入口检查图片、Transformer 和分类配置。
@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 0},
        {"in_channels": 0},
        {"num_classes": 0},
        {"patch_size": 0},
        {"image_size": 8, "patch_size": 3},
        {"embed_dim": 0},
        {"depth": 0},
        {"num_heads": 0},
        {"embed_dim": 10, "num_heads": 3},
        {"mlp_dim": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_minivit_rejects_invalid_configuration(kwargs):
    """Reject invalid image, Transformer, and classifier configuration."""
    with pytest.raises(ValueError):
        MiniViT(**kwargs)


# forward 必须拒绝错误维数、通道数以及与位置编码不匹配的图片尺寸。
@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((3, 32, 32), "expected input shape"),
        ((2, 1, 32, 32), "expected 3 input channels"),
        ((2, 3, 28, 32), "expected image size 32x32"),
        ((2, 3, 32, 28), "expected image size 32x32"),
    ],
)
def test_minivit_rejects_invalid_image_shape(shape, message):
    """Reject images incompatible with the model and positional embedding."""
    model = MiniViT()
    images = torch.randn(*shape)

    with pytest.raises(ValueError, match=message):
        model(images)


# 从交叉熵反向传播，确认完整图片到分类 loss 的梯度链没有断开。
def test_minivit_cross_entropy_reaches_all_parameters():
    """Backpropagate cross-entropy through every registered model parameter."""
    model = MiniViT(
        image_size=8,
        in_channels=1,
        num_classes=5,
        patch_size=2,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_dim=13,
        dropout=0.0,
    )
    images = torch.randn(3, 1, 8, 8)
    labels = torch.tensor([0, 1, 4], dtype=torch.long)

    logits = model(images)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
