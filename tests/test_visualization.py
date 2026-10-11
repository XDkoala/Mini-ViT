"""Tests for Stage 8 visualization calculations and figure contracts."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.models.minivit import MiniViT
from src.visualization import (
    attention_rollout,
    attention_to_heatmaps,
    patch_weights_to_heatmaps,
    build_confusion_matrix,
    plot_attention_overlays,
    plot_attention_comparison,
    plot_confusion_matrix,
    plot_error_cases,
    plot_training_history,
    select_error_cases,
)


matplotlib.use("Agg")


# 训练曲线应包含 loss/accuracy 两个坐标轴，并标记真实存在的 best epoch。
def test_plot_training_history_builds_two_panels_and_validates_best_epoch():
    """Plot complete history and reject an impossible best epoch."""
    history = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 1.1,
            "val_accuracy": 0.4,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "train_accuracy": 0.7,
            "val_loss": 0.9,
            "val_accuracy": 0.6,
        },
    ]

    figure = plot_training_history(history, best_epoch=2, title="run")
    assert len(figure.axes) == 2
    assert figure.axes[1].get_ylabel() == "Accuracy (%)"
    plt.close(figure)

    with pytest.raises(ValueError, match="must exist"):
        plot_training_history(history, best_epoch=3, title="run")


# 混淆矩阵的行是真实类、列是预测类；这一方向不能在报告中画反。
def test_build_and_plot_confusion_matrix_uses_true_rows_predicted_columns():
    """Count prediction pairs and produce a labeled normalized figure."""
    records = [
        {"true_label": 0, "predicted_label": 0},
        {"true_label": 0, "predicted_label": 1},
        {"true_label": 1, "predicted_label": 1},
    ]
    matrix = build_confusion_matrix(records, num_classes=2)

    np.testing.assert_array_equal(matrix, [[1, 1], [0, 1]])
    figure = plot_confusion_matrix(matrix, ["cat", "dog"], "matrix")
    assert figure.axes[0].get_xlabel() == "Predicted class"
    assert figure.axes[0].get_ylabel() == "True class"
    plt.close(figure)


# 错误案例先覆盖不同混淆类型，再按置信度补足，避免一张图全是同类错误。
def test_select_error_cases_prioritizes_pair_diversity_then_confidence():
    """Choose deterministic, diverse errors and ignore correct samples."""
    records = [
        {"sample_index": 0, "true_label": 0, "predicted_label": 1,
         "confidence": 0.99, "correct": False},
        {"sample_index": 1, "true_label": 0, "predicted_label": 1,
         "confidence": 0.98, "correct": False},
        {"sample_index": 2, "true_label": 1, "predicted_label": 0,
         "confidence": 0.90, "correct": False},
        {"sample_index": 3, "true_label": 1, "predicted_label": 1,
         "confidence": 1.00, "correct": True},
    ]

    selected = select_error_cases(records, max_cases=3)
    assert [row["sample_index"] for row in selected] == [0, 2, 1]


# Attention 转换只读取 CLS→patch，并应恢复成与原图相同大小的 [0, 1] 热图。
def test_attention_to_heatmaps_restores_image_size_and_normalizes():
    """Average heads, reshape square patch tokens, and interpolate maps."""
    attention = torch.zeros(2, 3, 5, 5)
    attention[:, :, 0, 1:] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    heatmaps = attention_to_heatmaps(attention, image_size=8)

    assert heatmaps.shape == (2, 8, 8)
    assert heatmaps.min().item() == pytest.approx(0.0)
    assert heatmaps.max().item() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="square image grid"):
        attention_to_heatmaps(torch.zeros(1, 1, 6, 6), image_size=8)


def test_attention_rollout_multiplies_later_layers_on_the_left():
    """CLS→P1→P2 路径必须由后层乘先层得到，残差留下 CLS 自身路径。"""

    first = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    second = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    all_attention = torch.stack((first, second))[:, None, None]

    patch_weights = attention_rollout(all_attention)

    assert patch_weights.shape == (1, 2)
    assert torch.allclose(patch_weights, torch.tensor([[0.25, 0.25]]))
    # 两个 patch 共 0.5，另 0.5 留在 CLS 自身，不能强行归一化成 1。
    assert patch_weights.sum().item() == pytest.approx(0.5)


def test_attention_rollout_averages_heads_before_residual():
    """两个头看不同 patch 时，先平均再加入 CLS 自连接。"""

    identity = torch.eye(3)
    head_one = identity.clone()
    head_two = identity.clone()
    head_one[0] = torch.tensor([0.0, 1.0, 0.0])
    head_two[0] = torch.tensor([0.0, 0.0, 1.0])
    all_attention = torch.stack((head_one, head_two))[None, None]

    assert torch.allclose(
        attention_rollout(all_attention), torch.tensor([[0.25, 0.25]])
    )


def test_attention_rollout_rejects_invalid_weight_maps():
    """只接受有限、非负、每行归一化的完整多层注意力。"""

    valid = torch.eye(3)[None, None, None]
    with pytest.raises(ValueError, match="shape"):
        attention_rollout(valid.squeeze(0))
    with pytest.raises(ValueError, match="sum to one"):
        attention_rollout(torch.zeros_like(valid))
    invalid = valid.clone()
    invalid[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        attention_rollout(invalid)
    invalid = valid.clone()
    invalid[0, 0, 0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        attention_rollout(invalid)


def test_attention_rollout_accepts_real_minivit_layers():
    """模型实际返回的多层注意力可直接计算 CLS→patch 权重。"""

    model = MiniViT(
        image_size=8,
        patch_size=4,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_dim=16,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        _, all_attention = model(
            torch.randn(2, 3, 8, 8),
            return_attention=True,
            attention_layers="all",
        )
        weights = attention_rollout(all_attention)
        heatmaps = patch_weights_to_heatmaps(weights, image_size=8)

    assert weights.shape == (2, 4)
    assert heatmaps.shape == (2, 8, 8)
    assert heatmaps.min() >= 0
    assert heatmaps.max() <= 1
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert (weights.sum(dim=1) <= 1.0 + 1e-6).all()


def test_patch_weights_to_heatmaps_are_relative_display_values():
    """插值后每张图独立缩放；原始路径权重保持不变。"""

    weights = torch.tensor([
        [0.01, 0.02, 0.03, 0.04],
        [0.10, 0.20, 0.30, 0.40],
    ])
    before = weights.clone()

    maps = patch_weights_to_heatmaps(weights, image_size=8)

    assert maps.shape == (2, 8, 8)
    assert torch.allclose(maps[0], maps[1], atol=1e-6)
    assert maps.amin(dim=(1, 2)).tolist() == pytest.approx([0.0, 0.0])
    assert maps.amax(dim=(1, 2)).tolist() == pytest.approx([1.0, 1.0])
    assert torch.equal(weights, before)
    assert torch.equal(
        patch_weights_to_heatmaps(torch.full((1, 4), 0.25), 8),
        torch.zeros(1, 8, 8),
    )


def test_patch_weights_to_heatmaps_validates_grid_and_values():
    """非平方 patch 网格和无效权重不能静默画成看似正常的热图。"""

    with pytest.raises(ValueError, match="square image grid"):
        patch_weights_to_heatmaps(torch.ones(1, 3), image_size=8)
    with pytest.raises(ValueError, match="finite"):
        patch_weights_to_heatmaps(torch.tensor([[float("nan")]]), 8)
    with pytest.raises(ValueError, match="non-negative"):
        patch_weights_to_heatmaps(torch.tensor([[-0.1]]), 8)
    with pytest.raises(ValueError, match="greater than zero"):
        patch_weights_to_heatmaps(torch.ones(1, 4), image_size=0)


# 两类图片函数应接受归一化 CHW batch，并为每个样本生成可读标题。
def test_error_and_attention_figures_accept_matching_batches():
    """Render diagnostic image grids from typed prediction records."""
    images = torch.zeros(2, 3, 4, 4)
    heatmaps = torch.rand(2, 4, 4)
    records = [
        {
            "true_class": "cat",
            "predicted_class": "dog",
            "confidence": 0.8,
            "correct": False,
        },
        {
            "true_class": "ship",
            "predicted_class": "ship",
            "confidence": 0.9,
            "correct": True,
        },
    ]

    error_figure = plot_error_cases(images, records, "errors")
    attention_figure = plot_attention_overlays(
        images,
        heatmaps,
        records,
        "attention",
    )
    assert len(error_figure.axes) == 2
    assert len(attention_figure.axes) == 4
    plt.close(error_figure)
    plt.close(attention_figure)


def test_attention_comparison_labels_methods_and_checks_alignment():
    """Label both methods and reject overlays with mismatched pixel geometry."""

    # 方法标签、显示缩放和解释边界必须出现在图中，热图应对齐模型输入。
    image = torch.zeros(1, 3, 4, 4)
    maps = {"Final layer": torch.zeros(1, 4, 4), "Rollout": torch.ones(1, 4, 4)}
    records = [{"predicted_class": "cat", "confidence": 0.8}]
    figure = plot_attention_comparison(image, maps, records, "comparison")
    assert [axis.get_title() for axis in figure.axes[1:]] == ["Final layer", "Rollout"]
    assert any("not causal" in text.get_text() for text in figure.texts)
    plt.close(figure)
    with pytest.raises(ValueError, match="dimensions"):
        plot_attention_comparison(image, {"Rollout": torch.zeros(1, 2, 2)}, records, "bad")
