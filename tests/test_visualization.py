"""Tests for Stage 8 visualization calculations and figure contracts."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.visualization import (
    attention_to_heatmaps,
    build_confusion_matrix,
    plot_attention_overlays,
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
