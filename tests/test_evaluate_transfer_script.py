"""Tests for frozen Stage 16 transfer-evaluation metrics and CLI."""

from pathlib import Path

import pytest

from scripts.evaluate_transfer import classification_report, parse_args


def test_evaluation_cli_requires_split_and_output_dir() -> None:
    """评估命令必须明确选择 validation 或 test 和独立输出目录。"""

    args = parse_args(["--split", "validation", "--output-dir", "outputs/check"])
    assert args.split == "validation"
    assert args.output_dir == Path("outputs/check")


def test_macro_f1_is_unweighted_mean_of_class_f1() -> None:
    """逐类 F1 先各自计算，再等权平均；不能用整体 accuracy 代替。"""

    records = [
        {"true_label": 0, "predicted_label": 0},
        {"true_label": 0, "predicted_label": 1},
        {"true_label": 1, "predicted_label": 1},
    ]
    report = classification_report(records, ("a", "b"))
    assert report["confusion_matrix"] == [[1, 1], [0, 1]]
    assert report["per_class"][0]["f1"] == pytest.approx(2 / 3)
    assert report["per_class"][1]["f1"] == pytest.approx(2 / 3)
    assert report["macro_f1"] == pytest.approx(2 / 3)


def test_macro_f1_zero_division_and_bad_labels() -> None:
    """无样本类别按零分处理；类别越界应立即报错。"""

    report = classification_report(
        [{"true_label": 0, "predicted_label": 0}], ("a", "b")
    )
    assert report["macro_f1"] == 0.5
    with pytest.raises(ValueError, match="out-of-range"):
        classification_report([{"true_label": 0, "predicted_label": 2}], ("a", "b"))
