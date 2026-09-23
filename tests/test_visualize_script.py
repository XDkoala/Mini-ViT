"""Tests for Stage 8 visualization command helpers."""

from pathlib import Path

import pytest

import scripts.visualize as visualize_script


# 命令行允许把测试生成物写到隔离目录，避免污染正式 assets。
def test_parse_args_accepts_output_and_case_counts():
    """Parse report output root and sample count overrides."""
    args = visualize_script.parse_args([
        "--output-root",
        "assets/test",
        "--error-cases",
        "4",
        "--attention-cases",
        "2",
    ])

    assert args.output_root == Path("assets/test")
    assert args.error_cases == 4
    assert args.attention_cases == 2


# CSV 中所有字段最初都是字符串，绘图前必须恢复正确的数据类型。
def test_parse_prediction_rows_restores_numeric_and_boolean_types():
    """Convert persisted prediction fields into typed values."""
    records = visualize_script.parse_prediction_rows([{
        "sample_index": "7",
        "true_label": "3",
        "true_class": "cat",
        "predicted_label": "5",
        "predicted_class": "dog",
        "confidence": "0.75",
        "correct": "False",
    }])

    assert records == [{
        "sample_index": 7,
        "true_label": 3,
        "true_class": "cat",
        "predicted_label": 5,
        "predicted_class": "dog",
        "confidence": 0.75,
        "correct": False,
    }]


# Attention 示例必须同时包含正确和错误样本，而不是只展示模型成功案例。
def test_select_attention_cases_mixes_correct_and_wrong_predictions():
    """Select a balanced deterministic attention sample."""
    records = []
    for index in range(6):
        correct = index < 3
        records.append({
            "sample_index": index,
            "true_label": index,
            "predicted_label": index if correct else (index + 1) % 10,
            "confidence": 0.9 - 0.01 * index,
            "correct": correct,
        })

    selected = visualize_script.select_attention_cases(records, max_cases=4)
    assert sum(bool(row["correct"]) for row in selected) == 2
    assert sum(not bool(row["correct"]) for row in selected) == 2

    with pytest.raises(ValueError, match="greater than one"):
        visualize_script.select_attention_cases(records, max_cases=1)
