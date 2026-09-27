"""Tests for static and runtime benchmark helpers."""

from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.benchmark import load_benchmark_config, parse_args
from src.benchmarking import (
    build_static_profile,
    estimate_macs,
    percentile,
    summarize_samples,
    tensor_tree_bytes,
)
from src.models.cnn import SimpleCNN
from src.models.minivit import MiniViT


def test_percentile_uses_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == pytest.approx(2.5)
    assert percentile(values, 0.95) == pytest.approx(3.85)


def test_percentile_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)


def test_summarize_samples_preserves_extrema() -> None:
    summary = summarize_samples([4.0, 1.0, 3.0, 2.0])

    assert summary["mean"] == pytest.approx(2.5)
    assert summary["p50"] == pytest.approx(2.5)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(4.0)


def test_tensor_tree_bytes_counts_nested_tensors_only() -> None:
    value = {
        "a": torch.zeros(3, dtype=torch.float32),
        "b": [torch.zeros(2, dtype=torch.float16), "ignored"],
    }

    assert tensor_tree_bytes(value) == 16


class FlattenLinear(nn.Module):
    """Adapt a 4-D benchmark input to one Linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 7)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs.flatten(start_dim=1))


def test_flattened_linear_macs() -> None:
    model = FlattenLinear()

    assert estimate_macs(model, (1, 5, 1, 1)) == 35


def test_grouped_convolution_macs() -> None:
    model = nn.Conv2d(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        groups=2,
        bias=False,
    )

    # 输出为 [1, 8, 3, 3]，每个输出观察 2*3*3 个输入。
    assert estimate_macs(model, (1, 4, 5, 5)) == 8 * 3 * 3 * 18


def test_project_cnn_macs_match_manual_derivation() -> None:
    model = SimpleCNN()

    assert estimate_macs(model, (1, 3, 32, 32)) == 115_905_536


def test_project_minivit_macs_match_manual_derivation() -> None:
    model = MiniViT()

    assert estimate_macs(model, (1, 3, 32, 32)) == 182_849_664


def test_static_profile_uses_two_flops_per_mac() -> None:
    model = FlattenLinear()
    profile = build_static_profile(model, (1, 5, 1, 1))

    assert profile.parameter_count == 42
    assert profile.trainable_parameter_count == 42
    assert profile.macs_per_image == 35
    assert profile.flops_per_image == 70
    assert profile.serialized_state_dict_bytes > profile.parameter_bytes


def test_benchmark_cli_default_config() -> None:
    args = parse_args([])

    assert args.config.name == "benchmark.yaml"


def test_load_benchmark_config(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        """
device: cuda
input_shape: [3, 32, 32]
precisions: [fp32, amp]
latency: {}
throughput: {}
training_memory: {}
candidates:
  - name: demo
    checkpoint: demo.pt
output: {}
""",
        encoding="utf-8",
    )

    config = load_benchmark_config(path)

    assert config["input_shape"] == [3, 32, 32]
