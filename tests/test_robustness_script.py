"""Tests for the Stage 15 robustness entry point."""

from pathlib import Path

import pytest
import torch

from scripts.robustness import (
    build_conditions,
    load_robustness_config,
    materialize_dataset,
    parse_args,
    summarize_results,
)
from src.corruptions import CORRUPTION_LEVELS, CorruptedSubset


class TinyRawDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        return torch.full((3, 4, 4), index / 4.0), index


def test_conditions_contain_clean_plus_eighteen_corruptions() -> None:
    conditions = build_conditions()

    assert len(conditions) == 19
    assert conditions[0] == {
        "corruption": "clean",
        "severity": 0,
        "parameter": None,
    }


def test_materialize_dataset_preserves_order_and_hash() -> None:
    dataset = CorruptedSubset(
        source=TinyRawDataset(),
        indices=[2, 0, 1],
        corruption_name="gaussian_noise",
        severity=1,
        corruption_seed=1501,
    )

    images_a, labels_a, hash_a = materialize_dataset(dataset, batch_size=2)
    images_b, labels_b, hash_b = materialize_dataset(dataset, batch_size=3)

    assert torch.equal(images_a, images_b)
    assert torch.equal(labels_a, torch.tensor([2, 0, 1]))
    assert torch.equal(labels_a, labels_b)
    assert hash_a == hash_b


def test_summarize_results_reports_macro_and_worst_case() -> None:
    rows = [{
        "split": "validation",
        "model": "demo",
        "corruption": "clean",
        "severity": 0,
        "accuracy": 0.9,
        "clean_accuracy_drop": 0.0,
    }]
    for corruption in CORRUPTION_LEVELS:
        for severity, accuracy in enumerate((0.8, 0.7, 0.6), start=1):
            rows.append({
                "split": "validation",
                "model": "demo",
                "corruption": corruption,
                "severity": severity,
                "accuracy": accuracy,
                "clean_accuracy_drop": 0.9 - accuracy,
            })

    summary = summarize_results(rows)[0]

    assert summary["clean_accuracy"] == pytest.approx(0.9)
    assert summary["macro_corrupted_accuracy"] == pytest.approx(0.7)
    assert summary["macro_clean_accuracy_drop"] == pytest.approx(0.2)
    assert summary["worst_accuracy"] == pytest.approx(0.6)
    assert summary["worst_severity"] == 3


def test_cli_defaults() -> None:
    args = parse_args([])

    assert args.config.name == "robustness.yaml"
    assert args.mode == "all"


def test_load_project_robustness_config() -> None:
    path = Path("configs/advanced/stage15/robustness.yaml")

    config = load_robustness_config(path)

    assert config["corruption_seed"] == 1501
    assert len(config["candidates"]) == 3


def test_config_rejects_drifted_corruption_levels(tmp_path: Path) -> None:
    project_config = Path(
        "configs/advanced/stage15/robustness.yaml"
    ).read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(
        project_config.replace(
            "brightness: [0.8, 0.6, 0.4]",
            "brightness: [0.9, 0.6, 0.4]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly match"):
        load_robustness_config(path)
