"""Tests for the frozen-checkpoint evaluation command-line interface."""

import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import scripts.evaluate as evaluate_script


def make_checkpoint_config(
    *,
    split_seed: int | None = None,
    train_seed: int | None = None,
) -> dict[str, object]:
    """Return a complete lightweight checkpoint configuration."""

    config: dict[str, object] = {
        "experiment": {
            "name": "evaluation_test",
            "output_dir": "outputs/evaluation_test",
        },
        "device": "auto",
        "data": {
            "data_dir": "data",
            "batch_size": 8,
            "num_workers": 0,
            "augmentation": "basic",
        },
        "model": {
            "name": "minivit",
        },
        "training": {
            "epochs": 1,
        },
    }

    if split_seed is None and train_seed is None:
        config["seed"] = 42
    elif split_seed is not None and train_seed is not None:
        config["randomness"] = {
            "split_seed": split_seed,
            "train_seed": train_seed,
        }
    else:
        raise ValueError("split_seed and train_seed must be provided together.")

    return config


# 命令行必须接收 best checkpoint，并正确解析可选 smoke 参数。
def test_parse_args_accepts_checkpoint_and_overrides():
    """Parse checkpoint, output directory, and partial-evaluation limit."""

    args = evaluate_script.parse_args([
        "--checkpoint",
        "outputs/run/best.pt",
        "--output-dir",
        "outputs/eval_smoke",
        "--max-batches",
        "2",
    ])

    assert args.checkpoint == Path("outputs/run/best.pt")
    assert args.output_dir == Path("outputs/eval_smoke")
    assert args.max_batches == 2


# 最终评估必须拒绝 last.pt，防止绕过验证集选出的最佳 epoch。
def test_load_evaluation_checkpoint_rejects_non_best_filename(tmp_path: Path):
    """Reject a checkpoint whose filename is not exactly best.pt."""

    with pytest.raises(ValueError, match="named best.pt"):
        evaluate_script.load_evaluation_checkpoint(
            tmp_path / "last.pt",
            torch.device("cpu"),
        )


# checkpoint 读取器应验证评估所需的关键字段，而非延迟到推理中失败。
def test_load_evaluation_checkpoint_validates_required_keys(tmp_path: Path):
    """Load a valid best checkpoint and reject incomplete metadata."""

    checkpoint_path = tmp_path / "best.pt"
    valid_checkpoint = {
        "epoch": 3,
        "model_state": {},
        "best_val_accuracy": 0.75,
        "history": [],
        "config": make_checkpoint_config(),
    }
    torch.save(valid_checkpoint, checkpoint_path)

    assert evaluate_script.load_evaluation_checkpoint(
        checkpoint_path,
        torch.device("cpu"),
    ) == valid_checkpoint

    torch.save({"epoch": 3}, checkpoint_path)

    with pytest.raises(ValueError, match="missing required keys"):
        evaluate_script.load_evaluation_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
        )


# v2 checkpoint 必须包含其声明的扩展字段，未知未来版本也不能猜测读取。
def test_load_evaluation_checkpoint_validates_versioned_fields(tmp_path: Path):
    """Reject incomplete current checkpoints and unsupported future formats."""

    checkpoint_path = tmp_path / "best.pt"
    base = {
        "epoch": 1,
        "model_state": {},
        "best_val_accuracy": 0.5,
        "history": [],
        "config": make_checkpoint_config(),
    }

    torch.save({**base, "format_version": 2}, checkpoint_path)
    with pytest.raises(ValueError, match="scheduler_state"):
        evaluate_script.load_evaluation_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
        )

    torch.save({**base, "format_version": 3}, checkpoint_path)
    with pytest.raises(ValueError, match="Unsupported format_version"):
        evaluate_script.load_evaluation_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
        )


# 类别名称扩充必须保留数值字段，并主动拒绝越界标签。
def test_add_class_names_maps_labels_and_rejects_invalid_values():
    """Map numeric labels to names while validating class boundaries."""

    records = [{
        "sample_index": 0,
        "true_label": 1,
        "predicted_label": 0,
        "confidence": 0.8,
        "correct": False,
    }]

    assert evaluate_script.add_class_names(records, ["cat", "dog"]) == [{
        "sample_index": 0,
        "true_label": 1,
        "true_class": "dog",
        "predicted_label": 0,
        "predicted_class": "cat",
        "confidence": 0.8,
        "correct": False,
    }]

    records[0]["predicted_label"] = 2

    with pytest.raises(ValueError, match="outside class_names"):
        evaluate_script.add_class_names(records, ["cat", "dog"])


# 主入口应兼容旧 checkpoint 和解耦 seed 的新 checkpoint。
@pytest.mark.parametrize(
    ("seed_pair", "expected_split_seed", "expected_train_seed"),
    [
        (None, 42, 42),
        ((17, 29), 17, 29),
    ],
)
def test_main_uses_checkpoint_config_and_writes_evaluation_files(
    tmp_path: Path,
    monkeypatch,
    seed_pair: tuple[int, int] | None,
    expected_split_seed: int,
    expected_train_seed: int,
):
    """Evaluate from embedded config and persist JSON and CSV artifacts."""

    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.touch()
    output_dir = tmp_path / "evaluation"
    config = (
        make_checkpoint_config()
        if seed_pair is None
        else make_checkpoint_config(
            split_seed=seed_pair[0],
            train_seed=seed_pair[1],
        )
    )
    model = nn.Linear(1, 2)
    checkpoint = {
        "epoch": 7,
        "model_state": model.state_dict(),
        "best_val_accuracy": 0.75,
        "history": [],
        "config": config,
    }
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        evaluate_script,
        "load_evaluation_checkpoint",
        lambda path, device: checkpoint,
    )
    monkeypatch.setattr(
        evaluate_script,
        "compute_sha256",
        lambda path: "ABC123",
    )
    monkeypatch.setattr(
        evaluate_script,
        "set_seed",
        lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setattr(
        evaluate_script,
        "resolve_device",
        lambda name: torch.device("cpu"),
    )

    def fake_build_dataloaders(**kwargs):
        calls["data"] = kwargs
        return "train", "val", "test", ["cat", "dog"]

    def fake_build_model(**kwargs):
        calls["model"] = kwargs
        return model

    def fake_evaluate(**kwargs):
        calls["evaluation"] = kwargs
        return (
            {"loss": 0.25, "accuracy": 0.5, "num_samples": 2},
            [
                {
                    "sample_index": 0,
                    "true_label": 0,
                    "predicted_label": 0,
                    "confidence": 0.9,
                    "correct": True,
                },
                {
                    "sample_index": 1,
                    "true_label": 1,
                    "predicted_label": 0,
                    "confidence": 0.7,
                    "correct": False,
                },
            ],
        )

    monkeypatch.setattr(
        evaluate_script,
        "build_dataloaders",
        fake_build_dataloaders,
    )
    monkeypatch.setattr(evaluate_script, "build_model", fake_build_model)
    monkeypatch.setattr(
        evaluate_script,
        "evaluate_with_predictions",
        fake_evaluate,
    )

    metrics = evaluate_script.main([
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
    ])

    assert calls["seed"] == expected_train_seed
    assert calls["data"]["batch_size"] == 8
    assert calls["data"]["augmentation"] == "basic"
    assert calls["data"]["split_seed"] == expected_split_seed
    assert calls["data"]["train_seed"] == expected_train_seed
    assert calls["model"]["model_config"] is config["model"]
    assert calls["evaluation"]["loader"] == "test"
    assert calls["evaluation"]["max_batches"] is None

    assert metrics["checkpoint_epoch"] == 7
    assert metrics["test_accuracy"] == pytest.approx(0.5)
    assert metrics["checkpoint_sha256"] == "ABC123"
    assert metrics["is_partial_evaluation"] is False

    with (output_dir / "metrics.json").open(encoding="utf-8") as file:
        assert json.load(file) == metrics

    with (output_dir / "predictions.csv").open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert tuple(rows[0]) == evaluate_script.PREDICTION_FIELDS
    assert rows[0]["true_class"] == "cat"
    assert rows[1]["predicted_class"] == "cat"
    assert rows[1]["correct"] == "False"


# 受限评估必须使用独立目录，防止覆盖正式的完整测试结果。
@pytest.mark.parametrize("max_batches", [0, -1])
def test_main_rejects_invalid_batch_limit(max_batches):
    """Reject non-positive partial-evaluation limits before loading a checkpoint."""

    with pytest.raises(ValueError, match="greater than zero"):
        evaluate_script.main([
            "--checkpoint",
            "outputs/run/best.pt",
            "--output-dir",
            "outputs/eval_smoke",
            "--max-batches",
            str(max_batches),
        ])


def test_main_requires_output_override_for_partial_evaluation():
    """Protect final evaluation files from bounded smoke results."""

    with pytest.raises(ValueError, match="explicit --output-dir"):
        evaluate_script.main([
            "--checkpoint",
            "outputs/run/best.pt",
            "--max-batches",
            "1",
        ])
