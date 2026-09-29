"""Tests for the Stage 16 transfer-training entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.train_transfer import (
    _check_expected_hash,
    build_transfer_optimizer,
    load_transfer_config,
    main,
    parse_args,
)
from src.transfer import TransferMiniViT


def test_transfer_cli_defaults_and_overrides() -> None:
    args = parse_args([
        "--config",
        "configs/advanced/stage16/fine_tune.yaml",
        "--epochs",
        "4",
        "--max-batches",
        "2",
        "--output-dir",
        "outputs/smoke",
    ])

    assert args.config.name == "fine_tune.yaml"
    assert args.epochs == 4
    assert args.max_batches == 2
    assert args.output_dir == Path("outputs/smoke")


def test_transfer_script_can_run_directly() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_transfer.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Stage 16 transfer" in result.stdout


def test_loader_rejects_split_seed_semantics(tmp_path: Path) -> None:
    source = Path("configs/advanced/stage16/scratch.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["randomness"]["split_seed"] = 42
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="only train_seed"):
        load_transfer_config(path)


def test_hash_guard_rejects_provenance_drift(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"stage16")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _check_expected_hash(path, "0" * 64, "artifact")


def test_probe_optimizer_contains_only_classifier() -> None:
    model = TransferMiniViT(
        transfer_mode="linear_probe",
        image_size=8,
        patch_size=4,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mlp_dim=16,
        num_classes=10,
    )
    optimizer = build_transfer_optimizer(
        model,
        {
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 0.05,
        },
    )

    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {id(parameter) for parameter in model.classifier.parameters()}
    assert optimized == expected


def test_smoke_limit_requires_separate_output_before_data_access() -> None:
    with pytest.raises(ValueError, match="requires --output-dir"):
        main([
            "--config",
            "configs/advanced/stage16/scratch.yaml",
            "--max-batches",
            "1",
        ])
