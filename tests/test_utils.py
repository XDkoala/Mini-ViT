"""Tests for experiment persistence utilities."""

import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.utils import (
    BASE_HISTORY_FIELDS,
    HISTORY_FIELDS,
    OPTIONAL_HISTORY_FIELDS,
    collect_run_metadata,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_history_csv,
    set_seed,
    write_run_status,
)


class StateProbe:
    """Small scheduler/scaler stand-in used to test state round-trips."""

    def __init__(self, value: int):
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


# 元数据必须只包含可写入 YAML/JSON 的事实，并明确记录两类 seed。
def test_collect_run_metadata_records_reproducibility_facts(monkeypatch):
    """Collect stable environment fields without embedding runtime objects."""

    monkeypatch.setattr(
        "src.utils._read_git_commit",
        lambda project_root: "abc123",
    )
    monkeypatch.setattr(
        "src.utils._read_git_is_dirty",
        lambda project_root: False,
    )

    metadata = collect_run_metadata(
        project_root=Path("."),
        device=torch.device("cpu"),
        split_seed=17,
        train_seed=29,
        is_smoke_test=True,
    )

    assert metadata["git_commit"] == "abc123"
    assert metadata["git_is_dirty"] is False
    assert metadata["split_seed"] == 17
    assert metadata["train_seed"] == 29
    assert metadata["is_smoke_test"] is True
    assert metadata["device_type"] == "cpu"
    assert isinstance(metadata["device_name"], str)
    assert metadata["python_version"]
    assert metadata["torch_version"] == str(torch.__version__)
    assert metadata["platform"]
    datetime.fromisoformat(metadata["created_at_utc"])


# Git 不可用不应阻止训练，缺失信息用 null 明确表示。
def test_collect_run_metadata_allows_unavailable_git(monkeypatch):
    """Represent an unavailable Git commit as None instead of failing a run."""

    monkeypatch.setattr(
        "src.utils._read_git_commit",
        lambda project_root: None,
    )
    monkeypatch.setattr(
        "src.utils._read_git_is_dirty",
        lambda project_root: None,
    )

    metadata = collect_run_metadata(
        project_root=Path("."),
        device=torch.device("cpu"),
        split_seed=1,
        train_seed=2,
        is_smoke_test=False,
    )

    assert metadata["git_commit"] is None
    assert metadata["git_is_dirty"] is None


# 状态文件使用固定 schema，并通过临时文件替换避免读到半份 JSON。
def test_write_run_status_atomically_replaces_lifecycle_state(tmp_path: Path):
    """Write running then completed state and compute aggregation eligibility."""

    status_path = tmp_path / "run" / "run_status.json"
    common = {
        "path": status_path,
        "experiment_id": "minivit_seed_42",
        "is_smoke_test": False,
        "git_commit": "abc123",
        "git_is_dirty": False,
        "target_epochs": 2,
    }

    running = write_run_status(**common, status="running")
    assert running["aggregation_eligible"] is False
    assert running["error"] is None

    completed = write_run_status(
        **common,
        status="completed",
        completed_epochs=2,
    )
    with status_path.open(encoding="utf-8") as file:
        persisted = json.load(file)

    assert persisted == completed
    assert persisted["format_version"] == 2
    assert persisted["run_type"] == "formal"
    assert persisted["aggregation_eligible"] is True
    assert not status_path.with_name("run_status.json.tmp").exists()


# smoke、screening、dirty 和失败运行都不能进入正式聚合。
@pytest.mark.parametrize(
    ("status", "is_smoke_test", "git_is_dirty", "error_type", "run_type"),
    [
        ("completed", True, False, None, "smoke"),
        ("completed", False, False, None, "screening"),
        ("completed", False, True, None, "formal"),
        ("failed", False, False, "RuntimeError", "formal"),
    ],
)
def test_write_run_status_excludes_ineligible_runs(
    tmp_path: Path,
    status,
    is_smoke_test,
    git_is_dirty,
    error_type,
    run_type,
):
    """Mark diagnostic, uncommitted, and failed runs as ineligible."""

    payload = write_run_status(
        tmp_path / f"{status}.json",
        status=status,
        experiment_id="test",
        is_smoke_test=is_smoke_test,
        git_commit="abc123",
        git_is_dirty=git_is_dirty,
        target_epochs=1,
        run_type=run_type,
        completed_epochs=1 if status == "completed" else None,
        error_type=error_type,
        error_message="boom" if error_type else None,
    )

    assert payload["aggregation_eligible"] is False


def test_write_run_status_rejects_inconsistent_terminal_states(tmp_path: Path):
    """Reject false completion and failures without an error type."""

    common = {
        "path": tmp_path / "status.json",
        "experiment_id": "test",
        "is_smoke_test": False,
        "git_commit": "abc123",
        "git_is_dirty": False,
        "target_epochs": 2,
    }

    with pytest.raises(ValueError, match="completed_epochs equal"):
        write_run_status(
            **common,
            status="completed",
            completed_epochs=1,
        )

    with pytest.raises(ValueError, match="include error_type"):
        write_run_status(**common, status="failed")


def test_set_seed_reproduces_python_numpy_and_torch_sequences():
    """Restart all supported random number sequences from the same seed."""

    set_seed(42)
    first_python = [random.random() for _ in range(3)]
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    set_seed(42)
    second_python = [random.random() for _ in range(3)]
    second_numpy = np.random.rand(3)
    second_torch = torch.rand(3)

    assert first_python == second_python
    assert np.array_equal(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_set_seed_rejects_values_outside_shared_range(seed):
    """Reject seeds that cannot be shared by all supported generators."""

    with pytest.raises(
        ValueError,
        match=r"range \[0, 2\*\*32\)",
    ):
        set_seed(seed)


@pytest.mark.parametrize(
    ("cuda_available", "expected_type"),
    [
        (True, "cuda"),
        (False, "cpu"),
    ],
)
def test_resolve_device_auto_uses_available_hardware(
    monkeypatch,
    cuda_available,
    expected_type,
):
    """Resolve auto to CUDA when available and otherwise to CPU."""

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: cuda_available,
    )

    assert resolve_device(" auto ").type == expected_type


def test_resolve_device_accepts_explicit_cpu(monkeypatch):
    """Honor an explicit CPU request even when CUDA is available."""

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    assert resolve_device("CPU").type == "cpu"


def test_resolve_device_accepts_available_cuda(monkeypatch):
    """Honor an explicit CUDA request when CUDA is available."""

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )

    assert resolve_device("CUDA").type == "cuda"


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    """Fail instead of silently falling back from explicit CUDA to CPU."""

    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA was requested but is not available",
    ):
        resolve_device("cuda")


def test_resolve_device_rejects_unknown_name():
    """Reject device names outside the supported configuration contract."""

    with pytest.raises(
        ValueError,
        match="auto, cpu, cuda",
    ):
        resolve_device("gpu")


# CSV 写入器应创建父目录，并保持固定列顺序与全部 epoch 数据。
def test_save_history_csv_writes_complete_table(tmp_path: Path):
    """Create missing directories and preserve all history records."""

    history: list[dict[str, int | float]] = [
        {
            "epoch": 1,
            "train_loss": 1.8,
            "train_accuracy": 0.4,
            "val_loss": 1.7,
            "val_accuracy": 0.45,
            "epoch_time_seconds": 12.3,
        },
        {
            "epoch": 2,
            "train_loss": 1.5,
            "train_accuracy": 0.55,
            "val_loss": 1.4,
            "val_accuracy": 0.6,
            "epoch_time_seconds": 11.8,
        },
    ]
    output_path = tmp_path / "nested" / "run" / "history.csv"

    save_history_csv(history, output_path)

    assert output_path.is_file()

    with output_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == HISTORY_FIELDS
    assert rows == [
        {
            "epoch": "1",
            "train_loss": "1.8",
            "train_accuracy": "0.4",
            "val_loss": "1.7",
            "val_accuracy": "0.45",
            "epoch_time_seconds": "12.3",
            "learning_rate": "",
            "gradient_norm": "",
            "gradient_clip_fraction": "",
            "peak_gpu_memory_allocated_mb": "",
            "peak_gpu_memory_reserved_mb": "",
        },
        {
            "epoch": "2",
            "train_loss": "1.5",
            "train_accuracy": "0.55",
            "val_loss": "1.4",
            "val_accuracy": "0.6",
            "epoch_time_seconds": "11.8",
            "learning_rate": "",
            "gradient_norm": "",
            "gradient_clip_fraction": "",
            "peak_gpu_memory_allocated_mb": "",
            "peak_gpu_memory_reserved_mb": "",
        },
    ]


# 续训迁移期允许旧行留空、新行填写扩展指标，但列顺序必须统一。
def test_save_history_csv_supports_mixed_base_and_extended_rows(tmp_path: Path):
    """Write heterogeneous history rows using blank optional CSV cells."""

    base_record = {
        "epoch": 1,
        "train_loss": 1.0,
        "train_accuracy": 0.5,
        "val_loss": 1.1,
        "val_accuracy": 0.4,
        "epoch_time_seconds": 2.0,
    }
    extended_record = {
        **base_record,
        "epoch": 2,
        "learning_rate": 0.0003,
        "gradient_norm": 1.25,
        "gradient_clip_fraction": 0.1,
        "peak_gpu_memory_allocated_mb": 512.5,
        "peak_gpu_memory_reserved_mb": 640.0,
    }
    output_path = tmp_path / "mixed_history.csv"

    save_history_csv([base_record, extended_record], output_path)

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert tuple(rows[0]) == HISTORY_FIELDS
    assert rows[0]["learning_rate"] == ""
    assert rows[0]["gradient_norm"] == ""
    assert rows[0]["gradient_clip_fraction"] == ""
    assert rows[1]["learning_rate"] == "0.0003"
    assert rows[1]["gradient_norm"] == "1.25"
    assert rows[1]["gradient_clip_fraction"] == "0.1"
    assert rows[0]["peak_gpu_memory_allocated_mb"] == ""
    assert rows[0]["peak_gpu_memory_reserved_mb"] == ""
    assert rows[1]["peak_gpu_memory_allocated_mb"] == "512.5"
    assert rows[1]["peak_gpu_memory_reserved_mb"] == "640.0"
    assert HISTORY_FIELDS == (
        *BASE_HISTORY_FIELDS,
        *OPTIONAL_HISTORY_FIELDS,
    )


# 缺失基础指标或悄悄加入未知列都会破坏聚合契约，应在写盘前失败。
@pytest.mark.parametrize(
    ("record_update", "removed_field", "message"),
    [
        ({}, "val_loss", "missing required fields: val_loss"),
        ({"mystery_metric": 1.0}, None, "unknown fields: mystery_metric"),
    ],
)
def test_save_history_csv_rejects_invalid_row_schema(
    tmp_path: Path,
    record_update,
    removed_field,
    message,
):
    """Reject incomplete or undeclared history rows before creating a CSV."""

    record = {
        "epoch": 1,
        "train_loss": 1.0,
        "train_accuracy": 0.5,
        "val_loss": 1.1,
        "val_accuracy": 0.4,
        "epoch_time_seconds": 2.0,
        **record_update,
    }
    if removed_field is not None:
        record.pop(removed_field)

    output_path = tmp_path / "invalid_history.csv"
    with pytest.raises(ValueError, match=message):
        save_history_csv([record], output_path)

    assert not output_path.exists()


# 空 history 通常意味着训练尚未发生，不应生成误导性的日志文件。
def test_save_history_csv_rejects_empty_history(tmp_path: Path):
    """Reject empty history instead of writing a header-only CSV file."""

    output_path = tmp_path / "history.csv"

    with pytest.raises(
        ValueError,
        match="at least one epoch",
    ):
        save_history_csv([], output_path)

    assert not output_path.exists()


# checkpoint 必须同时保存模型、优化器与恢复训练所需的元数据。
def test_save_checkpoint_preserves_complete_training_state(tmp_path: Path):
    """Persist model, optimizer, progress, history, and configuration."""

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05,
    )

    # 先完成一次更新，确保 AdamW 的 step、exp_avg、exp_avg_sq 已创建。
    inputs = torch.tensor([[1.0, -1.0]])
    labels = torch.tensor([1])
    loss = nn.CrossEntropyLoss()(model(inputs), labels)
    loss.backward()
    optimizer.step()

    history: list[dict[str, int | float]] = [
        {
            "epoch": 1,
            "train_loss": loss.item(),
            "train_accuracy": 1.0,
            "val_loss": 0.8,
            "val_accuracy": 0.75,
        }
    ]
    config: dict[str, object] = {
        "seed": 42,
        "model": {
            "name": "test_linear",
            "num_classes": 2,
        },
    }
    checkpoint_path = tmp_path / "nested" / "last.pt"

    expected_model_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    expected_optimizer_state = optimizer.state_dict()

    save_checkpoint(
        path=checkpoint_path,
        epoch=1,
        model=model,
        optimizer=optimizer,
        best_val_accuracy=0.75,
        history=history,
        config=config,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert set(checkpoint) == {
        "format_version",
        "epoch",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "best_val_accuracy",
        "history",
        "config",
        "run_metadata",
    }
    assert checkpoint["format_version"] == 2
    assert checkpoint["scheduler_state"] is None
    assert checkpoint["scaler_state"] is None
    assert checkpoint["run_metadata"] is None
    assert checkpoint["epoch"] == 1
    assert checkpoint["best_val_accuracy"] == pytest.approx(0.75)
    assert checkpoint["history"] == history
    assert checkpoint["config"] == config

    for name, expected_tensor in expected_model_state.items():
        assert torch.equal(
            checkpoint["model_state"][name],
            expected_tensor,
        )

    actual_optimizer_state = checkpoint["optimizer_state"]
    assert (
        actual_optimizer_state["param_groups"]
        == expected_optimizer_state["param_groups"]
    )
    assert (
        actual_optimizer_state["state"].keys()
        == expected_optimizer_state["state"].keys()
    )

    for parameter_id, expected_state in expected_optimizer_state["state"].items():
        actual_state = actual_optimizer_state["state"][parameter_id]
        assert actual_state.keys() == expected_state.keys()

        for state_name, expected_value in expected_state.items():
            actual_value = actual_state[state_name]

            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


# 加载后，新对象应恢复源对象的权重、优化器动量和训练元数据。
def test_load_checkpoint_restores_state_and_can_continue_training(
    tmp_path: Path,
):
    """Round-trip a checkpoint and continue optimization successfully."""

    device = torch.device("cpu")
    inputs = torch.tensor([[1.0, -1.0]])
    labels = torch.tensor([1])
    criterion = nn.CrossEntropyLoss()

    source_model = nn.Linear(2, 2).to(device)
    source_optimizer = torch.optim.AdamW(
        source_model.parameters(),
        lr=1e-3,
        weight_decay=0.05,
    )

    # 产生非空 AdamW 状态后保存源训练状态。
    source_loss = criterion(source_model(inputs), labels)
    source_loss.backward()
    source_optimizer.step()

    history: list[dict[str, int | float]] = [
        {
            "epoch": 1,
            "train_loss": source_loss.item(),
            "train_accuracy": 1.0,
            "val_loss": 0.8,
            "val_accuracy": 0.75,
        }
    ]
    config: dict[str, object] = {
        "seed": 42,
        "learning_rate": 1e-3,
    }
    checkpoint_path = tmp_path / "last.pt"

    save_checkpoint(
        path=checkpoint_path,
        epoch=1,
        model=source_model,
        optimizer=source_optimizer,
        best_val_accuracy=0.75,
        history=history,
        config=config,
    )

    # 目标对象使用不同参数和超参数，确保恢复并非碰巧相同。
    target_model = nn.Linear(2, 2).to(device)
    with torch.no_grad():
        target_model.weight.zero_()
        target_model.bias.zero_()

    target_optimizer = torch.optim.AdamW(
        target_model.parameters(),
        lr=9e-3,
        weight_decay=0.0,
    )
    target_weight_before = target_model.weight.detach().clone()

    resume_state = load_checkpoint(
        path=checkpoint_path,
        model=target_model,
        optimizer=target_optimizer,
        device=device,
    )

    assert resume_state == {
        "format_version": 2,
        "epoch": 1,
        "best_val_accuracy": 0.75,
        "history": history,
        "config": config,
        "scheduler_state": None,
        "scaler_state": None,
        "run_metadata": None,
    }
    assert not torch.equal(target_weight_before, target_model.weight)

    for name, source_tensor in source_model.state_dict().items():
        assert torch.equal(
            target_model.state_dict()[name],
            source_tensor,
        )

    source_optimizer_state = source_optimizer.state_dict()
    target_optimizer_state = target_optimizer.state_dict()
    assert (
        target_optimizer_state["param_groups"]
        == source_optimizer_state["param_groups"]
    )

    for parameter_id, source_state in source_optimizer_state["state"].items():
        target_state = target_optimizer_state["state"][parameter_id]

        for state_name, source_value in source_state.items():
            target_value = target_state[state_name]

            if isinstance(source_value, torch.Tensor):
                assert torch.equal(target_value, source_value)
            else:
                assert target_value == source_value

    # 恢复后的 optimizer 仍引用目标模型参数，可以继续更新。
    continued_weight_before = target_model.weight.detach().clone()
    target_optimizer.zero_grad(set_to_none=True)
    continued_loss = criterion(target_model(inputs), labels)
    continued_loss.backward()
    target_optimizer.step()

    assert not torch.equal(
        continued_weight_before,
        target_model.weight,
    )


# 历史 checkpoint 没有版本号和扩展状态，仍应按 v1 恢复基础训练状态。
def test_load_checkpoint_normalizes_legacy_v1_format(tmp_path: Path):
    """Load a manually constructed legacy checkpoint without v2 fields."""

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    legacy_checkpoint = {
        "epoch": 3,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_accuracy": 0.6,
        "history": [{"epoch": 3}],
        "config": {"seed": 42},
    }
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(legacy_checkpoint, checkpoint_path)

    target_model = nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(
        target_model.parameters(),
        lr=9e-3,
    )
    restored = load_checkpoint(
        checkpoint_path,
        target_model,
        target_optimizer,
        torch.device("cpu"),
    )

    assert restored["format_version"] == 1
    assert restored["scheduler_state"] is None
    assert restored["scaler_state"] is None
    assert restored["run_metadata"] is None
    assert restored["epoch"] == 3


# v2 应完整保存未来 scheduler/scaler 状态，且不能在恢复时静默丢弃。
def test_checkpoint_round_trips_optional_training_states(tmp_path: Path):
    """Persist and restore non-empty scheduler and scaler state dictionaries."""

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = StateProbe(7)
    scaler = StateProbe(11)
    config = {
        "format_version": 2,
        "runtime": {
            "metadata": {"git_commit": "abc123"},
        },
    }
    checkpoint_path = tmp_path / "current.pt"

    save_checkpoint(
        path=checkpoint_path,
        epoch=2,
        model=model,
        optimizer=optimizer,
        best_val_accuracy=0.8,
        history=[{"epoch": 2}],
        config=config,
        scheduler=scheduler,
        scaler=scaler,
    )

    target_model = nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=9e-3)
    target_scheduler = StateProbe(0)
    target_scaler = StateProbe(0)
    restored = load_checkpoint(
        checkpoint_path,
        target_model,
        target_optimizer,
        torch.device("cpu"),
        scheduler=target_scheduler,
        scaler=target_scaler,
    )

    assert target_scheduler.value == 7
    assert target_scaler.value == 11
    assert restored["scheduler_state"] == {"value": 7}
    assert restored["scaler_state"] == {"value": 11}
    assert restored["run_metadata"] == {"git_commit": "abc123"}

    with pytest.raises(ValueError, match="no scheduler"):
        load_checkpoint(
            checkpoint_path,
            nn.Linear(2, 2),
            torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3),
            torch.device("cpu"),
        )


# 启用 scheduler/scaler 后不能从缺失对应状态的 checkpoint 静默重启曲线。
def test_load_checkpoint_rejects_enabled_component_without_saved_state(
    tmp_path: Path,
):
    """Reject a resume that would restart newly enabled stateful components."""

    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "fixed_lr.pt"
    save_checkpoint(
        path=checkpoint_path,
        epoch=1,
        model=model,
        optimizer=optimizer,
        best_val_accuracy=0.5,
        history=[{"epoch": 1}],
        config={"seed": 42},
    )

    target_model = nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(
        target_model.parameters(),
        lr=1e-3,
    )
    with pytest.raises(ValueError, match="does not contain scheduler_state"):
        load_checkpoint(
            checkpoint_path,
            target_model,
            target_optimizer,
            torch.device("cpu"),
            scheduler=StateProbe(0),
        )
