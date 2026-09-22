"""Tests for experiment persistence utilities."""

import csv
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.utils import (
    HISTORY_FIELDS,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_history_csv,
    set_seed,
)


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
        },
        {
            "epoch": "2",
            "train_loss": "1.5",
            "train_accuracy": "0.55",
            "val_loss": "1.4",
            "val_accuracy": "0.6",
            "epoch_time_seconds": "11.8",
        },
    ]


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
        "epoch",
        "model_state",
        "optimizer_state",
        "best_val_accuracy",
        "history",
        "config",
    }
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
        "epoch": 1,
        "best_val_accuracy": 0.75,
        "history": history,
        "config": config,
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
