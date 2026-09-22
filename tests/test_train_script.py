"""Tests for the training command-line interface."""

from pathlib import Path

import pytest
import torch

import scripts.train as train_script
from scripts.train import (
    DEFAULT_CONFIG_PATH,
    build_model,
    build_optimizer,
    parse_args,
)
from src.models.minivit import MiniViT


def write_test_config(tmp_path: Path) -> Path:
    """Write a complete lightweight configuration for entry-point tests."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  name: isolated_test
  output_dir: outputs/isolated_test
seed: 42
device: auto
data:
  data_dir: data
  batch_size: 8
  num_workers: 0
model:
  image_size: 32
  in_channels: 3
  num_classes: 10
  patch_size: 4
  embed_dim: 16
  depth: 1
  num_heads: 4
  mlp_dim: 32
  dropout: 0.0
training:
  epochs: 5
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0
""",
        encoding="utf-8",
    )
    return config_path


def install_runtime_stubs(monkeypatch, calls: dict[str, object]) -> None:
    """Replace expensive runtime components with recording test doubles."""

    calls["order"] = []

    def fake_set_seed(seed):
        calls["seed"] = seed

    def fake_resolve_device(name):
        calls["device_name"] = name
        return torch.device("cpu")

    monkeypatch.setattr(
        train_script,
        "set_seed",
        fake_set_seed,
    )
    monkeypatch.setattr(
        train_script,
        "resolve_device",
        fake_resolve_device,
    )

    def fake_build_dataloaders(**kwargs):
        calls["data"] = kwargs
        return "train_loader", "val_loader", "test_loader", ["class"]

    monkeypatch.setattr(
        train_script,
        "build_dataloaders",
        fake_build_dataloaders,
    )
    def fake_build_model(**kwargs):
        calls["model"] = kwargs
        return "model"

    def fake_build_optimizer(**kwargs):
        calls["optimizer"] = kwargs
        return "optimizer"

    def fake_save_config(**kwargs):
        calls["save_config"] = kwargs
        calls["order"].append("save_config")

    monkeypatch.setattr(train_script, "build_model", fake_build_model)
    monkeypatch.setattr(
        train_script,
        "build_optimizer",
        fake_build_optimizer,
    )
    monkeypatch.setattr(
        train_script,
        "save_config",
        fake_save_config,
    )


# 无参数运行时，应使用项目内的正式 MiniViT 配置且不恢复训练。
def test_parse_args_uses_safe_defaults():
    """Use the baseline config without epoch override or resume path."""

    args = parse_args([])

    assert args.config == DEFAULT_CONFIG_PATH
    assert args.config.is_absolute()
    assert args.config.is_file()
    assert args.epochs is None
    assert args.resume is None


# 显式命令行参数必须转换成正确的 Path 和 int 类型。
def test_parse_args_accepts_runtime_overrides():
    """Parse config, total epochs, and checkpoint overrides."""

    args = parse_args([
        "--config",
        "configs/custom.yaml",
        "--epochs",
        "2",
        "--resume",
        "outputs/run/last.pt",
        "--output-dir",
        "outputs/smoke",
        "--max-batches",
        "3",
    ])

    assert args.config == Path("configs/custom.yaml")
    assert args.epochs == 2
    assert args.resume == Path("outputs/run/last.pt")
    assert args.output_dir == Path("outputs/smoke")
    assert args.max_batches == 3


# 工厂应按配置创建完整 MiniViT，并在创建 optimizer 前移到目标设备。
def test_build_model_and_optimizer_from_configuration():
    """Create a small MiniViT and AdamW with configured hyperparameters."""

    device = torch.device("cpu")
    model_config: dict[str, object] = {
        "image_size": 32,
        "in_channels": 3,
        "num_classes": 10,
        "patch_size": 4,
        "embed_dim": 16,
        "depth": 1,
        "num_heads": 4,
        "mlp_dim": 32,
        "dropout": 0.0,
    }
    training_config: dict[str, object] = {
        "optimizer": " AdamW ",
        "learning_rate": 0.0003,
        "weight_decay": 0.05,
    }

    model = build_model(
        model_config=model_config,
        device=device,
    )
    optimizer = build_optimizer(
        model=model,
        training_config=training_config,
    )

    assert isinstance(model, MiniViT)
    assert model.embed_dim == 16
    assert model.depth == 1
    assert next(model.parameters()).device.type == "cpu"
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0003)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)

    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
    }
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameter_ids == model_parameter_ids


# 第一版训练入口只支持 AdamW，未知名称必须明确失败。
def test_build_optimizer_rejects_unknown_name():
    """Reject optimizers not implemented by the training entry point."""

    model = torch.nn.Linear(2, 2)

    with pytest.raises(
        ValueError,
        match="currently be 'adamw'",
    ):
        build_optimizer(
            model=model,
            training_config={
                "optimizer": "sgd",
                "learning_rate": 0.1,
                "weight_decay": 0.0,
            },
        )


def test_resolve_project_path_handles_relative_and_absolute_paths(
    tmp_path: Path,
):
    """Anchor relative paths at the project root and preserve absolute paths."""

    assert train_script.resolve_project_path("data") == (
        train_script.PROJECT_ROOT / "data"
    )
    assert train_script.resolve_project_path(tmp_path) == tmp_path


# main 应按固定顺序组装新训练，并把命令行 epoch 覆盖写入有效配置。
def test_main_wires_fresh_training_without_real_data(
    tmp_path: Path,
    monkeypatch,
):
    """Assemble a fresh run and forward effective configuration to fit."""

    config_path = write_test_config(tmp_path)
    calls: dict[str, object] = {}
    install_runtime_stubs(monkeypatch, calls)

    def reject_resume(**kwargs):
        pytest.fail("load_checkpoint must not run for a fresh experiment")

    def fake_fit(**kwargs):
        calls["order"].append("fit")
        calls["fit"] = kwargs
        return [{
            "epoch": 2,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 1.1,
            "val_accuracy": 0.4,
        }]

    monkeypatch.setattr(train_script, "load_checkpoint", reject_resume)
    monkeypatch.setattr(train_script, "fit", fake_fit)

    history = train_script.main([
        "--config",
        str(config_path),
        "--epochs",
        "2",
    ])

    assert history[-1]["epoch"] == 2
    assert calls["seed"] == 42
    assert calls["device_name"] == "auto"
    assert calls["data"] == {
        "data_dir": train_script.PROJECT_ROOT / "data",
        "batch_size": 8,
        "num_workers": 0,
        "seed": 42,
    }

    fit_call = calls["fit"]
    assert fit_call["epochs"] == 2
    assert fit_call["start_epoch"] == 1
    assert fit_call["initial_history"] is None
    assert fit_call["best_val_accuracy"] == float("-inf")
    assert fit_call["max_train_batches"] is None
    assert fit_call["max_val_batches"] is None
    assert fit_call["verbose"] is True
    assert fit_call["output_dir"] == (
        train_script.PROJECT_ROOT / "outputs" / "isolated_test"
    )
    assert fit_call["config"]["training"]["epochs"] == 2
    assert fit_call["config"]["runtime"] == {
        "is_smoke_test": False,
        "max_train_batches": None,
        "max_val_batches": None,
    }
    assert calls["save_config"]["path"] == (
        train_script.PROJECT_ROOT
        / "outputs"
        / "isolated_test"
        / "config.yaml"
    )
    assert calls["save_config"]["config"] is fit_call["config"]
    assert calls["order"] == ["save_config", "fit"]


# 受限运行若未指定独立目录，必须在任何运行时组件创建前失败。
def test_main_requires_output_override_for_smoke_run(
    tmp_path: Path,
    monkeypatch,
):
    """Protect formal outputs from accidental limited-batch checkpoints."""

    config_path = write_test_config(tmp_path)

    def reject_seed_call(seed):
        pytest.fail("runtime setup must not start before safety validation")

    monkeypatch.setattr(train_script, "set_seed", reject_seed_call)

    with pytest.raises(
        ValueError,
        match="requires an explicit --output-dir",
    ):
        train_script.main([
            "--config",
            str(config_path),
            "--epochs",
            "2",
            "--max-batches",
            "2",
        ])


# 合法 smoke run 必须把目录、上限和诊断标记完整传递给 fit。
def test_main_records_and_forwards_smoke_settings(
    tmp_path: Path,
    monkeypatch,
):
    """Forward bounded-batch settings to a separate smoke output directory."""

    config_path = write_test_config(tmp_path)
    calls: dict[str, object] = {}
    install_runtime_stubs(monkeypatch, calls)

    def fake_fit(**kwargs):
        calls["order"].append("fit")
        calls["fit"] = kwargs
        return [{
            "epoch": 2,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 1.1,
            "val_accuracy": 0.4,
        }]

    monkeypatch.setattr(train_script, "fit", fake_fit)

    train_script.main([
        "--config",
        str(config_path),
        "--epochs",
        "2",
        "--max-batches",
        "2",
        "--output-dir",
        "outputs/stage5_smoke",
    ])

    fit_call = calls["fit"]
    assert fit_call["output_dir"] == (
        train_script.PROJECT_ROOT / "outputs" / "stage5_smoke"
    )
    assert fit_call["max_train_batches"] == 2
    assert fit_call["max_val_batches"] == 2
    assert Path(
        fit_call["config"]["experiment"]["output_dir"]
    ) == Path(
        "outputs/stage5_smoke"
    )
    assert fit_call["config"]["runtime"] == {
        "is_smoke_test": True,
        "max_train_batches": 2,
        "max_val_batches": 2,
    }
    assert calls["save_config"]["path"] == (
        train_script.PROJECT_ROOT
        / "outputs"
        / "stage5_smoke"
        / "config.yaml"
    )
    assert calls["order"] == ["save_config", "fit"]


# resume 元数据必须原样传给 fit，使目标总轮数和历史记录连续。
def test_main_forwards_checkpoint_resume_state(
    tmp_path: Path,
    monkeypatch,
):
    """Load checkpoint metadata and continue from the next epoch."""

    config_path = write_test_config(tmp_path)
    resume_path = tmp_path / "last.pt"
    prior_history = [
        {
            "epoch": 1,
            "train_loss": 1.5,
            "train_accuracy": 0.3,
            "val_loss": 1.4,
            "val_accuracy": 0.4,
        },
        {
            "epoch": 2,
            "train_loss": 1.2,
            "train_accuracy": 0.5,
            "val_loss": 1.1,
            "val_accuracy": 0.6,
        },
    ]
    calls: dict[str, object] = {}
    install_runtime_stubs(monkeypatch, calls)

    def fake_load_checkpoint(**kwargs):
        calls["order"].append("load_checkpoint")
        calls["load_checkpoint"] = kwargs
        return {
            "epoch": 2,
            "best_val_accuracy": 0.6,
            "history": prior_history,
            "config": {"old": True},
        }

    def fake_fit(**kwargs):
        calls["order"].append("fit")
        calls["fit"] = kwargs
        return prior_history + [{
            "epoch": 5,
            "train_loss": 0.8,
            "train_accuracy": 0.7,
            "val_loss": 0.9,
            "val_accuracy": 0.65,
        }]

    monkeypatch.setattr(
        train_script,
        "load_checkpoint",
        fake_load_checkpoint,
    )
    monkeypatch.setattr(train_script, "fit", fake_fit)

    history = train_script.main([
        "--config",
        str(config_path),
        "--resume",
        str(resume_path),
    ])

    assert history[-1]["epoch"] == 5
    assert calls["load_checkpoint"]["path"] == resume_path

    fit_call = calls["fit"]
    assert fit_call["epochs"] == 5
    assert fit_call["start_epoch"] == 3
    assert fit_call["initial_history"] == prior_history
    assert fit_call["initial_history"] is not prior_history
    assert fit_call["best_val_accuracy"] == pytest.approx(0.6)
    assert calls["order"] == [
        "load_checkpoint",
        "save_config",
        "fit",
    ]
