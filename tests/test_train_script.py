"""Tests for the training command-line interface."""

from pathlib import Path

import pytest
import torch

import scripts.train as train_script
from scripts.train import (
    DEFAULT_CONFIG_PATH,
    build_model,
    build_optimizer,
    build_scheduler,
    parse_args,
)
from src.models.minivit import MiniViT
from src.models.cnn import SimpleCNN


def write_test_config(
    tmp_path: Path,
    randomness_yaml: str = "seed: 42",
) -> Path:
    """Write a complete lightweight configuration for entry-point tests."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
experiment:
  name: isolated_test
  output_dir: outputs/isolated_test
{randomness_yaml}
device: auto
data:
  data_dir: data
  batch_size: 8
  num_workers: 0
  augmentation: basic
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

    def fake_collect_run_metadata(**kwargs):
        calls["metadata_input"] = kwargs
        return {
            "fixture": "run_metadata",
            "git_commit": "abc123",
            "git_is_dirty": False,
        }

    def fake_write_run_status(**kwargs):
        calls.setdefault("statuses", []).append(kwargs)
        return kwargs

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
    monkeypatch.setattr(
        train_script,
        "collect_run_metadata",
        fake_collect_run_metadata,
    )
    monkeypatch.setattr(
        train_script,
        "write_run_status",
        fake_write_run_status,
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


# 同一个模型工厂应根据 model.name 创建 CNN，而不修改传入的配置字典。
def test_build_model_selects_cnn_without_mutating_configuration():
    """Build SimpleCNN by name while preserving the source configuration."""

    device = torch.device("cpu")
    model_config: dict[str, object] = {
        "name": "cnn",
        "in_channels": 3,
        "num_classes": 10,
        "channels": [8, 16],
        "dropout": 0.0,
    }
    original_config = dict(model_config)

    model = build_model(
        model_config=model_config,
        device=device,
    )

    assert isinstance(model, SimpleCNN)
    assert model(torch.randn(2, 3, 16, 16)).shape == (2, 10)
    assert model_config == original_config


# 未实现的模型名必须在训练开始前明确失败，避免静默创建错误架构。
def test_build_model_rejects_unknown_model_name():
    """Reject model names not supported by the shared training entry point."""

    with pytest.raises(
        ValueError,
        match="minivit, cnn",
    ):
        build_model(
            model_config={"name": "resnet"},
            device=torch.device("cpu"),
        )


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


# scheduler 的总步数必须使用实际执行的 batch 上限，而不是完整 loader 长度。
def test_build_scheduler_uses_effective_optimizer_update_count():
    """Convert epochs and bounded batches into exact warmup and total steps."""

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    training_config: dict[str, object] = {
        "epochs": 4,
        "scheduler": {
            "name": "warmup_cosine",
            "warmup_epochs": 1,
            "start_factor": 0.1,
            "min_factor": 0.05,
        },
    }

    scheduler = build_scheduler(
        optimizer=optimizer,
        training_config=training_config,
        train_loader=[None] * 10,
        max_train_batches=3,
    )

    assert scheduler is not None
    assert scheduler.total_steps == 12
    assert scheduler.warmup_steps == 3
    assert scheduler.get_current_lrs() == pytest.approx([3e-5])


@pytest.mark.parametrize(
    "scheduler_config",
    [
        None,
        {"name": "none"},
    ],
)
def test_build_scheduler_preserves_fixed_lr_when_disabled(scheduler_config):
    """Keep legacy and explicit-none configurations on the fixed-LR path."""

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    training_config: dict[str, object] = {"epochs": 4}
    if scheduler_config is not None:
        training_config["scheduler"] = scheduler_config

    scheduler = build_scheduler(
        optimizer=optimizer,
        training_config=training_config,
        train_loader=object(),
        max_train_batches=None,
    )

    assert scheduler is None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


@pytest.mark.parametrize(
    ("scheduler_config", "message"),
    [
        ("cosine", "must be a mapping"),
        ({"name": "mystery"}, "none, warmup_cosine"),
        (
            {"name": "warmup_cosine"},
            "Missing training.scheduler keys",
        ),
        (
            {
                "name": "warmup_cosine",
                "warmup_epochs": 4,
                "start_factor": 0.1,
                "min_factor": 0.05,
            },
            "warmup_epochs",
        ),
    ],
)
def test_build_scheduler_rejects_invalid_configuration(
    scheduler_config,
    message,
):
    """Reject ambiguous scheduler names, fields, and warmup duration."""

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    with pytest.raises(ValueError, match=message):
        build_scheduler(
            optimizer=optimizer,
            training_config={
                "epochs": 4,
                "scheduler": scheduler_config,
            },
            train_loader=[None] * 3,
            max_train_batches=None,
        )


def test_resolve_project_path_handles_relative_and_absolute_paths(
    tmp_path: Path,
):
    """Anchor relative paths at the project root and preserve absolute paths."""

    assert train_script.resolve_project_path("data") == (
        train_script.PROJECT_ROOT / "data"
    )
    assert train_script.resolve_project_path(tmp_path) == tmp_path


# main 应兼容旧 seed 和解耦后的 randomness，并把两类 seed 传给正确位置。
@pytest.mark.parametrize(
    (
        "randomness_yaml",
        "expected_split_seed",
        "expected_train_seed",
    ),
    [
        ("seed: 42", 42, 42),
        (
            "randomness:\n  split_seed: 17\n  train_seed: 29",
            17,
            29,
        ),
    ],
)
def test_main_wires_fresh_training_without_real_data(
    tmp_path: Path,
    monkeypatch,
    randomness_yaml: str,
    expected_split_seed: int,
    expected_train_seed: int,
):
    """Assemble a fresh run and forward effective configuration to fit."""

    config_path = write_test_config(tmp_path, randomness_yaml)
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
    assert calls["seed"] == expected_train_seed
    assert calls["device_name"] == "auto"
    assert calls["data"] == {
        "data_dir": train_script.PROJECT_ROOT / "data",
        "batch_size": 8,
        "num_workers": 0,
        "augmentation": "basic",
        "split_seed": expected_split_seed,
        "train_seed": expected_train_seed,
    }

    fit_call = calls["fit"]
    assert fit_call["epochs"] == 2
    assert fit_call["start_epoch"] == 1
    assert fit_call["initial_history"] is None
    assert fit_call["best_val_accuracy"] == float("-inf")
    assert fit_call["max_train_batches"] is None
    assert fit_call["max_val_batches"] is None
    assert fit_call["verbose"] is True
    assert fit_call["gradient_clip_norm"] is None
    assert fit_call["batch_mixer"] is None
    assert fit_call["train_criterion"].label_smoothing == pytest.approx(0.0)
    assert fit_call["val_criterion"].label_smoothing == pytest.approx(0.0)
    assert fit_call["output_dir"] == (
        train_script.PROJECT_ROOT / "outputs" / "isolated_test"
    )
    assert fit_call["config"]["training"]["epochs"] == 2
    assert fit_call["config"]["training"]["label_smoothing"] == pytest.approx(
        0.0
    )
    assert fit_call["config"]["training"]["batch_mixing"] == {"name": "none"}
    assert fit_call["config"]["format_version"] == 2
    assert fit_call["config"]["runtime"] == {
        "is_smoke_test": False,
        "run_type": "formal",
        "max_train_batches": None,
        "max_val_batches": None,
        "amp": {
            "requested": False,
            "enabled": False,
            "device_type": "cpu",
            "dtype": None,
            "uses_grad_scaler": False,
            "fallback_reason": None,
        },
        "metadata": {
            "fixture": "run_metadata",
            "git_commit": "abc123",
            "git_is_dirty": False,
        },
    }
    assert calls["metadata_input"] == {
        "project_root": train_script.PROJECT_ROOT,
        "device": torch.device("cpu"),
        "split_seed": expected_split_seed,
        "train_seed": expected_train_seed,
        "is_smoke_test": False,
    }
    assert calls["save_config"]["path"] == (
        train_script.PROJECT_ROOT
        / "outputs"
        / "isolated_test"
        / "config.yaml"
    )
    assert calls["save_config"]["config"] is fit_call["config"]
    assert fit_call["amp_runtime"].enabled is False
    assert calls["order"] == ["save_config", "fit"]
    assert [item["status"] for item in calls["statuses"]] == [
        "running",
        "completed",
    ]
    assert calls["statuses"][-1]["completed_epochs"] == 2


# YAML 中的 label_smoothing 只允许改变训练 criterion，并写入有效配置。
def test_main_applies_label_smoothing_to_training_only(
    tmp_path: Path,
    monkeypatch,
):
    """Wire a normalized smoothing factor to training but not validation."""

    config_path = write_test_config(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  weight_decay: 0.0\n",
            "  weight_decay: 0.0\n  label_smoothing: 0.1\n",
        ),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}
    install_runtime_stubs(monkeypatch, calls)

    def fake_fit(**kwargs):
        calls["fit"] = kwargs
        return [{
            "epoch": 5,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 1.1,
            "val_accuracy": 0.4,
        }]

    monkeypatch.setattr(train_script, "fit", fake_fit)

    train_script.main(["--config", str(config_path)])

    fit_call = calls["fit"]
    assert fit_call["train_criterion"].label_smoothing == pytest.approx(0.1)
    assert fit_call["val_criterion"].label_smoothing == pytest.approx(0.0)
    assert fit_call["config"]["training"]["label_smoothing"] == pytest.approx(
        0.1
    )


# Mixup YAML 必须构建训练专用 mixer，并把规范化参数写入配置快照。
def test_main_builds_mixup_batch_mixer_from_yaml(
    tmp_path: Path,
    monkeypatch,
):
    """Build and forward a reproducible Mixup strategy from configuration."""

    config_path = write_test_config(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  weight_decay: 0.0\n",
            "  weight_decay: 0.0\n"
            "  batch_mixing:\n"
            "    name: mixup\n"
            "    alpha: 0.2\n",
        ),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}
    install_runtime_stubs(monkeypatch, calls)

    def fake_fit(**kwargs):
        calls["fit"] = kwargs
        return [{
            "epoch": 5,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 1.1,
            "val_accuracy": 0.4,
        }]

    monkeypatch.setattr(train_script, "fit", fake_fit)

    train_script.main(["--config", str(config_path)])

    fit_call = calls["fit"]
    assert fit_call["batch_mixer"].num_classes == 10
    assert fit_call["batch_mixer"].alpha == pytest.approx(0.2)
    assert fit_call["config"]["training"]["batch_mixing"] == {
        "name": "mixup",
        "alpha": pytest.approx(0.2),
    }


# 调度器工厂创建出的对象，必须原样传入 fit，不能只创建却未参与训练。
def test_main_forwards_built_scheduler_to_fit(
    tmp_path: Path,
    monkeypatch,
):
    """Forward the scheduler built by the entry point into the training loop."""

    config_path = write_test_config(tmp_path)
    calls: dict[str, object] = {}
    install_runtime_stubs(monkeypatch, calls)
    scheduler = object()

    def fake_build_scheduler(**kwargs):
        calls["scheduler_factory"] = kwargs
        return scheduler

    def fake_fit(**kwargs):
        calls["fit"] = kwargs
        return [{
            "epoch": 5,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 1.1,
            "val_accuracy": 0.4,
        }]

    monkeypatch.setattr(
        train_script,
        "build_scheduler",
        fake_build_scheduler,
    )
    monkeypatch.setattr(train_script, "fit", fake_fit)

    train_script.main(["--config", str(config_path)])

    assert calls["scheduler_factory"]["optimizer"] == "optimizer"
    assert calls["scheduler_factory"]["train_loader"] == "train_loader"
    assert calls["scheduler_factory"]["max_train_batches"] is None
    assert calls["fit"]["scheduler"] is scheduler


# YAML 中的裁剪阈值必须经过验证并传入正式训练循环。
def test_main_forwards_enabled_gradient_clip_norm(
    tmp_path: Path,
    monkeypatch,
):
    """Normalize and forward an enabled global gradient clipping threshold."""

    config_path = write_test_config(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "  weight_decay: 0.0\n",
            "  weight_decay: 0.0\n  gradient_clip_norm: 1\n",
        ),
        encoding="utf-8",
    )
    calls: dict[str, object] = {}
    install_runtime_stubs(monkeypatch, calls)

    def fake_fit(**kwargs):
        calls["fit"] = kwargs
        return [{
            "epoch": 5,
            "train_loss": 1.0,
            "train_accuracy": 0.5,
            "val_loss": 1.1,
            "val_accuracy": 0.4,
        }]

    monkeypatch.setattr(train_script, "fit", fake_fit)

    train_script.main(["--config", str(config_path)])

    assert calls["fit"]["gradient_clip_norm"] == 1.0


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
        "run_type": "smoke",
        "max_train_batches": 2,
        "max_val_batches": 2,
        "amp": {
            "requested": False,
            "enabled": False,
            "device_type": "cpu",
            "dtype": None,
            "uses_grad_scaler": False,
            "fallback_reason": None,
        },
        "metadata": {
            "fixture": "run_metadata",
            "git_commit": "abc123",
            "git_is_dirty": False,
        },
    }
    assert calls["metadata_input"]["is_smoke_test"] is True
    assert calls["save_config"]["path"] == (
        train_script.PROJECT_ROOT
        / "outputs"
        / "stage5_smoke"
        / "config.yaml"
    )
    assert calls["order"] == ["save_config", "fit"]
    assert calls["statuses"][-1]["status"] == "completed"
    assert calls["statuses"][-1]["is_smoke_test"] is True
    assert calls["statuses"][-1]["run_type"] == "smoke"


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
    assert calls["load_checkpoint"]["scaler"] is None

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


# 训练异常必须写入 failed 状态，同时把原异常继续抛给调用者。
def test_main_records_failed_status_and_reraises_error(
    tmp_path: Path,
    monkeypatch,
):
    """Persist failure metadata without swallowing the training exception."""

    config_path = write_test_config(tmp_path)
    calls: dict[str, object] = {}
    install_runtime_stubs(monkeypatch, calls)

    def fail_fit(**kwargs):
        raise RuntimeError("training exploded")

    monkeypatch.setattr(train_script, "fit", fail_fit)

    with pytest.raises(RuntimeError, match="training exploded"):
        train_script.main(["--config", str(config_path)])

    assert [item["status"] for item in calls["statuses"]] == [
        "running",
        "failed",
    ]
    failed = calls["statuses"][-1]
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "training exploded"
