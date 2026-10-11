"""Command-line entry point for configured image-classifier training."""

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from src.amp import build_amp_runtime
from src.models import build_model
from src.config import (
    CURRENT_FORMAT_VERSION,
    load_config,
    resolve_random_seeds,
    save_config,
)
from src.data import build_dataloaders
from src.engine import fit
from src.gradients import validate_gradient_clip_norm
from src.regularization import (
    build_batch_mixer,
    build_classification_criteria,
    normalize_batch_mixing_config,
    validate_label_smoothing,
)
from src.schedulers import WarmupCosineScheduler
from src.utils import (
    collect_run_metadata,
    load_checkpoint,
    resolve_device,
    set_seed,
    write_run_status,
)

# Stage 5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "minivit.yaml"
)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for model training."""

    parser = argparse.ArgumentParser(
        description=(
            "Train MiniViT from a YAML configuration."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML experiment configuration.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Override the total target epoch count "
            "from the configuration."
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint used to resume training.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override the experiment output directory."
        ),
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help=(
            "Limit both training and validation batches "
            "per epoch for smoke testing."
        ),
    )

    return parser.parse_args(argv)


def build_optimizer(
    model: nn.Module,
    training_config: dict[str, object],
) -> Optimizer:
    """Build the configured optimizer for trainable model parameters."""

    optimizer_name = str(
        training_config["optimizer"]
    ).strip().lower()

    if optimizer_name != "adamw":
        raise ValueError(
            "optimizer must currently be 'adamw'."
        )

    learning_rate = float(
        training_config["learning_rate"]
    )
    weight_decay = float(
        training_config["weight_decay"]
    )

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: Optimizer,
    training_config: dict[str, object],
    train_loader: object,
    max_train_batches: int | None,
) -> WarmupCosineScheduler | None:
    """Build an optional update-based learning-rate scheduler."""

    scheduler_config = training_config.get("scheduler")
    if scheduler_config is None:
        return None
    if not isinstance(scheduler_config, Mapping):
        raise ValueError("training.scheduler must be a mapping.")

    scheduler_name = str(
        scheduler_config.get("name", "")
    ).strip().lower()
    if scheduler_name == "none":
        return None
    if scheduler_name != "warmup_cosine":
        raise ValueError(
            "training.scheduler.name must be one of: "
            "none, warmup_cosine."
        )

    required_keys = {
        "warmup_epochs",
        "start_factor",
        "min_factor",
    }
    missing_keys = required_keys - scheduler_config.keys()
    if missing_keys:
        names = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing training.scheduler keys: {names}")

    epochs = training_config["epochs"]
    warmup_epochs = scheduler_config["warmup_epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("training.epochs must be a positive integer.")
    if (
        isinstance(warmup_epochs, bool)
        or not isinstance(warmup_epochs, int)
        or not 0 <= warmup_epochs < epochs
    ):
        raise ValueError(
            "training.scheduler.warmup_epochs must be an integer "
            "in the range [0, training.epochs)."
        )

    try:
        available_steps = len(train_loader)
    except TypeError as error:
        raise ValueError("train_loader must define its batch count.") from error
    if available_steps <= 0:
        raise ValueError("train_loader must contain at least one batch.")

    steps_per_epoch = (
        available_steps
        if max_train_batches is None
        else min(available_steps, max_train_batches)
    )
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    return WarmupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        start_factor=scheduler_config["start_factor"],
        min_factor=scheduler_config["min_factor"],
    )


def resolve_project_path(
    path_value: object,
) -> Path:
    """Resolve a configured path relative to the project root."""

    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def main(
    argv: list[str] | None = None,
) -> list[dict[str, int | float]]:
    """Run a configured MiniViT training experiment."""

    args = parse_args(argv)
    config = load_config(args.config)

    experiment_config = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    # 命令行 epochs 只覆盖本次运行，并写入有效配置快照。
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError(
                "--epochs must be greater than zero."
            )

        training_config["epochs"] = args.epochs

    if (
        args.max_batches is not None
        and args.max_batches <= 0
    ):
        raise ValueError(
            "--max-batches must be greater than zero."
        )

    # 只要限制 max_batch，就一定属于诊断运行，
    # 不能默认写进正式基线目录，因此要有额外的 output_dir。
    if (
        args.max_batches is not None
        and args.output_dir is None
    ):
        raise ValueError(
            "--max-batches requires an explicit "
            "--output-dir to protect formal results."
        )

    split_seed, train_seed = resolve_random_seeds(config)

    # train_seed 控制模型初始化、数据增强等训练随机过程；
    # split_seed 只在下方交给数据层决定训练集/验证集成员。
    set_seed(train_seed)

    device = resolve_device(
        str(config["device"])
    )
    amp_runtime = build_amp_runtime(
        training_config.get("amp", False),
        device,
    )

    data_dir = resolve_project_path(
        data_config["data_dir"]
    )

    if args.output_dir is not None:
        experiment_config["output_dir"] = str(
            args.output_dir
        )

    output_dir = resolve_project_path(
        experiment_config["output_dir"]
    )

    is_smoke_test = args.max_batches is not None
    configured_run_type = str(
        experiment_config.get("run_type", "formal")
    ).strip().lower()
    if configured_run_type not in {"formal", "screening"}:
        raise ValueError(
            "experiment.run_type must be 'formal' or 'screening'."
        )
    run_type = "smoke" if is_smoke_test else configured_run_type
    experiment_config["run_type"] = configured_run_type

    # 输入 YAML 可以是旧格式；本次新生成的有效配置属于当前产物格式。
    config["format_version"] = CURRENT_FORMAT_VERSION
    config["runtime"] = {
        "is_smoke_test": is_smoke_test,
        "run_type": run_type,
        "max_train_batches": args.max_batches,
        "max_val_batches": args.max_batches,
        "amp": amp_runtime.to_metadata(),
        "metadata": collect_run_metadata(
            project_root=PROJECT_ROOT,
            device=device,
            split_seed=split_seed,
            train_seed=train_seed,
            is_smoke_test=is_smoke_test,
        ),
    }

    run_metadata = config["runtime"]["metadata"]
    status_path = output_dir / "run_status.json"
    target_epochs = int(training_config["epochs"])
    gradient_clip_norm = validate_gradient_clip_norm(
        training_config.get("gradient_clip_norm")
    )
    label_smoothing = validate_label_smoothing(
        training_config.get("label_smoothing", 0.0)
    )
    batch_mixing_config = normalize_batch_mixing_config(
        training_config.get("batch_mixing")
    )
    if label_smoothing > 0.0 and batch_mixing_config["name"] != "none":
        raise ValueError(
            "Stage 12 single-variable experiments cannot combine "
            "label_smoothing with batch_mixing."
        )
    # 把默认值也写进有效配置快照，保证 checkpoint 可准确复现实验目标。
    training_config["label_smoothing"] = label_smoothing
    training_config["batch_mixing"] = batch_mixing_config
    status_arguments = {
        "path": status_path,
        "experiment_id": str(experiment_config["name"]),
        "is_smoke_test": is_smoke_test,
        "run_type": run_type,
        "git_commit": run_metadata.get("git_commit"),
        "git_is_dirty": run_metadata.get("git_is_dirty"),
        "target_epochs": target_epochs,
    }

    write_run_status(
        **status_arguments,
        status="running",
    )

    if amp_runtime.fallback_reason is not None:
        print(f"AMP fallback: {amp_runtime.fallback_reason}")

    try:
        train_loader, val_loader, _, _ = build_dataloaders(
            data_dir=data_dir,
            batch_size=int(data_config["batch_size"]),
            num_workers=int(data_config["num_workers"]),
            augmentation=str(
                data_config.get("augmentation", "basic")
            ),
            split_seed=split_seed,
            train_seed=train_seed,
        )

        # 创建模型、标准、优化器
        model = build_model(
            model_config=model_config,
            device=device,
        )

        train_criterion, val_criterion = build_classification_criteria(
            label_smoothing
        )
        batch_mixer = build_batch_mixer(
            batch_mixing_config,
            num_classes=int(model_config["num_classes"]),
        )

        optimizer = build_optimizer(
            model=model,
            training_config=training_config,
        )
        scheduler = build_scheduler(
            optimizer=optimizer,
            training_config=training_config,
            train_loader=train_loader,
            max_train_batches=args.max_batches,
        )

        start_epoch = 1
        initial_history = None
        best_val_accuracy = float("-inf")

        # 恢复数据
        if args.resume is not None:
            resume_state = load_checkpoint(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                scaler=amp_runtime.scaler,
            )

            start_epoch = int(
                resume_state["epoch"]
            ) + 1
            initial_history = list(
                resume_state["history"]
            )
            best_val_accuracy = float(
                resume_state["best_val_accuracy"]
            )

        # 在正式训练 fit() 前保存配置，可以知道如果失败运行采用了什么设置。
        save_config(
            config=config,
            path=output_dir / "config.yaml",
        )

        # 正式训练
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_criterion=train_criterion,
            val_criterion=val_criterion,
            optimizer=optimizer,
            device=device,
            epochs=target_epochs,
            output_dir=output_dir,
            config=config,
            start_epoch=start_epoch,
            initial_history=initial_history,
            best_val_accuracy=best_val_accuracy,
            max_train_batches=args.max_batches,
            max_val_batches=args.max_batches,
            verbose=True,
            scheduler=scheduler,
            gradient_clip_norm=gradient_clip_norm,
            amp_runtime=amp_runtime,
            batch_mixer=batch_mixer,
        )
    except BaseException as error:
        try:
            write_run_status(
                **status_arguments,
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
        except Exception as status_error:
            error.add_note(
                "Additionally failed to persist run_status.json: "
                f"{status_error}"
            )
        raise

    final_record = history[-1]
    write_run_status(
        **status_arguments,
        status="completed",
        completed_epochs=int(final_record["epoch"]),
    )

    print(
        f"Training finished on {device}. "
        f"epoch={final_record['epoch']}, "
        f"train_loss={final_record['train_loss']:.4f}, "
        f"train_accuracy={final_record['train_accuracy']:.2%}, "
        f"val_loss={final_record['val_loss']:.4f}, "
        f"val_accuracy={final_record['val_accuracy']:.2%}"
    )

    return history


if __name__ == "__main__":
    main()
