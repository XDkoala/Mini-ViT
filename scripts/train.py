"""Command-line entry point for MiniViT training."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from src.models.minivit import MiniViT
from src.config import load_config, save_config
from src.data import build_dataloaders
from src.engine import fit
from src.utils import (
    load_checkpoint,
    resolve_device,
    set_seed,
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


def build_model(
    model_config: dict[str, object],
    device: torch.device,
) -> MiniViT:
    """Build MiniViT from configuration and move it to a device."""

    model = MiniViT(
        **model_config,
    )

    return model.to(device)


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

    seed = int(config["seed"])
    set_seed(seed)

    device = resolve_device(
        str(config["device"])
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

    config["runtime"] = {
        "is_smoke_test": args.max_batches is not None,
        "max_train_batches": args.max_batches,
        "max_val_batches": args.max_batches,
    }

    train_loader, val_loader, _, _ = build_dataloaders(
        data_dir=data_dir,
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        seed=seed,
    )

    # 创建模型、标准、优化器
    model = build_model(
        model_config=model_config,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(
        model=model,
        training_config=training_config,
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
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(training_config["epochs"]),
        output_dir=output_dir,
        config=config,
        start_epoch=start_epoch,
        initial_history=initial_history,
        best_val_accuracy=best_val_accuracy,
        max_train_batches=args.max_batches,
        max_val_batches=args.max_batches,
        verbose=True,
    )

    final_record = history[-1]

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
