"""General utilities for experiment persistence."""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

# Stage 5

HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "epoch_time_seconds",
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""

    if not 0 <= seed < 2**32:
        raise ValueError(
            "seed must be in the range [0, 2**32)."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(
    device_name: str,
) -> torch.device:
    """Resolve a configured device name to an available PyTorch device."""

    normalized_name = device_name.strip().lower()

    if normalized_name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if normalized_name == "cpu":
        return torch.device("cpu")

    if normalized_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    raise ValueError(
        "device must be one of: auto, cpu, cuda."
    )


def save_history_csv(
    history: list[dict[str, int | float]],
    path: str | Path,
) -> None:
    """Write the complete training history to a CSV file."""

    if not history:
        raise ValueError("history must contain at least one epoch.")

    output_path = Path(path)

    # 如果父目录尚不存在，则递归创建。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_FIELDS,
        )

        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    best_val_accuracy: float,
    history: list[dict[str, int | float]],
    config: dict[str, object],
) -> None:
    """Save a complete training checkpoint."""

    output_path = Path(path)

    # checkpoint 目录可能尚未由 CSV 写入器创建。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "config": config,
    }

    torch.save(
        checkpoint,
        output_path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, object]:
    """Restore model and optimizer state from a checkpoint."""

    checkpoint_path = Path(path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    return {
        "epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["best_val_accuracy"],
        "history": checkpoint["history"],
        "config": checkpoint["config"],
    }
