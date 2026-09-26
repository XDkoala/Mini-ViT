"""General utilities for experiment persistence."""

import csv
import json
import platform
import random
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from src.config import (
    CURRENT_FORMAT_VERSION,
    resolve_format_version,
)

# Stage 5

BASE_HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "epoch_time_seconds",
)

OPTIONAL_HISTORY_FIELDS = (
    "learning_rate",
    "gradient_norm",
    "gradient_clip_fraction",
    "peak_gpu_memory_allocated_mb",
    "peak_gpu_memory_reserved_mb",
)

HISTORY_FIELDS = (
    *BASE_HISTORY_FIELDS,
    *OPTIONAL_HISTORY_FIELDS,
)

RUN_STATUSES = {
    "running",
    "completed",
    "failed",
}

RUN_TYPES = {
    "formal",
    "screening",
    "smoke",
}


def _read_git_commit(project_root: str | Path) -> str | None:
    """Return the checked-out Git commit, or None outside a readable repo."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        return None

    return commit


def _read_git_is_dirty(project_root: str | Path) -> bool | None:
    """Return whether tracked or untracked changes exist, or None on failure."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return bool(result.stdout.strip())


def collect_run_metadata(
    *,
    project_root: str | Path,
    device: torch.device,
    split_seed: int,
    train_seed: int,
    is_smoke_test: bool,
) -> dict[str, object]:
    """Collect YAML-safe facts needed to identify and reproduce one run."""

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or "CPU"

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": (
            None
            if torch.version.cuda is None
            else str(torch.version.cuda)
        ),
        "platform": platform.platform(),
        "device_type": device.type,
        "device_name": device_name,
        "git_commit": _read_git_commit(project_root),
        "git_is_dirty": _read_git_is_dirty(project_root),
        "split_seed": split_seed,
        "train_seed": train_seed,
        "is_smoke_test": is_smoke_test,
    }


def write_run_status(
    path: str | Path,
    *,
    status: str,
    experiment_id: str,
    is_smoke_test: bool,
    git_commit: str | None,
    git_is_dirty: bool | None,
    target_epochs: int,
    run_type: str = "formal",
    completed_epochs: int | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, object]:
    """Atomically write one machine-readable experiment lifecycle state."""

    if status not in RUN_STATUSES:
        allowed = ", ".join(sorted(RUN_STATUSES))
        raise ValueError(f"status must be one of: {allowed}.")

    if run_type not in RUN_TYPES:
        allowed = ", ".join(sorted(RUN_TYPES))
        raise ValueError(f"run_type must be one of: {allowed}.")

    if target_epochs <= 0:
        raise ValueError("target_epochs must be greater than zero.")

    if completed_epochs is not None and not 0 <= completed_epochs <= target_epochs:
        raise ValueError(
            "completed_epochs must be between zero and target_epochs."
        )

    if status == "completed" and completed_epochs != target_epochs:
        raise ValueError(
            "A completed run must have completed_epochs equal to target_epochs."
        )

    if status == "failed" and not error_type:
        raise ValueError("A failed run must include error_type.")

    aggregation_eligible = (
        status == "completed"
        and not is_smoke_test
        and run_type == "formal"
        and git_commit is not None
        and git_is_dirty is False
    )
    payload: dict[str, object] = {
        "format_version": CURRENT_FORMAT_VERSION,
        "experiment_id": experiment_id,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "target_epochs": target_epochs,
        "completed_epochs": completed_epochs,
        "is_smoke_test": is_smoke_test,
        "run_type": run_type,
        "git_commit": git_commit,
        "git_is_dirty": git_is_dirty,
        "aggregation_eligible": aggregation_eligible,
        "error": (
            {
                "type": error_type,
                "message": error_message or "",
            }
            if status == "failed"
            else None
        ),
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.name}.tmp"
    )

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
        file.flush()

    temporary_path.replace(output_path)
    return payload


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

    allowed_fields = set(HISTORY_FIELDS)
    required_fields = set(BASE_HISTORY_FIELDS)

    # 写文件前先验证全部行，避免中途失败后留下半份 CSV。
    for row_index, record in enumerate(history, start=1):
        missing_fields = required_fields - record.keys()
        if missing_fields:
            missing_names = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"History row {row_index} is missing required fields: "
                f"{missing_names}"
            )

        unknown_fields = record.keys() - allowed_fields
        if unknown_fields:
            unknown_names = ", ".join(sorted(unknown_fields))
            raise ValueError(
                f"History row {row_index} contains unknown fields: "
                f"{unknown_names}"
            )

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
    scheduler: object | None = None,
    scaler: object | None = None,
) -> None:
    """Save a versioned checkpoint with optional future training states."""

    output_path = Path(path)

    # checkpoint 目录可能尚未由 CSV 写入器创建。
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scheduler_state = (
        None
        if scheduler is None
        else scheduler.state_dict()
    )
    scaler_state = (
        None
        if scaler is None
        else scaler.state_dict()
    )

    runtime = config.get("runtime")
    run_metadata = (
        runtime.get("metadata")
        if isinstance(runtime, Mapping)
        else None
    )

    checkpoint = {
        "format_version": CURRENT_FORMAT_VERSION,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "best_val_accuracy": best_val_accuracy,
        "history": history,
        "config": config,
        "run_metadata": run_metadata,
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
    scheduler: object | None = None,
    scaler: object | None = None,
) -> dict[str, object]:
    """Restore a legacy or current checkpoint into supplied stateful objects."""

    checkpoint_path = Path(path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    format_version = resolve_format_version(checkpoint)
    required_keys = {
        "epoch",
        "model_state",
        "optimizer_state",
        "best_val_accuracy",
        "history",
        "config",
    }
    if format_version >= 2:
        required_keys.update({
            "scheduler_state",
            "scaler_state",
            "run_metadata",
        })

    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        missing_names = ", ".join(sorted(missing_keys))
        raise ValueError(
            "Checkpoint is missing required keys: "
            f"{missing_names}"
        )

    scheduler_state = checkpoint.get("scheduler_state")
    scaler_state = checkpoint.get("scaler_state")

    # 先验证恢复目标是否齐全，再修改任何模型或优化器状态，
    # 避免失败后留下只恢复一半的内存对象。
    if scheduler_state is not None and scheduler is None:
        raise ValueError(
            "Checkpoint contains scheduler_state but no scheduler "
            "was provided."
        )

    if scheduler is not None and scheduler_state is None:
        raise ValueError(
            "A scheduler was provided but the checkpoint does not contain "
            "scheduler_state."
        )

    if scaler_state is not None and scaler is None:
        raise ValueError(
            "Checkpoint contains scaler_state but no scaler was provided."
        )

    if scaler is not None and scaler_state is None:
        raise ValueError(
            "A scaler was provided but the checkpoint does not contain "
            "scaler_state."
        )

    model.load_state_dict(
        checkpoint["model_state"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    return {
        "format_version": format_version,
        "epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["best_val_accuracy"],
        "history": checkpoint["history"],
        "config": checkpoint["config"],
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "run_metadata": checkpoint.get("run_metadata"),
    }
