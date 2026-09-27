"""Run deterministic clean/corrupted robustness evaluation for Stage 15."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as vision_functional

from scripts.train import PROJECT_ROOT, build_model, resolve_project_path
from src.corruptions import (
    CORRUPTION_LEVELS,
    CORRUPTION_NAMES,
    CorruptedSubset,
    apply_corruption,
)
from src.data import CIFAR10_VAL_SIZE, split_train_val_indices
from src.utils import collect_git_metadata, resolve_device


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "advanced"
    / "stage15"
    / "robustness.yaml"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Stage 15 robustness command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate frozen models on deterministic corruptions."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--mode",
        choices=("visualize", "evaluate", "all"),
        default="all",
    )
    return parser.parse_args(argv)


def load_robustness_config(path: Path) -> dict[str, object]:
    """Load and validate the frozen Stage 15 protocol."""

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Robustness config root must be a mapping.")

    required = {
        "device",
        "data",
        "evaluation",
        "corruption_seed",
        "corruptions",
        "candidates",
        "visualization",
        "output",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(
            "Missing robustness config keys: "
            + ", ".join(sorted(missing))
        )

    configured_levels = config["corruptions"]
    if not isinstance(configured_levels, Mapping):
        raise ValueError("corruptions must be a mapping.")

    normalized_levels = {
        str(name): tuple(levels)
        for name, levels in configured_levels.items()
    }
    if normalized_levels != CORRUPTION_LEVELS:
        raise ValueError(
            "Configured corruptions must exactly match "
            "src.corruptions.CORRUPTION_LEVELS."
        )

    candidates = config["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list.")

    return config


def build_conditions() -> list[dict[str, object]]:
    """Return clean plus all registered corruption/severity conditions."""

    conditions: list[dict[str, object]] = [
        {
            "corruption": "clean",
            "severity": 0,
            "parameter": None,
        }
    ]

    for name in CORRUPTION_NAMES:
        for severity, parameter in enumerate(
            CORRUPTION_LEVELS[name],
            start=1,
        ):
            conditions.append({
                "corruption": name,
                "severity": severity,
                "parameter": parameter,
            })

    return conditions


def materialize_dataset(
    dataset: CorruptedSubset,
    batch_size: int,
) -> tuple[Tensor, Tensor, str]:
    """Materialize one condition once so every model sees identical tensors."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    image_batches: list[Tensor] = []
    label_batches: list[Tensor] = []

    for images, labels in loader:
        contiguous_images = images.contiguous()
        contiguous_labels = labels.contiguous()
        image_batches.append(contiguous_images)
        label_batches.append(contiguous_labels)

    all_images = torch.cat(image_batches, dim=0).contiguous()
    all_labels = torch.cat(label_batches, dim=0).contiguous()
    digest = hashlib.sha256()
    digest.update(all_images.numpy().tobytes())
    digest.update(all_labels.numpy().tobytes())

    return (
        all_images,
        all_labels,
        digest.hexdigest(),
    )


def evaluate_tensors(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate one frozen model on one already-materialized condition."""

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()

    with torch.inference_mode():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            logits = model(batch_images)
            total_loss += float(criterion(logits, batch_labels).item())
            total_correct += int(
                (logits.argmax(dim=1) == batch_labels).sum().item()
            )
            total_samples += int(batch_labels.numel())

    return {
        "sample_count": total_samples,
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def summarize_results(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarize clean, macro, per-corruption and worst-case accuracy."""

    groups = sorted({
        (str(row["split"]), str(row["model"]))
        for row in rows
    })
    summaries: list[dict[str, object]] = []

    for split, model in groups:
        group = [
            row
            for row in rows
            if row["split"] == split and row["model"] == model
        ]
        clean = next(
            row
            for row in group
            if row["corruption"] == "clean"
        )
        corrupted = [
            row
            for row in group
            if row["corruption"] != "clean"
        ]
        worst = min(corrupted, key=lambda row: float(row["accuracy"]))
        per_corruption = {}

        for corruption in CORRUPTION_NAMES:
            subset = [
                row
                for row in corrupted
                if row["corruption"] == corruption
            ]
            per_corruption[corruption] = {
                "mean_accuracy": sum(
                    float(row["accuracy"])
                    for row in subset
                ) / len(subset),
                "mean_drop": sum(
                    float(row["clean_accuracy_drop"])
                    for row in subset
                ) / len(subset),
            }

        summaries.append({
            "split": split,
            "model": model,
            "clean_accuracy": float(clean["accuracy"]),
            "macro_corrupted_accuracy": sum(
                float(row["accuracy"])
                for row in corrupted
            ) / len(corrupted),
            "macro_clean_accuracy_drop": sum(
                float(row["clean_accuracy_drop"])
                for row in corrupted
            ) / len(corrupted),
            "worst_accuracy": float(worst["accuracy"]),
            "worst_corruption": str(worst["corruption"]),
            "worst_severity": int(worst["severity"]),
            "per_corruption": per_corruption,
        })

    return summaries


def _load_models(
    candidates: Sequence[Mapping[str, object]],
    device: torch.device,
) -> dict[str, nn.Module]:
    models = {}

    for candidate in candidates:
        checkpoint_path = resolve_project_path(candidate["checkpoint"])
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model = build_model(checkpoint["config"]["model"], device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[str(candidate["name"])] = model

    return models


def _raw_sources(
    data_dir: Path,
    download: bool,
) -> tuple[CIFAR10, CIFAR10, list[str]]:
    train_source = CIFAR10(
        root=data_dir,
        train=True,
        transform=None,
        download=download,
    )
    test_source = CIFAR10(
        root=data_dir,
        train=False,
        transform=None,
        download=download,
    )
    return train_source, test_source, list(train_source.classes)


def create_corruption_grid(
    config: Mapping[str, object],
) -> Path:
    """Render one validation image across every corruption and severity."""

    data_config = config["data"]
    train_source, _, class_names = _raw_sources(
        data_dir=resolve_project_path(data_config["data_dir"]),
        download=bool(data_config["download"]),
    )
    _, val_indices = split_train_val_indices(
        dataset_size=len(train_source),
        val_size=CIFAR10_VAL_SIZE,
        seed=int(data_config["split_seed"]),
    )
    visualization = config["visualization"]
    val_position = int(visualization["validation_position"])
    source_index = val_indices[val_position]
    raw_image, label = train_source[source_index]
    image = vision_functional.to_tensor(raw_image)
    figure, axes = plt.subplots(
        len(CORRUPTION_NAMES),
        4,
        figsize=(10, 14),
        constrained_layout=True,
    )

    for row_index, corruption in enumerate(CORRUPTION_NAMES):
        axes[row_index, 0].imshow(image.permute(1, 2, 0))
        axes[row_index, 0].set_title(
            f"clean\n{class_names[int(label)]}"
        )

        for severity in (1, 2, 3):
            corrupted = apply_corruption(
                image,
                corruption,
                severity,
                sample_index=source_index,
                corruption_seed=int(config["corruption_seed"]),
            )
            axes[row_index, severity].imshow(
                corrupted.permute(1, 2, 0)
            )
            parameter = CORRUPTION_LEVELS[corruption][severity - 1]
            axes[row_index, severity].set_title(
                f"{corruption}\nS{severity}: {parameter}"
            )

        for axis in axes[row_index]:
            axis.axis("off")

    output_path = resolve_project_path(visualization["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(visualization["dpi"]))
    plt.close(figure)
    print(output_path.relative_to(PROJECT_ROOT).as_posix())
    return output_path


def _metadata(device: torch.device) -> dict[str, object]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "platform": platform.platform(),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device),
        **collect_git_metadata(PROJECT_ROOT),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_evaluation(config: Mapping[str, object]) -> dict[str, object]:
    """Evaluate all candidates on validation and frozen test corruptions."""

    device = resolve_device(str(config["device"]))
    if device.type != "cuda":
        raise RuntimeError("Formal Stage 15 evaluation requires CUDA.")

    run_metadata = _metadata(device)
    if run_metadata["git_is_dirty"] is not False:
        raise RuntimeError(
            "Formal robustness evaluation requires a clean Git worktree."
        )

    data_config = config["data"]
    train_source, test_source, _ = _raw_sources(
        data_dir=resolve_project_path(data_config["data_dir"]),
        download=bool(data_config["download"]),
    )
    _, val_indices = split_train_val_indices(
        dataset_size=len(train_source),
        val_size=CIFAR10_VAL_SIZE,
        seed=int(data_config["split_seed"]),
    )
    test_indices = list(range(len(test_source)))
    models = _load_models(config["candidates"], device)
    evaluation_config = config["evaluation"]
    materialize_batch_size = int(
        evaluation_config["materialize_batch_size"]
    )
    model_batch_size = int(evaluation_config["model_batch_size"])
    rows: list[dict[str, object]] = []

    for split, source, indices in (
        ("validation", train_source, val_indices),
        ("test", test_source, test_indices),
    ):
        split_rows: list[dict[str, object]] = []

        for condition in build_conditions():
            corruption = str(condition["corruption"])
            severity = int(condition["severity"])
            dataset = CorruptedSubset(
                source=source,
                indices=indices,
                corruption_name=(
                    None if corruption == "clean" else corruption
                ),
                severity=None if corruption == "clean" else severity,
                corruption_seed=int(config["corruption_seed"]),
            )
            images, labels, input_sha256 = materialize_dataset(
                dataset,
                batch_size=materialize_batch_size,
            )

            for model_name, model in models.items():
                metrics = evaluate_tensors(
                    model=model,
                    images=images,
                    labels=labels,
                    batch_size=model_batch_size,
                    device=device,
                )
                row = {
                    "split": split,
                    "model": model_name,
                    "corruption": corruption,
                    "severity": severity,
                    "parameter": condition["parameter"],
                    "sample_count": metrics["sample_count"],
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                    "clean_accuracy_drop": 0.0,
                    "input_sha256": input_sha256,
                }
                split_rows.append(row)
                print(
                    f"{split} {model_name} {corruption} S{severity}: "
                    f"accuracy={100 * float(metrics['accuracy']):.2f}%"
                )

        clean_by_model = {
            str(row["model"]): float(row["accuracy"])
            for row in split_rows
            if row["corruption"] == "clean"
        }
        for row in split_rows:
            row["clean_accuracy_drop"] = (
                clean_by_model[str(row["model"])]
                - float(row["accuracy"])
            )
        rows.extend(split_rows)

    summaries = summarize_results(rows)
    output = config["output"]
    conditions_csv = resolve_project_path(output["conditions_csv"])
    summary_csv = resolve_project_path(output["summary_csv"])
    raw_json = resolve_project_path(output["raw_json"])
    _write_csv(conditions_csv, rows)

    flat_summaries = []
    for summary in summaries:
        flat = {
            key: value
            for key, value in summary.items()
            if key != "per_corruption"
        }
        for corruption, metrics in summary["per_corruption"].items():
            flat[f"{corruption}_mean_accuracy"] = metrics["mean_accuracy"]
            flat[f"{corruption}_mean_drop"] = metrics["mean_drop"]
        flat_summaries.append(flat)
    _write_csv(summary_csv, flat_summaries)

    payload = {
        "protocol": config,
        "metadata": run_metadata,
        "conditions": rows,
        "summaries": summaries,
    }
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    raw_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(conditions_csv.relative_to(PROJECT_ROOT).as_posix())
    print(summary_csv.relative_to(PROJECT_ROOT).as_posix())
    print(raw_json.relative_to(PROJECT_ROOT).as_posix())
    return payload


def main(argv: list[str] | None = None) -> dict[str, object] | Path:
    args = parse_args(argv)
    config_path = (
        args.config
        if args.config.is_absolute()
        else PROJECT_ROOT / args.config
    )
    config = load_robustness_config(config_path)

    if args.mode == "visualize":
        return create_corruption_grid(config)

    if args.mode == "evaluate":
        return run_evaluation(config)

    create_corruption_grid(config)
    return run_evaluation(config)


if __name__ == "__main__":
    main()
