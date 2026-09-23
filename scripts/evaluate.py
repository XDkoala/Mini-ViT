"""Evaluate a frozen best checkpoint on the CIFAR-10 test split."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from scripts.train import build_model, resolve_project_path
from src.data import build_dataloaders
from src.engine import evaluate_with_predictions
from src.utils import resolve_device, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_FIELDS = (
    "sample_index",
    "true_label",
    "true_class",
    "predicted_label",
    "predicted_class",
    "confidence",
    "correct",
)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for frozen-checkpoint evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate a best checkpoint on CIFAR-10 test data."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a frozen best.pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override the evaluation output directory. By default, "
            "files are written beside best.pt under evaluation/."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit test batches for an isolated evaluation smoke test.",
    )
    return parser.parse_args(argv)


def compute_sha256(path: str | Path) -> str:
    """Compute a file SHA-256 digest without loading it all into memory."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest().upper()


def load_evaluation_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict[str, object]:
    """Load metadata and model weights from a frozen best checkpoint."""

    checkpoint_path = Path(path)

    # 阶段 8 的模型选择已经由验证集完成；拒绝 last.pt 可防止误用最终轮。
    if checkpoint_path.name != "best.pt":
        raise ValueError(
            "Final evaluation requires a checkpoint named best.pt."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    required_keys = {
        "epoch",
        "model_state",
        "best_val_accuracy",
        "history",
        "config",
    }
    missing_keys = required_keys - checkpoint.keys()

    if missing_keys:
        missing_names = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Checkpoint is missing required keys: {missing_names}"
        )

    if not isinstance(checkpoint["config"], dict):
        raise ValueError("Checkpoint config must be a mapping.")

    return checkpoint


def add_class_names(
    records: list[dict[str, int | float | bool]],
    class_names: list[str],
) -> list[dict[str, int | float | bool | str]]:
    """Add human-readable CIFAR-10 class names to numeric predictions."""

    named_records: list[dict[str, int | float | bool | str]] = []

    for record in records:
        true_label = int(record["true_label"])
        predicted_label = int(record["predicted_label"])

        if not 0 <= true_label < len(class_names):
            raise ValueError(
                f"true label {true_label} is outside class_names."
            )

        if not 0 <= predicted_label < len(class_names):
            raise ValueError(
                f"predicted label {predicted_label} is outside class_names."
            )

        named_records.append({
            "sample_index": int(record["sample_index"]),
            "true_label": true_label,
            "true_class": class_names[true_label],
            "predicted_label": predicted_label,
            "predicted_class": class_names[predicted_label],
            "confidence": float(record["confidence"]),
            "correct": bool(record["correct"]),
        })

    return named_records


def save_metrics_json(
    metrics: dict[str, object],
    path: str | Path,
) -> None:
    """Write final aggregate metrics as readable UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")


def save_predictions_csv(
    records: list[dict[str, int | float | bool | str]],
    path: str | Path,
) -> None:
    """Write one stable, human-readable prediction row per test sample."""

    if not records:
        raise ValueError("prediction records must not be empty.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=PREDICTION_FIELDS,
        )
        writer.writeheader()
        writer.writerows(records)


def display_path(path: Path) -> str:
    """Prefer a project-relative path in persisted evaluation metadata."""

    resolved_path = path.resolve()

    try:
        return resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved_path)


def main(
    argv: list[str] | None = None,
) -> dict[str, object]:
    """Evaluate one frozen checkpoint and persist aggregate and sample results."""

    args = parse_args(argv)

    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be greater than zero.")

    if args.max_batches is not None and args.output_dir is None:
        raise ValueError(
            "--max-batches requires an explicit --output-dir "
            "to protect final evaluation results."
        )

    checkpoint_path = resolve_project_path(args.checkpoint)

    # 先在 CPU 上读取内嵌配置，再根据配置决定最终推理设备。
    checkpoint = load_evaluation_checkpoint(
        path=checkpoint_path,
        device=torch.device("cpu"),
    )
    config = checkpoint["config"]

    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(str(config["device"]))

    data_config = config["data"]
    model_config = config["model"]
    experiment_config = config["experiment"]

    data_dir = resolve_project_path(data_config["data_dir"])
    _, _, test_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        seed=seed,
        augmentation=str(data_config.get("augmentation", "basic")),
    )

    model = build_model(
        model_config=model_config,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state"])

    aggregate_metrics, numeric_records = evaluate_with_predictions(
        model=model,
        loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        max_batches=args.max_batches,
    )
    named_records = add_class_names(numeric_records, class_names)

    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir is not None
        else checkpoint_path.parent / "evaluation"
    )
    checkpoint_digest = compute_sha256(checkpoint_path)

    final_metrics: dict[str, object] = {
        "experiment_name": str(experiment_config["name"]),
        "checkpoint_path": display_path(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": checkpoint_digest,
        "test_loss": float(aggregate_metrics["loss"]),
        "test_accuracy": float(aggregate_metrics["accuracy"]),
        "num_samples": int(aggregate_metrics["num_samples"]),
        "num_classes": len(class_names),
        "is_partial_evaluation": args.max_batches is not None,
        "max_batches": args.max_batches,
    }

    save_metrics_json(
        final_metrics,
        output_dir / "metrics.json",
    )
    save_predictions_csv(
        named_records,
        output_dir / "predictions.csv",
    )

    print(
        f"Evaluated {final_metrics['experiment_name']} on {device}. "
        f"epoch={final_metrics['checkpoint_epoch']}, "
        f"samples={final_metrics['num_samples']}, "
        f"test_loss={final_metrics['test_loss']:.4f}, "
        f"test_accuracy={final_metrics['test_accuracy']:.2%}"
    )

    return final_metrics


if __name__ == "__main__":
    main()
