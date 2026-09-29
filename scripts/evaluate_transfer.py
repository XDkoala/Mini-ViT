"""Evaluate frozen Stage 16 CORe50 checkpoints, one split at a time."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core50 import (  # noqa: E402
    CORE50_CATEGORY_NAMES,
    build_core50_datasets,
    sha256_file,
)
from src.engine import evaluate_with_predictions  # noqa: E402
from src.transfer import TransferMiniViT  # noqa: E402
from src.utils import resolve_device  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/advanced/stage16/evaluation.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def resolve_project_path(value: str) -> Path:
    """Interpret frozen relative paths from the repository root."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def classification_report(
    predictions: list[dict[str, int | float | bool]],
    class_names: tuple[str, ...] = CORE50_CATEGORY_NAMES,
) -> dict[str, object]:
    """Compute per-class one-vs-rest metrics and unweighted macro F1."""

    if not predictions:
        raise ValueError("Cannot report metrics for an empty split.")
    size = len(class_names)
    confusion = [[0 for _ in range(size)] for _ in range(size)]
    for row in predictions:
        true = int(row["true_label"])
        predicted = int(row["predicted_label"])
        if not (0 <= true < size and 0 <= predicted < size):
            raise ValueError("Prediction contains an out-of-range class label.")
        confusion[true][predicted] += 1

    per_class = []
    for index, name in enumerate(class_names):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        denominator = support + predicted_count
        f1 = 2 * tp / denominator if denominator else 0.0
        per_class.append({
            "label": index,
            "name": name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return {
        "macro_f1": sum(row["f1"] for row in per_class) / size,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def evaluate_model(
    *,
    mode: str,
    specification: dict[str, object],
    dataset: object,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    split: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Check a pinned best checkpoint and evaluate every sample once."""

    checkpoint_path = resolve_project_path(str(specification["checkpoint"]))
    if checkpoint_path.name != "best.pt":
        raise ValueError(f"{mode}: evaluation requires best.pt.")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != specification["sha256"]:
        raise ValueError(f"{mode}: frozen checkpoint SHA-256 mismatch.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if int(checkpoint["epoch"]) != int(specification["best_epoch"]):
        raise ValueError(f"{mode}: frozen best epoch mismatch.")
    config = checkpoint["config"]
    if config["transfer"]["mode"] != mode:
        raise ValueError(f"{mode}: checkpoint transfer mode mismatch.")

    model_config = dict(config["model"])
    model_config.pop("name")
    model = TransferMiniViT(transfer_mode=mode, **model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    aggregate, predictions = evaluate_with_predictions(
        model, loader, nn.CrossEntropyLoss(), device
    )
    if len(predictions) != len(dataset):
        raise RuntimeError(f"{mode}: incomplete {split} evaluation.")
    for row, record in zip(predictions, dataset.records, strict=True):
        if int(row["true_label"]) != record.category_label:
            raise RuntimeError(f"{mode}: manifest/prediction order mismatch.")
        row["relative_path"] = record.relative_path

    if split == "validation":
        history_row = checkpoint["history"][int(checkpoint["epoch"]) - 1]
        if abs(float(aggregate["accuracy"]) - float(history_row["val_accuracy"])) > 1e-8:
            raise RuntimeError(f"{mode}: validation accuracy differs from training history.")

    report = classification_report(predictions)
    errors = sorted(
        (row for row in predictions if not row["correct"]),
        key=lambda row: (-float(row["confidence"]), int(row["sample_index"])),
    )
    report.update({
        "mode": mode,
        "split": split,
        "checkpoint": str(specification["checkpoint"]),
        "checkpoint_sha256": checkpoint_hash,
        "best_epoch": int(checkpoint["epoch"]),
        "loss": aggregate["loss"],
        "accuracy": aggregate["accuracy"],
        "num_samples": aggregate["num_samples"],
        "top_confident_errors": errors[:5],
    })
    return report, predictions


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Evaluation output directory must be empty.")
    with args.protocol.open(encoding="utf-8") as file:
        protocol = yaml.safe_load(file)
    if protocol["protocol"] != "stage16_core50_v1":
        raise ValueError("Unknown Stage 16 evaluation protocol.")
    manifest = resolve_project_path(protocol["manifest"])
    if sha256_file(manifest) != protocol["manifest_sha256"]:
        raise ValueError("Frozen CORe50 manifest SHA-256 mismatch.")
    dataset_root = resolve_project_path(protocol["dataset_root"])
    _, validation, test, class_names = build_core50_datasets(
        dataset_root, manifest, augmentation="basic"
    )
    dataset = validation if args.split == "validation" else test
    device = resolve_device("auto")
    reports = {}
    prediction_tables = {}
    for mode in ("scratch", "linear_probe", "fine_tune"):
        report, predictions = evaluate_model(
            mode=mode,
            specification=protocol["models"][mode],
            dataset=dataset,
            batch_size=int(protocol["batch_size"]),
            num_workers=int(protocol["num_workers"]),
            device=device,
            split=args.split,
        )
        reports[mode] = report
        prediction_tables[mode] = predictions
        print(
            f"{mode}: accuracy={100 * report['accuracy']:.2f}% "
            f"macro_f1={100 * report['macro_f1']:.2f}%"
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "protocol": protocol["protocol"],
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": protocol["manifest_sha256"],
        "split": args.split,
        "class_names": class_names,
        "models": reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for mode, predictions in prediction_tables.items():
        with (output_dir / f"{mode}_predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
    return result


if __name__ == "__main__":
    main()
