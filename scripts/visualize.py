"""Generate Stage 8 figures from frozen training and evaluation artifacts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from scripts.evaluate import load_evaluation_checkpoint
from scripts.train import build_model, resolve_project_path
from src.data import build_datasets
from src.utils import resolve_device, set_seed
from src.visualization import (
    attention_to_heatmaps,
    build_confusion_matrix,
    plot_attention_overlays,
    plot_confusion_matrix,
    plot_error_cases,
    plot_training_history,
    select_error_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_EXPERIMENTS = {
    "minivit_baseline": PROJECT_ROOT / "outputs" / "minivit_baseline",
    "cnn_baseline": PROJECT_ROOT / "outputs" / "cnn_baseline",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Stage 8 visualization command line."""
    parser = argparse.ArgumentParser(
        description="Generate final MiniViT and CNN analysis figures."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "assets",
        help="Root directory for generated figure folders.",
    )
    parser.add_argument(
        "--error-cases",
        type=int,
        default=8,
        help="Number of high-confidence, diverse errors per baseline.",
    )
    parser.add_argument(
        "--attention-cases",
        type=int,
        default=6,
        help="Number of MiniViT attention examples (half correct, half wrong).",
    )
    return parser.parse_args(argv)


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV artifact into dictionaries."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_metrics(path: str | Path) -> dict[str, object]:
    """Read and validate final evaluation metadata."""
    with Path(path).open(encoding="utf-8") as file:
        metrics = json.load(file)

    if metrics.get("is_partial_evaluation"):
        raise ValueError("Visualizations require complete evaluation metrics.")
    if int(metrics.get("num_samples", 0)) != 10_000:
        raise ValueError("Visualizations require all 10,000 CIFAR-10 samples.")
    return metrics


def parse_prediction_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Convert CSV strings back to typed prediction records."""
    records: list[dict[str, object]] = []
    for row in rows:
        records.append({
            "sample_index": int(row["sample_index"]),
            "true_label": int(row["true_label"]),
            "true_class": row["true_class"],
            "predicted_label": int(row["predicted_label"]),
            "predicted_class": row["predicted_class"],
            "confidence": float(row["confidence"]),
            "correct": row["correct"].strip().lower() == "true",
        })
    return records


def save_figure(figure, path: str | Path) -> None:
    """Save one figure at report quality and release Matplotlib memory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def collect_images(dataset, records: list[dict[str, object]]) -> Tensor:
    """Fetch normalized test images by their stable prediction indices."""
    images = [dataset[int(record["sample_index"])][0] for record in records]
    return torch.stack(images)


def select_attention_cases(
    records: list[dict[str, object]],
    max_cases: int,
) -> list[dict[str, object]]:
    """Choose a deterministic mix of confident correct and wrong samples."""
    if max_cases <= 1:
        raise ValueError("attention cases must be greater than one.")

    wrong_count = max_cases // 2
    correct_count = max_cases - wrong_count
    wrong = select_error_cases(records, wrong_count)

    # 正确样本同样按置信度排序，并优先覆盖不同类别。
    correct_candidates = sorted(
        (record for record in records if bool(record["correct"])),
        key=lambda row: float(row["confidence"]),
        reverse=True,
    )
    correct: list[dict[str, object]] = []
    seen_classes: set[int] = set()
    for record in correct_candidates:
        label = int(record["true_label"])
        if label in seen_classes:
            continue
        correct.append(record)
        seen_classes.add(label)
        if len(correct) == correct_count:
            break

    return [*correct, *wrong]


def main(argv: list[str] | None = None) -> list[Path]:
    """Generate baseline curves, confusion, errors, and ViT attention."""
    args = parse_args(argv)
    if args.error_cases <= 0:
        raise ValueError("--error-cases must be greater than zero.")
    if args.attention_cases <= 1:
        raise ValueError("--attention-cases must be greater than one.")

    output_root = resolve_project_path(args.output_root)
    generated: list[Path] = []
    baseline_records: dict[str, list[dict[str, object]]] = {}
    minivit_checkpoint: dict[str, object] | None = None

    # 先使用训练日志和一次性测试预测生成两种无需重新推理的图。
    for experiment_name, experiment_dir in BASELINE_EXPERIMENTS.items():
        checkpoint_path = experiment_dir / "best.pt"
        checkpoint = load_evaluation_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
        )
        metrics = load_metrics(experiment_dir / "evaluation" / "metrics.json")
        history = load_csv_rows(experiment_dir / "history.csv")
        records = parse_prediction_rows(
            load_csv_rows(experiment_dir / "evaluation" / "predictions.csv")
        )
        if len(records) != int(metrics["num_samples"]):
            raise ValueError(f"Prediction count mismatch for {experiment_name}.")

        baseline_records[experiment_name] = records
        if experiment_name == "minivit_baseline":
            minivit_checkpoint = checkpoint

        curve_path = output_root / "training_curves" / f"{experiment_name}.png"
        curve = plot_training_history(
            history=history,
            best_epoch=int(checkpoint["epoch"]),
            title=f"{experiment_name}: training history",
        )
        save_figure(curve, curve_path)
        generated.append(curve_path)

        class_names = [""] * int(metrics["num_classes"])
        for record in records:
            class_names[int(record["true_label"])] = str(record["true_class"])

        matrix = build_confusion_matrix(records, len(class_names))
        matrix_path = (
            output_root / "confusion_matrices" / f"{experiment_name}.png"
        )
        matrix_figure = plot_confusion_matrix(
            matrix,
            class_names,
            title=f"{experiment_name}: test confusion matrix",
        )
        save_figure(matrix_figure, matrix_path)
        generated.append(matrix_path)

    if minivit_checkpoint is None:
        raise RuntimeError("MiniViT baseline checkpoint was not loaded.")

    config = minivit_checkpoint["config"]
    seed = int(config["seed"])
    set_seed(seed)
    data_config = config["data"]
    _, _, test_dataset, _ = build_datasets(
        data_dir=resolve_project_path(data_config["data_dir"]),
        seed=seed,
        augmentation=str(data_config.get("augmentation", "basic")),
    )

    # 错误样本图严格按 predictions.csv 的 sample_index 回取原测试图片。
    for experiment_name, records in baseline_records.items():
        selected = [dict(row) for row in select_error_cases(records, args.error_cases)]
        images = collect_images(test_dataset, selected)
        error_path = output_root / "error_cases" / f"{experiment_name}.png"
        error_figure = plot_error_cases(
            images,
            selected,
            title=f"{experiment_name}: diverse high-confidence errors",
        )
        save_figure(error_figure, error_path)
        generated.append(error_path)

    # Attention 需要一次额外前向传播，但不重新计算或修改测试指标。
    minivit_records = baseline_records["minivit_baseline"]
    attention_records = select_attention_cases(
        minivit_records,
        args.attention_cases,
    )
    attention_images = collect_images(test_dataset, attention_records)
    device = resolve_device(str(config["device"]))
    model = build_model(config["model"], device)
    model.load_state_dict(minivit_checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        _, attention = model(
            attention_images.to(device),
            return_attention=True,
        )

    heatmaps = attention_to_heatmaps(
        attention,
        image_size=int(config["model"]["image_size"]),
    )
    attention_path = output_root / "attention" / "minivit_attention_examples.png"
    attention_figure = plot_attention_overlays(
        attention_images,
        heatmaps,
        attention_records,
        title="MiniViT final-block attention (descriptive, not causal)",
    )
    save_figure(attention_figure, attention_path)
    generated.append(attention_path)

    for path in generated:
        print(path.relative_to(PROJECT_ROOT).as_posix())
    return generated


if __name__ == "__main__":
    main()
