"""Predict one image using the checkpoint's dataset and class mapping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core50 import sha256_file  # noqa: E402
from src.inference import (  # noqa: E402
    ATTENTION_METHODS, load_inference_runtime, predict_batch, top_k_predictions,
)
from src.visualization import plot_attention_comparison  # noqa: E402


METHOD_LABELS = {
    "last": "Final-block CLS attention\n(mean heads)",
    "rollout": "Attention Rollout\n(mean heads + residual)",
}


def project_path(value: str | Path) -> Path:
    """Resolve CLI paths against the project root, as training does."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--attention", choices=ATTENTION_METHODS, default="none")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-sha256", help="Optional checkpoint hash verification.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Load once, run once, and save a prediction report plus optional PNG."""

    args = parse_args(argv)
    image_path = project_path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    runtime = load_inference_runtime(
        project_path(args.checkpoint), device=args.device,
        expected_sha256=args.expected_sha256,
    )
    # Validate k before inference; a typo should not silently clamp the request.
    # 提前拒绝超出类别数的 k，避免把错误参数默默改成另一个数。
    if not 1 <= args.top_k <= len(runtime.class_names):
        raise ValueError(f"top-k must be between 1 and {len(runtime.class_names)}.")
    with Image.open(image_path) as image:
        original_size = list(image.size)
        normalized, _ = runtime.prepare_image(image)
    batch = predict_batch(runtime, normalized[None], attention=args.attention)
    top_k = top_k_predictions(batch.probabilities[0], runtime.class_names, args.top_k)
    output_dir = (
        project_path(args.output_dir) if args.output_dir is not None
        else PROJECT_ROOT / "outputs/predictions" / image_path.stem
    )
    protected_paths = {image_path, runtime.checkpoint_path}
    destination_paths = {output_dir / "prediction.json"}
    if batch.heatmaps:
        destination_paths.add(output_dir / "attention.png")
    if any(path.resolve() in protected_paths for path in destination_paths):
        raise ValueError("Output would overwrite the input image or checkpoint.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        **runtime.metadata(),
        "image": str(image_path),
        "image_sha256": sha256_file(image_path),
        "source_size_wh": original_size,
        "top_k": top_k,
        "attention_method": args.attention,
        "raw_patch_weights": {
            method: weights[0].tolist() for method, weights in batch.patch_weights.items()
        },
        "interpretation": (
            "Softmax scores are uncalibrated. Maps are per-image relative displays; "
            "attention and rollout do not establish causal contributions."
        ),
        "attention_image": None,
    }
    if batch.heatmaps:
        prediction = top_k[0]
        records = [{"predicted_class": prediction["class"],
                    "confidence": prediction["probability"]}]
        figure = plot_attention_comparison(
            normalized[None],
            {METHOD_LABELS[method]: maps for method, maps in batch.heatmaps.items()},
            records,
            title=f"{runtime.dataset} | checkpoint epoch {runtime.epoch}",
        )
        attention_path = output_dir / "attention.png"
        figure.savefig(attention_path, dpi=160)
        plt.close(figure)
        report["attention_image"] = str(attention_path.resolve())
    report_path = output_dir / "prediction.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset: {runtime.dataset} | Device: {runtime.device} | Epoch: {runtime.epoch}")
    print(f"Checkpoint SHA-256: {runtime.checkpoint_sha256}")
    print("Preprocess: RGB -> 32x32 bilinear/antialias if required -> ToTensor -> Normalize")
    for rank, prediction in enumerate(top_k, start=1):
        print(f"{rank}. {prediction['class']}: {prediction['probability']:.2%}")
    print(f"Report: {report_path.resolve()}")
    if report["attention_image"]:
        print(f"Attention: {report['attention_image']}")
    return report


def cli() -> None:
    """Display actionable file/configuration errors without a traceback."""

    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"predict: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    cli()
