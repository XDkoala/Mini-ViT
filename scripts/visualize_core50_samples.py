"""Create the Stage 16 CORe50 original-versus-32px QA grid."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core50 import (  # noqa: E402
    CORE50_CATEGORY_NAMES,
    Core50Record,
    load_manifest,
)


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "core50" / "raw" / "core50_128x128"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "core50" / "manifests" / "stage16_core50.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "assets" / "advanced" / "stage16" / "core50_samples.png"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse data, manifest, and output paths."""

    parser = argparse.ArgumentParser(
        description="Visualize Stage 16 CORe50 samples before training."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def select_train_examples(
    records: list[Core50Record],
) -> list[Core50Record]:
    """Select one deterministic middle train record from every category."""

    grouped: dict[int, list[Core50Record]] = defaultdict(list)
    for record in records:
        if record.split == "train":
            grouped[record.category_label].append(record)

    if set(grouped) != set(range(10)):
        raise ValueError("Train records must cover all 10 CORe50 categories.")

    # 取排序后中间项，避免人为挑选“最好看”的样本。
    return [grouped[label][len(grouped[label]) // 2] for label in range(10)]


def create_sample_grid(
    dataset_root: Path,
    records: list[Core50Record],
    output_path: Path,
) -> None:
    """Save five rows of paired 128px and 32px model-input previews."""

    selected = select_train_examples(records)
    figure, axes = plt.subplots(5, 4, figsize=(12, 15))

    for label, record in enumerate(selected):
        row = label // 2
        pair_start = (label % 2) * 2
        with Image.open(Path(dataset_root) / record.relative_path) as source:
            original = source.convert("RGB")
            model_input = resize(
                original,
                [32, 32],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            # 用最近邻放回 128，仅为让 32×32 像素在图中清楚可见。
            pixel_preview = model_input.resize((128, 128), Image.Resampling.NEAREST)

        original_axis = axes[row, pair_start]
        preview_axis = axes[row, pair_start + 1]
        original_axis.imshow(original)
        preview_axis.imshow(pixel_preview)
        title = CORE50_CATEGORY_NAMES[label].replace("_", " ")
        original_axis.set_title(
            f"{label}: {title}\n128×128 source",
            fontsize=11,
        )
        preview_axis.set_title(
            "32×32 model input\n(nearest-neighbor preview)",
            fontsize=10,
        )
        original_axis.set_axis_off()
        preview_axis.set_axis_off()

    figure.suptitle(
        "CORe50 Stage 16 — deterministic train samples",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load the frozen manifest and create the visual QA artifact."""

    args = parse_args()
    records = load_manifest(args.manifest)
    create_sample_grid(args.dataset_root, records, args.output)
    print(f"Saved CORe50 sample grid: {args.output}")


if __name__ == "__main__":
    main()
