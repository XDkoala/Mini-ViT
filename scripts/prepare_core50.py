"""Create and verify the frozen Stage 16 CORe50 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 直接运行 `python scripts/prepare_core50.py` 时，sys.path 默认只有 scripts/
# 附近；显式加入仓库根目录，保证 src 包和模块运行方式都可用。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core50 import (
    CORE50_FRAME_STRIDE,
    CORE50_SESSION_SPLITS,
    discover_core50_records,
    sha256_file,
    summarize_core50_records,
    verify_selected_images,
    write_manifest,
)


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "core50" / "raw" / "core50_128x128"
)
DEFAULT_ARCHIVE = (
    PROJECT_ROOT / "data" / "core50" / "downloads" / "core50_128x128.zip"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "data" / "core50" / "manifests" / "stage16_core50.csv"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "assets" / "advanced" / "stage16" / "dataset_summary.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse paths and the optional decode-check switch."""

    parser = argparse.ArgumentParser(
        description="Prepare the frozen Stage 16 CORe50 manifest."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Skip full decoding; forbidden for the frozen formal manifest.",
    )
    return parser.parse_args(argv)


def prepare_core50(
    dataset_root: Path,
    archive: Path,
    manifest: Path,
    summary_path: Path,
    *,
    verify_images: bool = True,
) -> dict[str, object]:
    """Build the manifest and write a provenance summary."""

    records, raw_image_count = discover_core50_records(dataset_root)
    if verify_images:
        verify_selected_images(dataset_root, records)

    write_manifest(manifest, records)
    summary = {
        "dataset": "CORe50",
        "task": "10-class category-level RGB classification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "homepage": "https://vlomonaco.github.io/core50/",
            "archive_filename": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "raw_png_count": raw_image_count,
        },
        "protocol": {
            "frame_stride": CORE50_FRAME_STRIDE,
            "frame_rule": "frame_id % 10 == 0",
            "session_splits": {
                split: list(sessions)
                for split, sessions in CORE50_SESSION_SPLITS.items()
            },
            "input_source_size": [128, 128],
            "model_input_size": [32, 32],
            "selected_images_fully_decoded": verify_images,
        },
        "manifest": {
            "filename": manifest.name,
            "sha256": sha256_file(manifest),
        },
        **summarize_core50_records(records),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    """Prepare CORe50 and print the exact frozen counts."""

    args = parse_args()
    summary = prepare_core50(
        dataset_root=args.dataset_root,
        archive=args.archive,
        manifest=args.manifest,
        summary_path=args.summary,
        verify_images=not args.skip_image_verification,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
