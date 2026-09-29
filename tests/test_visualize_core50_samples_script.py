"""Tests for the Stage 16 CORe50 sample-grid script."""

from pathlib import Path

from PIL import Image

from scripts.visualize_core50_samples import (
    create_sample_grid,
    parse_args,
    select_train_examples,
)
from src.core50 import CORE50_CATEGORY_NAMES, Core50Record


def _records() -> list[Core50Record]:
    records = []
    for label, name in enumerate(CORE50_CATEGORY_NAMES):
        object_id = label * 5 + 1
        for frame_id in (0, 10, 20):
            records.append(Core50Record(
                relative_path=(
                    f"s1/o{object_id}/"
                    f"C_01_{object_id:02d}_{frame_id:03d}.png"
                ),
                split="train",
                session=1,
                object_id=object_id,
                frame_id=frame_id,
                category_label=label,
                category_name=name,
            ))
    return records


def test_select_train_examples_is_deterministic_and_not_first() -> None:
    selected = select_train_examples(_records())

    assert len(selected) == 10
    assert [record.frame_id for record in selected] == [10] * 10


def test_sample_grid_is_written(tmp_path: Path) -> None:
    records = _records()
    for record in records:
        path = tmp_path / "data" / record.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (128, 128),
            color=(record.category_label * 20, 40, 80),
        ).save(path)
    output = tmp_path / "grid.png"

    create_sample_grid(tmp_path / "data", records, output)

    assert output.is_file()
    with Image.open(output) as image:
        assert image.width > image.height // 2


def test_visualize_core50_cli_defaults() -> None:
    args = parse_args([])

    assert args.manifest.name == "stage16_core50.csv"
    assert args.output.name == "core50_samples.png"
