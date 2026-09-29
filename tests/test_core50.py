"""Tests for the frozen Stage 16 CORe50 data protocol."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from src.core50 import (
    CORE50_CATEGORY_NAMES,
    Core50Record,
    Core50Dataset,
    build_core50_dataloaders,
    build_core50_transforms,
    category_label_from_object_id,
    discover_core50_records,
    load_manifest,
    split_from_session,
    validate_core50_records,
    verify_selected_images,
    write_manifest,
)


def _record(
    *,
    session: int,
    object_id: int,
    frame_id: int = 0,
) -> Core50Record:
    """Build one internally consistent test record."""

    label = category_label_from_object_id(object_id)
    return Core50Record(
        relative_path=(
            f"s{session}/o{object_id}/"
            f"C_{session:02d}_{object_id:02d}_{frame_id:03d}.png"
        ),
        split=split_from_session(session),
        session=session,
        object_id=object_id,
        frame_id=frame_id,
        category_label=label,
        category_name=CORE50_CATEGORY_NAMES[label],
    )


@pytest.mark.parametrize(
    ("object_id", "expected"),
    [(1, 0), (5, 0), (6, 1), (45, 8), (46, 9), (50, 9)],
)
def test_object_ids_map_to_ten_category_blocks(
    object_id: int,
    expected: int,
) -> None:
    assert category_label_from_object_id(object_id) == expected


@pytest.mark.parametrize("object_id", [0, 51])
def test_invalid_object_id_is_rejected(object_id: int) -> None:
    with pytest.raises(ValueError, match="object_id"):
        category_label_from_object_id(object_id)


def test_official_test_sessions_and_frozen_validation_session() -> None:
    assert split_from_session(8) == "validation"
    assert {session for session in range(1, 12) if split_from_session(session) == "test"} == {3, 7, 10}


def test_validation_rejects_duplicate_identity() -> None:
    records = [_record(session=1, object_id=1)] * 2

    with pytest.raises(ValueError, match="Duplicate"):
        validate_core50_records(records)


def test_validation_requires_stride_rule() -> None:
    records = [
        _record(session=session, object_id=object_id)
        for session in range(1, 12)
        for object_id in range(1, 51)
    ]
    records[0] = _record(session=1, object_id=1, frame_id=1)

    with pytest.raises(ValueError, match="stride"):
        validate_core50_records(records)


def test_discovery_parses_paths_and_samples_deterministically(
    tmp_path: Path,
) -> None:
    # 只创建最小空文件即可测试路径协议；图像解码由独立测试负责。
    for session in range(1, 12):
        for object_id in range(1, 51):
            object_dir = tmp_path / f"s{session}" / f"o{object_id}"
            object_dir.mkdir(parents=True)
            for frame_id in (0, 1, 10):
                (object_dir / (
                    f"C_{session:02d}_{object_id:02d}_{frame_id:03d}.png"
                )).touch()

    records, raw_count = discover_core50_records(
        tmp_path,
        expected_image_count=None,
    )

    assert raw_count == 11 * 50 * 3
    assert len(records) == 11 * 50 * 2
    assert {record.frame_id for record in records} == {0, 10}
    assert records[0] == _record(session=1, object_id=1, frame_id=0)


def test_discovery_rejects_filename_directory_disagreement(
    tmp_path: Path,
) -> None:
    for session in range(1, 12):
        for object_id in range(1, 51):
            object_dir = tmp_path / f"s{session}" / f"o{object_id}"
            object_dir.mkdir(parents=True)
            filename_object = 2 if (session, object_id) == (1, 1) else object_id
            (object_dir / (
                f"C_{session:02d}_{filename_object:02d}_000.png"
            )).touch()

    with pytest.raises(ValueError, match="disagrees"):
        discover_core50_records(tmp_path, expected_image_count=None)


def test_image_verification_checks_decode_and_size(tmp_path: Path) -> None:
    record = _record(session=1, object_id=1)
    path = tmp_path / record.relative_path
    path.parent.mkdir(parents=True)
    Image.new("RGB", (128, 128), color=(10, 20, 30)).save(path)

    verify_selected_images(tmp_path, [record])

    Image.new("RGB", (64, 64)).save(path)
    with pytest.raises(ValueError, match="decode"):
        verify_selected_images(tmp_path, [record])


def test_manifest_schema_and_order_are_stable(tmp_path: Path) -> None:
    records = [
        _record(session=session, object_id=object_id)
        for session in range(1, 12)
        for object_id in range(1, 51)
    ]
    path = tmp_path / "manifest.csv"

    write_manifest(path, records)

    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == [
        "relative_path",
        "split",
        "session",
        "object_id",
        "frame_id",
        "category_label",
        "category_name",
    ]
    assert len(rows) == 550

    loaded = load_manifest(path)
    assert loaded == records


def test_core50_dataset_returns_normalized_32px_tensor(
    tmp_path: Path,
) -> None:
    record = _record(session=1, object_id=1)
    image_path = tmp_path / record.relative_path
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (128, 128), color=(128, 64, 32)).save(image_path)
    _, eval_transform = build_core50_transforms("none")
    dataset = Core50Dataset(tmp_path, [record], eval_transform)

    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label == 0
    expected_red = ((128 / 255.0) - 0.4914) / 0.2470
    assert image[0].mean().item() == pytest.approx(expected_red, abs=1e-5)


def test_eval_transform_is_deterministic(tmp_path: Path) -> None:
    record = _record(session=1, object_id=1)
    image_path = tmp_path / record.relative_path
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (128, 128), color=(20, 40, 80)).save(image_path)
    _, eval_transform = build_core50_transforms("basic")
    dataset = Core50Dataset(tmp_path, [record], eval_transform)

    image_a, _ = dataset[0]
    image_b, _ = dataset[0]

    assert torch.equal(image_a, image_b)


def test_core50_dataloaders_use_frozen_splits_and_sampler_types(
    tmp_path: Path,
) -> None:
    records = [
        _record(session=session, object_id=object_id)
        for session in range(1, 12)
        for object_id in range(1, 51)
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, records)

    train_loader, validation_loader, test_loader, classes = (
        build_core50_dataloaders(
            tmp_path,
            manifest,
            batch_size=32,
            num_workers=0,
            train_seed=42,
        )
    )

    assert len(train_loader.dataset) == 7 * 50
    assert len(validation_loader.dataset) == 1 * 50
    assert len(test_loader.dataset) == 3 * 50
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(validation_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    assert classes == list(CORE50_CATEGORY_NAMES)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 0, "batch_size"),
        ("num_workers", -1, "num_workers"),
        ("train_seed", -1, "train_seed"),
    ],
)
def test_core50_dataloader_rejects_invalid_runtime_values(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    kwargs = {"batch_size": 4, "num_workers": 0, "train_seed": 42}
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        build_core50_dataloaders(tmp_path, tmp_path / "missing.csv", **kwargs)
