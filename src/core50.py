"""CORe50 discovery, split validation, and manifest utilities."""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data import CIFAR10_MEAN, CIFAR10_STD


CORE50_CATEGORY_NAMES = (
    "plug_adapters",
    "mobile_phones",
    "scissors",
    "light_bulbs",
    "cans",
    "glasses",
    "balls",
    "markers",
    "cups",
    "remote_controls",
)

CORE50_SESSION_SPLITS = {
    "train": (1, 2, 4, 5, 6, 9, 11),
    "validation": (8,),
    "test": (3, 7, 10),
}

CORE50_EXPECTED_IMAGE_COUNT = 164_866
CORE50_FRAME_STRIDE = 10

_IMAGE_NAME_PATTERN = re.compile(
    r"^C_(?P<session>\d{2})_(?P<object>\d{2})_(?P<frame>\d{3})\.png$"
)


@dataclass(frozen=True, slots=True)
class Core50Record:
    """One selected CORe50 image and its immutable metadata."""

    relative_path: str
    split: str
    session: int
    object_id: int
    frame_id: int
    category_label: int
    category_name: str

    def to_row(self) -> dict[str, object]:
        """Return a CSV/JSON-friendly row."""

        return asdict(self)


def category_label_from_object_id(object_id: int) -> int:
    """Map the 50 instance IDs to the 10 official semantic categories."""

    if not 1 <= object_id <= 50:
        raise ValueError("CORe50 object_id must be in [1, 50].")
    return (object_id - 1) // 5


def split_from_session(session: int) -> str:
    """Return the pre-registered split for one complete session."""

    for split, sessions in CORE50_SESSION_SPLITS.items():
        if session in sessions:
            return split
    raise ValueError("CORe50 session must be in [1, 11].")


def _numeric_directory_names(
    root: Path,
    prefix: str,
) -> set[int]:
    """Collect numeric IDs from directories such as s1 or o34."""

    values: set[int] = set()
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            suffix = path.name[len(prefix):]
            if suffix.isdigit():
                values.add(int(suffix))
    return values


def discover_core50_records(
    dataset_root: Path,
    *,
    frame_stride: int = CORE50_FRAME_STRIDE,
    expected_image_count: int | None = CORE50_EXPECTED_IMAGE_COUNT,
) -> tuple[list[Core50Record], int]:
    """Scan the official directory tree and apply deterministic frame sampling."""

    dataset_root = Path(dataset_root)
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive.")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"CORe50 root does not exist: {dataset_root}")

    expected_sessions = set(range(1, 12))
    actual_sessions = _numeric_directory_names(dataset_root, "s")
    if actual_sessions != expected_sessions:
        raise ValueError(
            "CORe50 must contain exactly sessions s1-s11; "
            f"found {sorted(actual_sessions)}."
        )

    records: list[Core50Record] = []
    raw_image_count = 0
    expected_objects = set(range(1, 51))

    for session in sorted(actual_sessions):
        session_dir = dataset_root / f"s{session}"
        actual_objects = _numeric_directory_names(session_dir, "o")
        if actual_objects != expected_objects:
            raise ValueError(
                f"Session s{session} must contain exactly o1-o50; "
                f"found {len(actual_objects)} object directories."
            )

        for object_id in sorted(actual_objects):
            object_dir = session_dir / f"o{object_id}"
            for image_path in object_dir.glob("*.png"):
                raw_image_count += 1
                match = _IMAGE_NAME_PATTERN.fullmatch(image_path.name)
                if match is None:
                    raise ValueError(
                        f"Invalid CORe50 image filename: {image_path.name}"
                    )

                filename_session = int(match.group("session"))
                filename_object = int(match.group("object"))
                frame_id = int(match.group("frame"))
                if (
                    filename_session != session
                    or filename_object != object_id
                ):
                    raise ValueError(
                        "Filename metadata disagrees with its directories: "
                        f"{image_path}"
                    )

                # 固定间隔抽帧；不使用随机数，也不受遍历顺序影响。
                if frame_id % frame_stride != 0:
                    continue

                category_label = category_label_from_object_id(object_id)
                records.append(Core50Record(
                    relative_path=image_path.relative_to(
                        dataset_root
                    ).as_posix(),
                    split=split_from_session(session),
                    session=session,
                    object_id=object_id,
                    frame_id=frame_id,
                    category_label=category_label,
                    category_name=CORE50_CATEGORY_NAMES[category_label],
                ))

    if (
        expected_image_count is not None
        and raw_image_count != expected_image_count
    ):
        raise ValueError(
            "Unexpected CORe50 PNG count: "
            f"expected {expected_image_count}, found {raw_image_count}."
        )

    records.sort(key=lambda item: (
        item.session,
        item.object_id,
        item.frame_id,
    ))
    validate_core50_records(records, frame_stride=frame_stride)
    return records, raw_image_count


def validate_core50_records(
    records: list[Core50Record],
    *,
    frame_stride: int = CORE50_FRAME_STRIDE,
) -> None:
    """Validate identity, sampling, class coverage, and session isolation."""

    if not records:
        raise ValueError("CORe50 manifest cannot be empty.")

    identities: set[tuple[int, int, int]] = set()
    paths: set[str] = set()
    session_splits: dict[int, set[str]] = defaultdict(set)
    split_classes: dict[str, set[int]] = defaultdict(set)

    for record in records:
        identity = (record.session, record.object_id, record.frame_id)
        if identity in identities:
            raise ValueError(f"Duplicate CORe50 identity: {identity}")
        if record.relative_path in paths:
            raise ValueError(
                f"Duplicate CORe50 relative path: {record.relative_path}"
            )
        identities.add(identity)
        paths.add(record.relative_path)

        if record.frame_id % frame_stride != 0:
            raise ValueError("Manifest contains a frame outside the stride rule.")
        if record.split != split_from_session(record.session):
            raise ValueError("Manifest record violates the frozen session split.")
        expected_label = category_label_from_object_id(record.object_id)
        if record.category_label != expected_label:
            raise ValueError("Manifest record has an invalid category label.")
        if record.category_name != CORE50_CATEGORY_NAMES[expected_label]:
            raise ValueError("Manifest record has an invalid category name.")

        session_splits[record.session].add(record.split)
        split_classes[record.split].add(record.category_label)

    leaked_sessions = {
        session: splits
        for session, splits in session_splits.items()
        if len(splits) != 1
    }
    if leaked_sessions:
        raise ValueError(f"Sessions leak across splits: {leaked_sessions}")

    expected_classes = set(range(10))
    for split in CORE50_SESSION_SPLITS:
        if split_classes[split] != expected_classes:
            raise ValueError(
                f"Split {split} does not contain all 10 categories."
            )


def verify_selected_images(
    dataset_root: Path,
    records: list[Core50Record],
) -> None:
    """Fully decode every selected RGB image and reject corrupt/wrong-size data."""

    for record in records:
        path = Path(dataset_root) / record.relative_path
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                rgb.load()
                if rgb.size != (128, 128):
                    raise ValueError(
                        f"Unexpected image size {rgb.size}: {path}"
                    )
        except Exception as error:
            raise ValueError(f"Failed to decode CORe50 image: {path}") from error


def write_manifest(path: Path, records: list[Core50Record]) -> None:
    """Write records in deterministic order using a stable CSV schema."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].to_row())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.to_row() for record in records)


def load_manifest(path: Path) -> list[Core50Record]:
    """Load a frozen manifest and revalidate every protocol field."""

    records: list[Core50Record] = []
    with Path(path).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        expected_fields = list(Core50Record.__dataclass_fields__)
        if reader.fieldnames != expected_fields:
            raise ValueError(
                "Unexpected CORe50 manifest schema: "
                f"{reader.fieldnames}"
            )
        for row in reader:
            records.append(Core50Record(
                relative_path=row["relative_path"],
                split=row["split"],
                session=int(row["session"]),
                object_id=int(row["object_id"]),
                frame_id=int(row["frame_id"]),
                category_label=int(row["category_label"]),
                category_name=row["category_name"],
            ))

    validate_core50_records(records)
    return records


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a lowercase SHA-256 digest without loading a file into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_core50_records(
    records: list[Core50Record],
) -> dict[str, object]:
    """Return split and class counts suitable for provenance JSON."""

    split_counts = Counter(record.split for record in records)
    class_counts: dict[str, dict[str, int]] = {}
    for split in CORE50_SESSION_SPLITS:
        counts = Counter(
            record.category_name
            for record in records
            if record.split == split
        )
        class_counts[split] = {
            name: counts[name]
            for name in CORE50_CATEGORY_NAMES
        }

    return {
        "selected_image_count": len(records),
        "split_counts": {
            split: split_counts[split]
            for split in CORE50_SESSION_SPLITS
        },
        "class_counts": class_counts,
    }


class Core50Dataset(Dataset[tuple[Tensor, int]]):
    """Read one frozen CORe50 split from paths stored in the manifest."""

    def __init__(
        self,
        dataset_root: Path,
        records: list[Core50Record],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        if not records:
            raise ValueError("Core50Dataset records cannot be empty.")
        self.dataset_root = Path(dataset_root)
        self.records = tuple(records)
        self.transform = transform
        self.classes = list(CORE50_CATEGORY_NAMES)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        image_path = self.dataset_root / record.relative_path
        with Image.open(image_path) as source:
            # load/convert 后数据脱离文件句柄，避免 worker 长时间占用 PNG。
            image = source.convert("RGB")
            image.load()
        return self.transform(image), record.category_label


def build_core50_transforms(
    augmentation: str = "basic",
) -> tuple[transforms.Compose, transforms.Compose]:
    """Build 32px source-compatible train and deterministic eval transforms."""

    augmentation_name = augmentation.strip().lower()
    resize_operation = transforms.Resize(
        (32, 32),
        interpolation=transforms.InterpolationMode.BILINEAR,
        antialias=True,
    )
    if augmentation_name == "basic":
        train_operations = [
            resize_operation,
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif augmentation_name == "none":
        train_operations = [resize_operation]
    else:
        raise ValueError("augmentation must be one of: basic, none.")

    common_operations = [
        transforms.ToTensor(),
        # 沿用源任务数值约定，避免把 normalization 变成第四个实验变量。
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    train_transform = transforms.Compose(
        [*train_operations, *common_operations]
    )
    eval_transform = transforms.Compose(
        [resize_operation, *common_operations]
    )
    return train_transform, eval_transform


def build_core50_datasets(
    dataset_root: Path,
    manifest_path: Path,
    *,
    augmentation: str = "basic",
) -> tuple[Core50Dataset, Core50Dataset, Core50Dataset, list[str]]:
    """Build the three datasets from immutable manifest membership."""

    records = load_manifest(manifest_path)
    train_transform, eval_transform = build_core50_transforms(augmentation)
    by_split = {
        split: [record for record in records if record.split == split]
        for split in CORE50_SESSION_SPLITS
    }
    train_dataset = Core50Dataset(
        dataset_root,
        by_split["train"],
        train_transform,
    )
    validation_dataset = Core50Dataset(
        dataset_root,
        by_split["validation"],
        eval_transform,
    )
    test_dataset = Core50Dataset(
        dataset_root,
        by_split["test"],
        eval_transform,
    )
    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        list(CORE50_CATEGORY_NAMES),
    )


def build_core50_dataloaders(
    dataset_root: Path,
    manifest_path: Path,
    *,
    batch_size: int,
    num_workers: int,
    train_seed: int,
    augmentation: str = "basic",
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build shuffled train and sequential evaluation CORe50 loaders."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if isinstance(train_seed, bool) or not isinstance(train_seed, int):
        raise ValueError("train_seed must be an integer.")
    if not 0 <= train_seed < 2**32:
        raise ValueError("train_seed must be in the range [0, 2**32).")

    train_dataset, validation_dataset, test_dataset, class_names = (
        build_core50_datasets(
            dataset_root,
            manifest_path,
            augmentation=augmentation,
        )
    )
    generator = torch.Generator().manual_seed(train_seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, validation_loader, test_loader, class_names
