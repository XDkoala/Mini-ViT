"""Tests for deterministic Stage 15 image corruptions."""

import pytest
import torch
from torch.utils.data import Dataset

from src.corruptions import (
    CORRUPTION_NAMES,
    CorruptedSubset,
    apply_corruption,
    validate_corruption,
)
from src.data import CIFAR10_MEAN, CIFAR10_STD


class RawTensorDataset(Dataset):
    """Small raw [0,1] image fixture with index-identifiable labels."""

    def __init__(self) -> None:
        self.images = [
            torch.full((3, 32, 32), fill_value=index / 10.0)
            for index in range(5)
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return self.images[index], index


def test_registered_corruption_parameters() -> None:
    assert validate_corruption("brightness", 1) == ("brightness", 0.8)
    assert validate_corruption("motion_blur", 3) == ("motion_blur", 7)
    assert validate_corruption("occlusion", 2) == ("occlusion", 10)


def test_invalid_corruption_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="name must be one of"):
        validate_corruption("unknown", 1)

    with pytest.raises(ValueError, match="severity"):
        validate_corruption("brightness", 0)


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_corruptions_preserve_shape_range_and_input(
    name: str,
    severity: int,
) -> None:
    image = torch.linspace(0.0, 1.0, 3 * 32 * 32).reshape(3, 32, 32)
    original = image.clone()

    output = apply_corruption(
        image,
        name,
        severity,
        sample_index=7,
        corruption_seed=1501,
    )

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert torch.equal(image, original)


@pytest.mark.parametrize("name", CORRUPTION_NAMES)
def test_same_key_is_exactly_reproducible(name: str) -> None:
    image = torch.rand(3, 32, 32)

    first = apply_corruption(
        image,
        name,
        2,
        sample_index=123,
        corruption_seed=1501,
    )
    second = apply_corruption(
        image,
        name,
        2,
        sample_index=123,
        corruption_seed=1501,
    )

    assert torch.equal(first, second)


def test_noise_uses_same_base_pattern_across_severities() -> None:
    image = torch.full((3, 32, 32), 0.5)
    low = apply_corruption(
        image,
        "gaussian_noise",
        1,
        sample_index=12,
        corruption_seed=1501,
    )
    high = apply_corruption(
        image,
        "gaussian_noise",
        2,
        sample_index=12,
        corruption_seed=1501,
    )

    # 远离 clamp 边界的像素应使用同一噪声图案，幅度正好翻倍。
    safe = ((low > 0.1) & (low < 0.9) & (high > 0.1) & (high < 0.9))
    assert torch.allclose(
        (high - image)[safe],
        2.0 * (low - image)[safe],
        rtol=1e-5,
        atol=1e-7,
    )


def test_occlusion_area_matches_registered_severity() -> None:
    image = torch.ones(3, 32, 32)

    for severity, side in ((1, 6), (2, 10), (3, 14)):
        output = apply_corruption(
            image,
            "occlusion",
            severity,
            sample_index=9,
            corruption_seed=1501,
        )
        zero_pixels = int((output[0] == 0.0).sum().item())
        assert zero_pixels == side**2


def test_brightness_distance_increases_with_severity() -> None:
    image = torch.full((3, 32, 32), 0.5)
    distances = []

    for severity in (1, 2, 3):
        output = apply_corruption(
            image,
            "brightness",
            severity,
            sample_index=1,
            corruption_seed=1501,
        )
        distances.append(float((output - image).abs().mean()))

    assert distances[0] < distances[1] < distances[2]


def test_corrupted_subset_preserves_length_order_and_labels() -> None:
    source = RawTensorDataset()
    indices = [4, 1, 3]
    dataset = CorruptedSubset(
        source=source,
        indices=indices,
        corruption_name="translation",
        severity=2,
        corruption_seed=1501,
    )

    assert len(dataset) == 3
    assert [dataset[index][1] for index in range(3)] == indices


def test_clean_subset_only_normalizes() -> None:
    source = RawTensorDataset()
    dataset = CorruptedSubset(
        source=source,
        indices=[2],
        corruption_name=None,
        severity=None,
    )

    image, label = dataset[0]
    expected = torch.empty_like(image)
    for channel, (mean, std) in enumerate(zip(CIFAR10_MEAN, CIFAR10_STD)):
        expected[channel].fill_((0.2 - mean) / std)

    assert label == 2
    assert torch.allclose(image, expected)


def test_access_order_does_not_change_corrupted_sample() -> None:
    source = RawTensorDataset()
    dataset = CorruptedSubset(
        source=source,
        indices=[4, 1, 3],
        corruption_name="gaussian_noise",
        severity=3,
        corruption_seed=1501,
    )

    expected, _ = dataset[1]
    dataset[2]
    dataset[0]
    actual, _ = dataset[1]

    assert torch.equal(actual, expected)
