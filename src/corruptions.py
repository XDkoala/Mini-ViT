"""Deterministic image corruptions for Stage 15 robustness evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as vision_functional

from src.data import CIFAR10_MEAN, CIFAR10_STD


CORRUPTION_LEVELS: dict[str, tuple[float | int, ...]] = {
    "brightness": (0.8, 0.6, 0.4),
    "contrast": (0.75, 0.50, 0.25),
    "gaussian_noise": (0.04, 0.08, 0.12),
    "motion_blur": (3, 5, 7),
    "occlusion": (6, 10, 14),
    "translation": (1, 2, 4),
}
CORRUPTION_NAMES = tuple(CORRUPTION_LEVELS)
SEVERITIES = (1, 2, 3)


def validate_corruption(name: str, severity: int) -> tuple[str, float | int]:
    """Validate a corruption selection and return its registered parameter."""

    normalized_name = name.strip().lower()

    if normalized_name not in CORRUPTION_LEVELS:
        choices = ", ".join(CORRUPTION_NAMES)
        raise ValueError(f"corruption name must be one of: {choices}.")

    if isinstance(severity, bool) or severity not in SEVERITIES:
        raise ValueError("severity must be one of: 1, 2, 3.")

    return normalized_name, CORRUPTION_LEVELS[normalized_name][severity - 1]


def _sample_generator(
    corruption_seed: int,
    sample_index: int,
    corruption_name: str,
) -> torch.Generator:
    """Create an access-order-independent RNG for one sample/corruption."""

    if (
        isinstance(corruption_seed, bool)
        or not isinstance(corruption_seed, int)
        or corruption_seed < 0
    ):
        raise ValueError("corruption_seed must be a non-negative integer.")

    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 0
    ):
        raise ValueError("sample_index must be a non-negative integer.")

    key = f"{corruption_seed}:{sample_index}:{corruption_name}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    torch_seed = int.from_bytes(digest, "little") % (2**63 - 1)
    return torch.Generator().manual_seed(torch_seed)


def _validate_image(image: Tensor) -> None:
    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W].")

    if image.device.type != "cpu":
        raise ValueError("dataset corruptions must run on CPU tensors.")

    if not image.is_floating_point():
        raise ValueError("image must be a floating-point tensor in [0, 1].")

    if not torch.isfinite(image).all():
        raise ValueError("image must contain only finite values.")

    if image.min().item() < 0.0 or image.max().item() > 1.0:
        raise ValueError("image values must be in the range [0, 1].")


def _adjust_contrast(image: Tensor, factor: float) -> Tensor:
    if image.shape[0] == 3:
        luminance = (
            0.2989 * image[0]
            + 0.5870 * image[1]
            + 0.1140 * image[2]
        )
        mean = luminance.mean()
    else:
        mean = image.mean()

    return mean + factor * (image - mean)


def _horizontal_motion_blur(image: Tensor, kernel_size: int) -> Tensor:
    channels = image.shape[0]
    kernel = torch.full(
        (channels, 1, 1, kernel_size),
        fill_value=1.0 / kernel_size,
        dtype=image.dtype,
    )
    padding = kernel_size // 2
    padded = functional.pad(
        image.unsqueeze(0),
        pad=(padding, padding, 0, 0),
        mode="replicate",
    )
    return functional.conv2d(
        padded,
        kernel,
        groups=channels,
    ).squeeze(0)


def _occlude(
    image: Tensor,
    side_length: int,
    generator: torch.Generator,
) -> Tensor:
    _, height, width = image.shape
    side_length = min(side_length, height, width)
    center_y = int(torch.randint(0, height, (), generator=generator).item())
    center_x = int(torch.randint(0, width, (), generator=generator).item())
    top = min(max(center_y - side_length // 2, 0), height - side_length)
    left = min(max(center_x - side_length // 2, 0), width - side_length)
    output = image.clone()
    output[:, top : top + side_length, left : left + side_length] = 0.0
    return output


def _translate(
    image: Tensor,
    distance: int,
    generator: torch.Generator,
) -> Tensor:
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    direction_index = int(
        torch.randint(0, len(directions), (), generator=generator).item()
    )
    direction_y, direction_x = directions[direction_index]
    shift_y = direction_y * distance
    shift_x = direction_x * distance
    output = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))

    if shift_y > 0:
        output[:, :shift_y, :] = 0.0
    elif shift_y < 0:
        output[:, shift_y:, :] = 0.0

    if shift_x > 0:
        output[:, :, :shift_x] = 0.0
    elif shift_x < 0:
        output[:, :, shift_x:] = 0.0

    return output


def apply_corruption(
    image: Tensor,
    name: str,
    severity: int,
    *,
    sample_index: int,
    corruption_seed: int,
) -> Tensor:
    """Apply one registered corruption without modifying the input Tensor."""

    _validate_image(image)
    corruption_name, parameter = validate_corruption(name, severity)
    generator = _sample_generator(
        corruption_seed=corruption_seed,
        sample_index=sample_index,
        corruption_name=corruption_name,
    )

    if corruption_name == "brightness":
        output = image * float(parameter)
    elif corruption_name == "contrast":
        output = _adjust_contrast(image, float(parameter))
    elif corruption_name == "gaussian_noise":
        noise = torch.randn(
            image.shape,
            generator=generator,
            dtype=image.dtype,
        )
        output = image + float(parameter) * noise
    elif corruption_name == "motion_blur":
        output = _horizontal_motion_blur(image, int(parameter))
    elif corruption_name == "occlusion":
        output = _occlude(image, int(parameter), generator)
    elif corruption_name == "translation":
        output = _translate(image, int(parameter), generator)
    else:  # pragma: no cover - validate_corruption makes this unreachable.
        raise AssertionError(f"Unhandled corruption: {corruption_name}")

    return output.clamp(0.0, 1.0)


class CorruptedSubset(Dataset):
    """Expose a fixed raw-image subset through one deterministic corruption."""

    def __init__(
        self,
        source: Dataset,
        indices: Sequence[int],
        corruption_name: str | None,
        severity: int | None,
        corruption_seed: int = 1501,
    ) -> None:
        self.source = source
        self.indices = tuple(int(index) for index in indices)
        self.corruption_seed = corruption_seed

        if corruption_name is None:
            if severity is not None:
                raise ValueError("clean data must not define severity.")
            self.corruption_name = None
            self.severity = None
        else:
            if severity is None:
                raise ValueError("corrupted data requires severity.")
            normalized_name, _ = validate_corruption(
                corruption_name,
                severity,
            )
            self.corruption_name = normalized_name
            self.severity = severity

        # Normalize 是确定性的，并且明确放在 corruption 之后。
        self.normalize = vision_functional.normalize

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[Tensor, int]:
        source_index = self.indices[position]
        image, label = self.source[source_index]

        if isinstance(image, Tensor):
            tensor = image.detach().clone()
            if not tensor.is_floating_point():
                tensor = tensor.to(torch.float32) / 255.0
        else:
            tensor = vision_functional.to_tensor(image)

        _validate_image(tensor)

        if self.corruption_name is not None:
            tensor = apply_corruption(
                tensor,
                self.corruption_name,
                int(self.severity),
                sample_index=source_index,
                corruption_seed=self.corruption_seed,
            )

        normalized = self.normalize(
            tensor,
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD,
        )
        return normalized, int(label)
