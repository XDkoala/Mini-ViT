"""Training-target regularization utilities."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BatchMixResult:
    """A mixed image batch and the exact soft-target construction metadata."""

    images: Tensor
    targets: Tensor
    lambda_value: float
    permutation: Tensor


@dataclass(frozen=True)
class BoundingBox:
    """A right-open image-space rectangle [x1, x2) x [y1, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        """Return the exact number of pixels inside the rectangle."""

        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class CutMixResult(BatchMixResult):
    """A CutMix result with both sampled and area-corrected metadata."""

    sampled_lambda_value: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class Mixup:
    """Apply one symmetric-Beta Mixup sample to each training batch."""

    num_classes: int
    alpha: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_classes, bool)
            or not isinstance(self.num_classes, int)
            or self.num_classes <= 1
        ):
            raise ValueError("num_classes must be an integer greater than one.")
        object.__setattr__(self, "alpha", validate_mixup_alpha(self.alpha))

    def __call__(self, images: Tensor, labels: Tensor) -> BatchMixResult:
        return mixup_batch(
            images,
            labels,
            num_classes=self.num_classes,
            alpha=self.alpha,
        )


@dataclass(frozen=True)
class CutMix:
    """Paste one random rectangle per batch and correct targets by area."""

    num_classes: int
    alpha: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_classes, bool)
            or not isinstance(self.num_classes, int)
            or self.num_classes <= 1
        ):
            raise ValueError("num_classes must be an integer greater than one.")
        object.__setattr__(self, "alpha", validate_mixup_alpha(self.alpha))

    def __call__(self, images: Tensor, labels: Tensor) -> CutMixResult:
        return cutmix_batch(
            images,
            labels,
            num_classes=self.num_classes,
            alpha=self.alpha,
        )


def validate_label_smoothing(value: object) -> float:
    """Return a finite label-smoothing factor in the half-open range [0, 1)."""

    # bool 是 int 的子类，但 true/false 不应被误当成 1.0/0.0 强度。
    if isinstance(value, bool):
        raise ValueError(
            "training.label_smoothing must be a finite number "
            "in the range [0, 1)."
        )

    try:
        factor = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "training.label_smoothing must be a finite number "
            "in the range [0, 1)."
        ) from error

    if not math.isfinite(factor) or not 0.0 <= factor < 1.0:
        raise ValueError(
            "training.label_smoothing must be a finite number "
            "in the range [0, 1)."
        )

    return factor


def build_classification_criteria(
    label_smoothing: float,
) -> tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss]:
    """Build a regularized training loss and a hard-label validation loss."""

    factor = validate_label_smoothing(label_smoothing)

    # 正则化只属于训练目标；验证始终保留硬标签交叉熵这一统一尺度。
    return (
        nn.CrossEntropyLoss(label_smoothing=factor),
        nn.CrossEntropyLoss(),
    )


def validate_mixup_alpha(value: object) -> float:
    """Return a finite, strictly positive Mixup Beta shape parameter."""

    if isinstance(value, bool):
        raise ValueError("mixup alpha must be a finite number greater than zero.")

    try:
        alpha = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "mixup alpha must be a finite number greater than zero."
        ) from error

    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("mixup alpha must be a finite number greater than zero.")

    return alpha


def normalize_batch_mixing_config(value: object) -> dict[str, object]:
    """Normalize an optional batch-mixing mapping for snapshots and builders."""

    if value is None:
        return {"name": "none"}
    if not isinstance(value, Mapping):
        raise ValueError("training.batch_mixing must be a mapping or null.")

    name = str(value.get("name", "none")).strip().lower()
    if name == "none":
        return {"name": "none"}
    if name not in {"mixup", "cutmix"}:
        raise ValueError(
            "training.batch_mixing.name must be one of: none, mixup, cutmix."
        )
    if "alpha" not in value:
        raise ValueError(
            f"{name.title()} requires training.batch_mixing.alpha."
        )

    return {
        "name": name,
        "alpha": validate_mixup_alpha(value["alpha"]),
    }


def build_batch_mixer(
    config: Mapping[str, object],
    *,
    num_classes: int,
) -> Mixup | CutMix | None:
    """Build the normalized optional training-batch regularizer."""

    normalized = normalize_batch_mixing_config(config)
    if normalized["name"] == "none":
        return None

    mixer_type = Mixup if normalized["name"] == "mixup" else CutMix
    return mixer_type(
        num_classes=num_classes,
        alpha=float(normalized["alpha"]),
    )


def mixup_batch(
    images: Tensor,
    labels: Tensor,
    *,
    num_classes: int,
    alpha: float,
    lambda_value: float | None = None,
    permutation: Tensor | None = None,
) -> BatchMixResult:
    """Mix one image batch and return probability targets plus exact metadata."""

    alpha = validate_mixup_alpha(alpha)

    if images.ndim < 2 or not images.is_floating_point():
        raise ValueError("images must be a floating-point batch tensor.")
    if labels.ndim != 1 or labels.size(0) != images.size(0):
        raise ValueError("labels must have shape [batch_size].")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long class indices.")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 1
    ):
        raise ValueError("num_classes must be an integer greater than one.")
    if images.device != labels.device:
        raise ValueError("images and labels must be on the same device.")

    batch_size = images.size(0)
    if batch_size == 0:
        raise ValueError("Mixup requires a non-empty batch.")

    if lambda_value is None:
        beta = torch.distributions.Beta(alpha, alpha)
        lambda_value = float(beta.sample().item())
    elif isinstance(lambda_value, bool):
        raise ValueError("lambda_value must be finite and in the range [0, 1].")
    else:
        lambda_value = float(lambda_value)

    if not math.isfinite(lambda_value) or not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_value must be finite and in the range [0, 1].")

    if permutation is None:
        permutation = torch.randperm(batch_size, device=images.device)
    else:
        if (
            permutation.dtype != torch.long
            or permutation.ndim != 1
            or permutation.shape != (batch_size,)
            or permutation.device != images.device
        ):
            raise ValueError(
                "permutation must be a same-device torch.long vector "
                "with shape [batch_size]."
            )
        expected = torch.arange(batch_size, device=images.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("permutation must contain every batch index once.")

    paired_images = images[permutation]
    hard_targets = F.one_hot(labels, num_classes=num_classes).to(
        dtype=torch.float32
    )
    paired_targets = hard_targets[permutation]

    mixed_images = (
        lambda_value * images
        + (1.0 - lambda_value) * paired_images
    )
    mixed_targets = (
        lambda_value * hard_targets
        + (1.0 - lambda_value) * paired_targets
    )

    return BatchMixResult(
        images=mixed_images,
        targets=mixed_targets,
        lambda_value=lambda_value,
        permutation=permutation,
    )


def cutmix_batch(
    images: Tensor,
    labels: Tensor,
    *,
    num_classes: int,
    alpha: float,
    lambda_value: float | None = None,
    permutation: Tensor | None = None,
    center: tuple[int, int] | None = None,
) -> CutMixResult:
    """Paste paired image regions and correct target weights by actual area."""

    alpha = validate_mixup_alpha(alpha)
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("CutMix images must be a floating [B, C, H, W] tensor.")
    if labels.ndim != 1 or labels.size(0) != images.size(0):
        raise ValueError("labels must have shape [batch_size].")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long class indices.")
    if images.device != labels.device:
        raise ValueError("images and labels must be on the same device.")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 1
    ):
        raise ValueError("num_classes must be an integer greater than one.")

    batch_size, _, height, width = images.shape
    if batch_size == 0 or height == 0 or width == 0:
        raise ValueError("CutMix requires a non-empty spatial image batch.")

    if lambda_value is None:
        beta = torch.distributions.Beta(alpha, alpha)
        sampled_lambda = float(beta.sample().item())
    elif isinstance(lambda_value, bool):
        raise ValueError("lambda_value must be finite and in the range [0, 1].")
    else:
        sampled_lambda = float(lambda_value)
    if not math.isfinite(sampled_lambda) or not 0.0 <= sampled_lambda <= 1.0:
        raise ValueError("lambda_value must be finite and in the range [0, 1].")

    if permutation is None:
        permutation = torch.randperm(batch_size, device=images.device)
    else:
        if (
            permutation.dtype != torch.long
            or permutation.shape != (batch_size,)
            or permutation.device != images.device
        ):
            raise ValueError(
                "permutation must be a same-device torch.long vector "
                "with shape [batch_size]."
            )
        expected = torch.arange(batch_size, device=images.device)
        if not torch.equal(permutation.sort().values, expected):
            raise ValueError("permutation must contain every batch index once.")

    if center is None:
        center_x = int(torch.randint(0, width, ()).item())
        center_y = int(torch.randint(0, height, ()).item())
    else:
        if (
            len(center) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in center)
        ):
            raise ValueError("center must contain integer (x, y) coordinates.")
        center_x, center_y = center
        if not 0 <= center_x < width or not 0 <= center_y < height:
            raise ValueError("center must lie inside the image.")

    cut_ratio = math.sqrt(1.0 - sampled_lambda)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)
    left = cut_width // 2
    right = cut_width - left
    top = cut_height // 2
    bottom = cut_height - top
    box = BoundingBox(
        x1=max(center_x - left, 0),
        y1=max(center_y - top, 0),
        x2=min(center_x + right, width),
        y2=min(center_y + bottom, height),
    )

    mixed_images = images.clone()
    mixed_images[:, :, box.y1:box.y2, box.x1:box.x2] = images[
        permutation,
        :,
        box.y1:box.y2,
        box.x1:box.x2,
    ]

    actual_lambda = 1.0 - box.area / (height * width)
    hard_targets = F.one_hot(labels, num_classes=num_classes).to(
        dtype=torch.float32
    )
    mixed_targets = (
        actual_lambda * hard_targets
        + (1.0 - actual_lambda) * hard_targets[permutation]
    )

    return CutMixResult(
        images=mixed_images,
        targets=mixed_targets,
        lambda_value=actual_lambda,
        permutation=permutation,
        sampled_lambda_value=sampled_lambda,
        bounding_box=box,
    )
