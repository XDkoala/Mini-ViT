"""Tests for training-target regularization utilities."""

import pytest
import torch

from src.regularization import (
    BatchMixResult,
    BoundingBox,
    CutMix,
    Mixup,
    build_batch_mixer,
    build_classification_criteria,
    cutmix_batch,
    mixup_batch,
    normalize_batch_mixing_config,
    validate_label_smoothing,
    validate_mixup_alpha,
)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [(0, 0.0), (0.05, 0.05), ("0.1", 0.1)],
)
def test_validate_label_smoothing_normalizes_valid_values(
    raw_value,
    expected,
):
    """Normalize supported numeric inputs to a reproducible float."""

    assert validate_label_smoothing(raw_value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw_value",
    [True, False, None, "invalid", float("nan"), float("inf"), -0.1, 1.0],
)
def test_validate_label_smoothing_rejects_invalid_values(raw_value):
    """Reject ambiguous, non-finite, or supervision-erasing factors."""

    with pytest.raises(ValueError, match=r"range \[0, 1\)"):
        validate_label_smoothing(raw_value)


def test_build_classification_criteria_regularizes_training_only():
    """Apply smoothing to training while keeping validation hard-labelled."""

    train_criterion, val_criterion = build_classification_criteria(0.1)
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    labels = torch.tensor([0])

    assert train_criterion.label_smoothing == pytest.approx(0.1)
    assert val_criterion.label_smoothing == pytest.approx(0.0)
    assert train_criterion(logits, labels) == pytest.approx(0.3365127)
    assert val_criterion(logits, labels) == pytest.approx(0.1698460)


@pytest.mark.parametrize("alpha", [0.2, 0.4, "1.0"])
def test_validate_mixup_alpha_accepts_positive_finite_values(alpha):
    """Normalize valid symmetric-Beta shape parameters."""

    assert validate_mixup_alpha(alpha) == pytest.approx(float(alpha))


@pytest.mark.parametrize(
    "alpha",
    [True, False, None, "invalid", float("nan"), float("inf"), 0.0, -0.1],
)
def test_validate_mixup_alpha_rejects_invalid_values(alpha):
    """Reject values that cannot define the intended Mixup distribution."""

    with pytest.raises(ValueError, match="greater than zero"):
        validate_mixup_alpha(alpha)


def test_mixup_batch_uses_one_lambda_and_permutation_for_images_and_targets():
    """Mix inputs and labels with the same exact batch-level construction."""

    images = torch.tensor([
        [[[0.0, 2.0]]],
        [[[4.0, 6.0]]],
    ])
    labels = torch.tensor([0, 2])
    permutation = torch.tensor([1, 0])

    result = mixup_batch(
        images,
        labels,
        num_classes=3,
        alpha=0.2,
        lambda_value=0.25,
        permutation=permutation,
    )

    assert isinstance(result, BatchMixResult)
    assert result.lambda_value == pytest.approx(0.25)
    assert torch.equal(result.permutation, permutation)
    assert torch.allclose(
        result.images,
        torch.tensor([
            [[[3.0, 5.0]]],
            [[[1.0, 3.0]]],
        ]),
    )
    assert torch.allclose(
        result.targets,
        torch.tensor([
            [0.25, 0.0, 0.75],
            [0.75, 0.0, 0.25],
        ]),
    )
    assert torch.allclose(result.targets.sum(dim=1), torch.ones(2))

    # 函数必须返回新张量，不能原地污染 DataLoader 提供的原 batch。
    assert torch.equal(images, torch.tensor([
        [[[0.0, 2.0]]],
        [[[4.0, 6.0]]],
    ]))


def test_mixup_soft_target_loss_matches_weighted_hard_label_losses():
    """Confirm soft-target CE equals the defining weighted two-label loss."""

    logits = torch.tensor([
        [2.0, 0.0, -1.0],
        [-1.0, 0.0, 2.0],
    ])
    images = torch.zeros(2, 1, 1, 1)
    labels = torch.tensor([0, 2])
    permutation = torch.tensor([1, 0])
    lambda_value = 0.7
    result = mixup_batch(
        images,
        labels,
        num_classes=3,
        alpha=0.2,
        lambda_value=lambda_value,
        permutation=permutation,
    )

    soft_target_loss = torch.nn.functional.cross_entropy(
        logits,
        result.targets,
    )
    weighted_loss = (
        lambda_value * torch.nn.functional.cross_entropy(logits, labels)
        + (1.0 - lambda_value)
        * torch.nn.functional.cross_entropy(logits, labels[permutation])
    )

    assert soft_target_loss == pytest.approx(weighted_loss)


def test_mixup_batch_is_reproducible_after_resetting_torch_seed():
    """Reproduce both Beta sampling and batch pairing from the train seed."""

    images = torch.arange(16, dtype=torch.float32).reshape(4, 1, 2, 2)
    labels = torch.tensor([0, 1, 2, 3])

    torch.manual_seed(123)
    first = mixup_batch(images, labels, num_classes=4, alpha=0.4)
    torch.manual_seed(123)
    second = mixup_batch(images, labels, num_classes=4, alpha=0.4)

    assert first.lambda_value == second.lambda_value
    assert torch.equal(first.permutation, second.permutation)
    assert torch.equal(first.images, second.images)
    assert torch.equal(first.targets, second.targets)


def test_normalize_batch_mixing_config_canonicalizes_none_and_mixup():
    """Persist explicit canonical batch-mixing settings in run snapshots."""

    assert normalize_batch_mixing_config(None) == {"name": "none"}
    assert normalize_batch_mixing_config({"name": " NONE "}) == {
        "name": "none"
    }
    assert normalize_batch_mixing_config({
        "name": " MIXUP ",
        "alpha": "0.2",
    }) == {
        "name": "mixup",
        "alpha": pytest.approx(0.2),
    }


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (True, "mapping or null"),
        ({"name": "unknown", "alpha": 1.0}, "must be one of"),
        ({"name": "mixup"}, "requires"),
    ],
)
def test_normalize_batch_mixing_config_rejects_invalid_settings(
    config,
    message,
):
    """Reject ambiguous or incomplete strategy descriptions."""

    with pytest.raises(ValueError, match=message):
        normalize_batch_mixing_config(config)


def test_build_batch_mixer_returns_configured_strategy():
    """Build exactly the optional strategy described by the snapshot."""

    assert build_batch_mixer({"name": "none"}, num_classes=10) is None

    mixer = build_batch_mixer(
        {"name": "mixup", "alpha": 0.4},
        num_classes=10,
    )
    assert isinstance(mixer, Mixup)
    assert mixer.num_classes == 10
    assert mixer.alpha == pytest.approx(0.4)

    cutmix = build_batch_mixer(
        {"name": "cutmix", "alpha": 1.0},
        num_classes=10,
    )
    assert isinstance(cutmix, CutMix)
    assert cutmix.alpha == pytest.approx(1.0)


def test_cutmix_batch_replaces_exact_region_and_builds_area_targets():
    """Paste the paired rectangle without modifying either source batch."""

    images = torch.stack((
        torch.zeros(1, 4, 4),
        torch.ones(1, 4, 4),
    ))
    original = images.clone()
    result = cutmix_batch(
        images,
        torch.tensor([0, 1]),
        num_classes=2,
        alpha=1.0,
        lambda_value=0.75,
        permutation=torch.tensor([1, 0]),
        center=(2, 2),
    )

    assert result.bounding_box == BoundingBox(x1=1, y1=1, x2=3, y2=3)
    assert result.bounding_box.area == 4
    assert result.sampled_lambda_value == pytest.approx(0.75)
    assert result.lambda_value == pytest.approx(0.75)
    assert torch.equal(result.images[0, :, 1:3, 1:3], torch.ones(1, 2, 2))
    assert torch.equal(result.images[1, :, 1:3, 1:3], torch.zeros(1, 2, 2))
    assert torch.allclose(
        result.targets,
        torch.tensor([[0.75, 0.25], [0.25, 0.75]]),
    )
    assert torch.equal(images, original)


def test_cutmix_batch_corrects_lambda_after_edge_clipping():
    """Derive target weights from the clipped box rather than sampled lambda."""

    result = cutmix_batch(
        torch.zeros(2, 1, 6, 8),
        torch.tensor([0, 1]),
        num_classes=2,
        alpha=1.0,
        lambda_value=0.75,
        permutation=torch.tensor([1, 0]),
        center=(1, 5),
    )

    assert result.bounding_box == BoundingBox(x1=0, y1=4, x2=3, y2=6)
    assert result.bounding_box.area == 6
    assert result.sampled_lambda_value == pytest.approx(0.75)
    assert result.lambda_value == pytest.approx(0.875)
    assert torch.allclose(
        result.targets,
        torch.tensor([[0.875, 0.125], [0.125, 0.875]]),
    )
