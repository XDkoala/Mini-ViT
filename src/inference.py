"""Shared checkpoint loading, preprocessing, and prediction for Stage 17."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms import functional as vision_functional

from src.config import resolve_format_version
from src.core50 import CORE50_CATEGORY_NAMES, build_core50_transforms, sha256_file
from src.data import (
    CIFAR10_CLASS_NAMES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    build_transforms,
)
from src.models import MiniViT, build_model
from src.transfer import validate_transfer_mode
from src.utils import resolve_device
from src.visualization import attention_rollout, patch_weights_to_heatmaps


ATTENTION_METHODS = ("none", "last", "rollout", "both")


@dataclass
class InferenceRuntime:
    """A restored model and its label/preprocessing contract."""

    model: nn.Module
    config: dict[str, object]
    class_names: tuple[str, ...]
    dataset: str
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    epoch: int

    def prepare_image(self, image: Image.Image) -> tuple[Tensor, Tensor]:
        """Return normalized model input and the exact displayed RGB pixels."""

        rgb = ImageOps.exif_transpose(image).convert("RGB")
        if self.dataset == "core50":
            # Reuse the deterministic CORe50 evaluation transform.
            # 直接复用正式评估的缩放、ToTensor 和 Normalize。
            _, evaluation_transform = build_core50_transforms("none")
        else:
            # CIFAR evaluation receives native 32px images. External images
            # are explicitly resized first; native CIFAR images are untouched.
            # 原始 CIFAR 样本无需缩放；外部图片先按记录的规则缩到 32×32。
            if rgb.size != (32, 32):
                rgb = transforms.Resize(
                    (32, 32),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )(rgb)
            _, evaluation_transform = build_transforms("none")
        normalized = evaluation_transform(rgb)
        if self.dataset == "core50":
            rgb = transforms.Resize(
                (32, 32),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )(rgb)
        pixels = vision_functional.to_tensor(rgb)
        return normalized, pixels

    def metadata(self) -> dict[str, object]:
        """Describe model identity and the exact preprocessing convention."""

        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_epoch": self.epoch,
            "experiment": self.config.get("experiment", {}).get("name"),
            "dataset": self.dataset,
            "class_names": list(self.class_names),
            "device": str(self.device),
            "preprocessing": {
                "color": "EXIF orientation then RGB",
                "input_size": [32, 32],
                "resize": (
                    "always resize to 32x32: bilinear with antialias"
                    if self.dataset == "core50"
                    else "native 32x32 unchanged; external sizes use bilinear with antialias"
                ),
                "to_tensor": "float32 CHW in [0,1]",
                "mean": list(CIFAR10_MEAN),
                "std": list(CIFAR10_STD),
                "random_augmentation": False,
                "display": "heatmaps overlay the resized model input",
            },
        }


def load_inference_runtime(
    path: str | Path,
    *,
    device: str = "auto",
    expected_sha256: str | None = None,
) -> InferenceRuntime:
    """Restore target weights directly, without loading a source checkpoint."""

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    digest = sha256_file(checkpoint_path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 differs from the pinned value.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a mapping.")
    resolve_format_version(checkpoint)
    if not {"config", "model_state", "epoch"}.issubset(checkpoint):
        raise ValueError("Checkpoint requires config, model_state, and epoch.")
    config = checkpoint["config"]
    if not isinstance(config, dict) or not all(
        isinstance(config.get(key), dict) for key in ("data", "model")
    ):
        raise ValueError("Checkpoint must contain data/model configuration mappings.")
    if "experiment" in config and not isinstance(config["experiment"], dict):
        raise ValueError("Checkpoint experiment configuration must be a mapping.")

    data_config = config["data"]
    if "transfer" in config:
        if not isinstance(config["transfer"], dict):
            raise ValueError("transfer configuration must be a mapping.")
        validate_transfer_mode(str(config["transfer"].get("mode", "")))
        if not {"dataset_root", "manifest", "manifest_sha256"}.issubset(data_config):
            raise ValueError("CORe50 checkpoint is missing its dataset contract.")
        dataset, class_names = "core50", CORE50_CATEGORY_NAMES
    elif "data_dir" in data_config and "manifest" not in data_config:
        dataset, class_names = "cifar10", CIFAR10_CLASS_NAMES
    else:
        raise ValueError("Cannot identify checkpoint dataset as CIFAR-10 or CORe50.")

    model_config = config["model"]
    if int(model_config.get("num_classes", 10)) != len(class_names):
        raise ValueError("Classifier size differs from the dataset class mapping.")
    if int(model_config.get("image_size", 32)) != 32 or int(
        model_config.get("in_channels", 3)
    ) != 3:
        raise ValueError("These dataset contracts require RGB 32x32 model inputs.")
    target_device = resolve_device(device)
    # The target checkpoint already includes its new classifier. Restore all
    # weights strictly; inference does not repeat transfer initialization.
    # 目标 checkpoint 已含新分类头，推理严格加载全部权重，无需源域文件。
    model = build_model(model_config, target_device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return InferenceRuntime(
        model, config, tuple(class_names), dataset, target_device,
        checkpoint_path, digest, int(checkpoint["epoch"]),
    )


@dataclass
class PredictionBatch:
    """CPU outputs ready for JSON and plotting, with raw weights retained."""

    logits: Tensor
    probabilities: Tensor
    patch_weights: dict[str, Tensor]
    heatmaps: dict[str, Tensor]


@torch.inference_mode()
def predict_batch(
    runtime: InferenceRuntime,
    images: Tensor,
    *,
    attention: str = "none",
) -> PredictionBatch:
    """Use one forward pass for predictions and the requested attention maps."""

    if attention not in ATTENTION_METHODS:
        raise ValueError(f"attention must be one of: {', '.join(ATTENTION_METHODS)}.")
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32) or len(images) == 0:
        raise ValueError("images must have non-empty shape [B,3,32,32].")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("images must contain finite floating-point values.")
    if attention != "none" and not isinstance(runtime.model, MiniViT):
        raise ValueError("Attention visualization requires a MiniViT checkpoint.")
    runtime.model.eval()
    inputs = images.to(runtime.device)
    weights = {}
    if attention == "none":
        logits = runtime.model(inputs)
    elif attention == "last":
        logits, last = runtime.model(inputs, return_attention=True)
        weights["last"] = last[:, :, 0, 1:].mean(dim=1)
    else:
        logits, all_layers = runtime.model(
            inputs, return_attention=True, attention_layers="all"
        )
        if attention == "both":
            weights["last"] = all_layers[-1, :, :, 0, 1:].mean(dim=1)
        weights["rollout"] = attention_rollout(all_layers)
    if logits.shape != (len(images), len(runtime.class_names)) or not torch.isfinite(logits).all():
        raise ValueError("Model produced invalid classification logits.")
    heatmaps = {
        method: patch_weights_to_heatmaps(values, image_size=32).cpu()
        for method, values in weights.items()
    }
    return PredictionBatch(
        logits.cpu(), logits.softmax(dim=-1).cpu(),
        {method: values.cpu() for method, values in weights.items()}, heatmaps,
    )


def top_k_predictions(
    probabilities: Tensor, class_names: tuple[str, ...], k: int
) -> list[dict[str, object]]:
    """Sort probabilities descending, breaking exact ties by label index."""

    if probabilities.ndim != 1 or len(probabilities) != len(class_names):
        raise ValueError("probabilities must match the class mapping.")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(class_names):
        raise ValueError(f"top-k must be between 1 and {len(class_names)}.")
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative.")
    indices = torch.argsort(probabilities, descending=True, stable=True)[:k]
    return [
        {"label": int(index), "class": class_names[int(index)],
         "probability": float(probabilities[index])}
        for index in indices
    ]
