"""Transfer-learning models and checkpoint loading for Stage 16."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager

import torch
from torch import Tensor, nn

from src.models.minivit import MiniViT


TRANSFER_MODES = ("scratch", "linear_probe", "fine_tune")
CLASSIFIER_PREFIX = "classifier."

_MODEL_DEFAULTS = {
    "image_size": 32,
    "in_channels": 3,
    "num_classes": 10,
    "patch_size": 4,
    "embed_dim": 192,
    "depth": 6,
    "num_heads": 6,
    "mlp_dim": 768,
    "dropout": 0.1,
}
_ARCHITECTURE_FIELDS = (
    "image_size",
    "in_channels",
    "patch_size",
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_dim",
    "dropout",
)


@dataclass(frozen=True, slots=True)
class TransferInitialization:
    """Serializable evidence of how one target model was initialized."""

    mode: str
    source_checkpoint: str | None
    source_checkpoint_sha256: str | None
    loaded_encoder_key_count: int
    discarded_source_keys: tuple[str, ...]


def validate_transfer_mode(mode: str) -> str:
    """Normalize and validate a transfer-learning mode."""

    normalized = mode.strip().lower()
    if normalized not in TRANSFER_MODES:
        raise ValueError(
            "transfer mode must be one of: " + ", ".join(TRANSFER_MODES)
        )
    return normalized


class TransferMiniViT(MiniViT):
    """MiniViT with explicit scratch, probe, and fine-tune semantics."""

    def __init__(self, *, transfer_mode: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.transfer_mode = validate_transfer_mode(transfer_mode)
        self._configure_trainable_parameters()

    def encoder_named_parameters(self):
        """Yield every feature-extractor parameter, excluding the target head."""

        for name, parameter in self.named_parameters():
            if not name.startswith(CLASSIFIER_PREFIX):
                yield name, parameter

    def _configure_trainable_parameters(self) -> None:
        """Apply the mode's parameter-freezing contract."""

        encoder_requires_grad = self.transfer_mode != "linear_probe"
        for _, parameter in self.encoder_named_parameters():
            parameter.requires_grad_(encoder_requires_grad)
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "TransferMiniViT":
        """Keep a probed encoder deterministic while training its new head."""

        super().train(mode)
        if mode and self.transfer_mode == "linear_probe":
            # requires_grad=False 不会关闭 Dropout；encoder 必须单独 eval。
            self.patch_embedding.eval()
            self.blocks.eval()
            self.final_norm.eval()
            self.classifier.train(True)
        return self

    def forward(
        self,
        images: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run the encoder with no graph only in linear-probe mode."""

        encoder_context: ContextManager[None] = (
            torch.no_grad()
            if self.transfer_mode == "linear_probe"
            else nullcontext()
        )
        with encoder_context:
            encoded = self.forward_features(
                images,
                return_attention=return_attention,
            )

        if return_attention:
            features, attention = encoded
            return self.classifier(features), attention
        return self.classifier(encoded)


def _normalized_model_config(
    model_config: dict[str, object],
) -> dict[str, object]:
    """Resolve supported MiniViT fields without silently accepting extras."""

    allowed = {"name", *_MODEL_DEFAULTS}
    unknown = set(model_config) - allowed
    if unknown:
        raise ValueError(
            "Unknown transfer model config fields: "
            + ", ".join(sorted(unknown))
        )
    if str(model_config.get("name", "minivit")).lower() != "minivit":
        raise ValueError("Stage 16 transfer currently supports only MiniViT.")
    return {
        field: model_config.get(field, default)
        for field, default in _MODEL_DEFAULTS.items()
    }


def _sha256(path: Path) -> str:
    """Hash a source checkpoint for transfer provenance."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_encoder_from_checkpoint(
    model: TransferMiniViT,
    checkpoint_path: Path,
    device: torch.device,
    target_config: dict[str, object],
) -> TransferInitialization:
    """Load every source encoder tensor while explicitly discarding its head."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Source checkpoint root must be a mapping.")
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            "Source checkpoint must contain model_state and config."
        )

    source_config = checkpoint["config"]
    if not isinstance(source_config, dict):
        raise ValueError("Source checkpoint config must be a mapping.")
    source_model_config = source_config.get("model")
    if not isinstance(source_model_config, dict):
        raise ValueError("Source checkpoint config.model must be a mapping.")
    normalized_source = _normalized_model_config(source_model_config)
    mismatches = {
        field: (normalized_source[field], target_config[field])
        for field in _ARCHITECTURE_FIELDS
        if normalized_source[field] != target_config[field]
    }
    if mismatches:
        raise ValueError(
            "Source and target encoder architectures differ: "
            f"{mismatches}"
        )

    source_state = checkpoint["model_state"]
    if not isinstance(source_state, dict):
        raise ValueError("Source checkpoint model_state must be a mapping.")
    discarded_keys = tuple(sorted(
        key for key in source_state if key.startswith(CLASSIFIER_PREFIX)
    ))
    if discarded_keys != ("classifier.bias", "classifier.weight"):
        raise ValueError(
            "Source checkpoint must contain exactly classifier weight and bias."
        )
    encoder_state = {
        key: value
        for key, value in source_state.items()
        if not key.startswith(CLASSIFIER_PREFIX)
    }
    expected_encoder_keys = {
        key
        for key in model.state_dict()
        if not key.startswith(CLASSIFIER_PREFIX)
    }
    if set(encoder_state) != expected_encoder_keys:
        missing = expected_encoder_keys - set(encoder_state)
        unexpected = set(encoder_state) - expected_encoder_keys
        raise ValueError(
            "Source encoder state does not exactly match the target: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    incompatible = model.load_state_dict(encoder_state, strict=False)
    if set(incompatible.missing_keys) != {
        "classifier.weight",
        "classifier.bias",
    } or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected state-dict result while loading the source encoder."
        )

    return TransferInitialization(
        mode=model.transfer_mode,
        source_checkpoint=str(checkpoint_path),
        source_checkpoint_sha256=_sha256(checkpoint_path),
        loaded_encoder_key_count=len(encoder_state),
        discarded_source_keys=discarded_keys,
    )


def build_transfer_model(
    model_config: dict[str, object],
    *,
    mode: str,
    device: torch.device,
    source_checkpoint: Path | None = None,
) -> tuple[TransferMiniViT, TransferInitialization]:
    """Build one target model under the frozen three-way transfer protocol."""

    normalized_mode = validate_transfer_mode(mode)
    normalized_config = _normalized_model_config(model_config)
    model = TransferMiniViT(
        transfer_mode=normalized_mode,
        **normalized_config,
    ).to(device)

    if normalized_mode == "scratch":
        if source_checkpoint is not None:
            raise ValueError("Scratch mode must not receive a source checkpoint.")
        initialization = TransferInitialization(
            mode=normalized_mode,
            source_checkpoint=None,
            source_checkpoint_sha256=None,
            loaded_encoder_key_count=0,
            discarded_source_keys=(),
        )
    else:
        if source_checkpoint is None:
            raise ValueError(
                f"{normalized_mode} mode requires a source checkpoint."
            )
        initialization = _load_encoder_from_checkpoint(
            model,
            Path(source_checkpoint),
            device,
            normalized_config,
        )

    model._configure_trainable_parameters()
    return model, initialization


def trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    """Return the exact parameters that an optimizer is allowed to receive."""

    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return only parameters permitted to enter the target optimizer."""

    parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise ValueError("Transfer model has no trainable parameters.")
    return parameters


def clone_encoder_state(model: TransferMiniViT) -> dict[str, Tensor]:
    """Clone encoder tensors for exact post-update freeze verification."""

    return {
        name: parameter.detach().clone()
        for name, parameter in model.encoder_named_parameters()
    }


def encoder_state_is_unchanged(
    model: TransferMiniViT,
    before: dict[str, Tensor],
) -> bool:
    """Check bitwise equality for every encoder parameter."""

    current = dict(model.encoder_named_parameters())
    return set(current) == set(before) and all(
        torch.equal(before[name], current[name].detach())
        for name in before
    )
