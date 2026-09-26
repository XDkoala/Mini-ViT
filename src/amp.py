"""Device-aware automatic mixed-precision runtime policy."""

from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer


@dataclass(frozen=True)
class AmpRuntime:
    """Describe the requested and effective mixed-precision execution mode."""

    requested: bool
    enabled: bool
    device_type: str
    autocast_dtype: torch.dtype | None
    scaler: torch.amp.GradScaler | None
    fallback_reason: str | None

    def autocast(self) -> AbstractContextManager:
        """Return the autocast context matching the effective AMP policy."""

        return torch.autocast(
            device_type=self.device_type,
            dtype=self.autocast_dtype,
            enabled=self.enabled,
        )

    def to_metadata(self) -> dict[str, object]:
        """Return a YAML/JSON-safe description of the effective AMP mode."""

        return {
            "requested": self.requested,
            "enabled": self.enabled,
            "device_type": self.device_type,
            "dtype": (
                None
                if self.autocast_dtype is None
                else str(self.autocast_dtype).removeprefix("torch.")
            ),
            "uses_grad_scaler": self.scaler is not None,
            "fallback_reason": self.fallback_reason,
        }

    def backward(self, loss: Tensor) -> None:
        """Backpropagate through either the FP32 or scaled AMP path."""

        if self.enabled:
            if self.scaler is None:
                raise RuntimeError("Enabled AMP requires a GradScaler.")
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def unscale_gradients(self, optimizer: Optimizer) -> None:
        """Restore true gradient magnitudes before measurement or clipping."""

        if self.enabled:
            if self.scaler is None:
                raise RuntimeError("Enabled AMP requires a GradScaler.")
            self.scaler.unscale_(optimizer)

    def step_optimizer(self, optimizer: Optimizer) -> bool:
        """Step the optimizer and report whether GradScaler skipped the update."""

        if not self.enabled:
            optimizer.step()
            return True

        if self.scaler is None:
            raise RuntimeError("Enabled AMP requires a GradScaler.")

        scale_before = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()

        # 默认动态 scaler 仅在检测到 Inf/NaN 并跳过 optimizer.step 时
        # 使用 backoff_factor 降低 scale；成功时 scale 保持或增长。
        return scale_after >= scale_before


def build_amp_runtime(
    enabled: object,
    device: torch.device,
) -> AmpRuntime:
    """Build CUDA FP16 AMP or an explicit FP32 execution fallback."""

    if not isinstance(enabled, bool):
        raise ValueError("training.amp must be true or false.")

    if not enabled:
        return AmpRuntime(
            requested=False,
            enabled=False,
            device_type=device.type,
            autocast_dtype=None,
            scaler=None,
            fallback_reason=None,
        )

    if device.type != "cuda":
        return AmpRuntime(
            requested=True,
            enabled=False,
            device_type=device.type,
            autocast_dtype=None,
            scaler=None,
            fallback_reason=(
                "CUDA FP16 AMP was requested, but the resolved device is "
                f"{device.type}; training will use FP32."
            ),
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA AMP was requested for a CUDA device, but CUDA is unavailable."
        )

    return AmpRuntime(
        requested=True,
        enabled=True,
        device_type="cuda",
        autocast_dtype=torch.float16,
        scaler=torch.amp.GradScaler("cuda", enabled=True),
        fallback_reason=None,
    )
