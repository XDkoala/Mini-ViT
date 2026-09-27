"""Reusable static and runtime model-benchmarking utilities."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor, nn

from src.models.minivit import MultiHeadSelfAttention


MEBIBYTE = 1024**2


@dataclass(frozen=True)
class StaticModelProfile:
    """Describe device-independent model size and matrix computation."""

    parameter_count: int
    trainable_parameter_count: int
    parameter_bytes: int
    model_state_tensor_bytes: int
    serialized_state_dict_bytes: int
    macs_per_image: int
    flops_per_image: int

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe representation."""

        return asdict(self)


def tensor_tree_bytes(value: object) -> int:
    """Count Tensor payload bytes recursively, excluding Python overhead."""

    if isinstance(value, Tensor):
        return value.numel() * value.element_size()

    if isinstance(value, Mapping):
        return sum(tensor_tree_bytes(item) for item in value.values())

    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_bytes(item) for item in value)

    return 0


def percentile(values: Iterable[float], quantile: float) -> float:
    """Compute a linearly interpolated percentile without extra dependencies."""

    samples = sorted(float(value) for value in values)

    if not samples:
        raise ValueError("percentile requires at least one value.")

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in the range [0, 1].")

    position = quantile * (len(samples) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return samples[lower_index]

    fraction = position - lower_index
    return (
        samples[lower_index] * (1.0 - fraction)
        + samples[upper_index] * fraction
    )


def summarize_samples(values: Iterable[float]) -> dict[str, float]:
    """Summarize raw timing samples while preserving robust percentiles."""

    samples = [float(value) for value in values]

    if not samples:
        raise ValueError("summarize_samples requires at least one value.")

    return {
        "mean": sum(samples) / len(samples),
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "min": min(samples),
        "max": max(samples),
    }


def estimate_macs(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
) -> int:
    """Estimate Conv/Linear MACs plus the two attention matrix products.

    The count deliberately excludes normalization, activation, softmax,
    pooling, residual additions and memory-layout operations. Bias additions
    are also excluded. The returned value includes the supplied batch size.
    """

    if len(input_shape) != 4 or any(size <= 0 for size in input_shape):
        raise ValueError("input_shape must contain four positive dimensions.")

    total_macs = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def count_conv(
        module: nn.Conv2d,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal total_macs
        kernel_height, kernel_width = module.kernel_size
        kernel_macs = (
            module.in_channels
            // module.groups
            * kernel_height
            * kernel_width
        )
        total_macs += output.numel() * kernel_macs

    def count_linear(
        module: nn.Linear,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal total_macs
        total_macs += output.numel() * module.in_features

    def count_attention_products(
        _module: MultiHeadSelfAttention,
        inputs: tuple[Tensor, ...],
        _output: object,
    ) -> None:
        nonlocal total_macs
        tokens = inputs[0]
        batch_size, token_count, embed_dim = tokens.shape
        # QK^T 与 attention @ V 各需要 N^2 * D MACs。
        total_macs += 2 * batch_size * token_count**2 * embed_dim

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(count_conv))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(count_linear))
        elif isinstance(module, MultiHeadSelfAttention):
            handles.append(
                module.register_forward_hook(count_attention_products)
            )

    parameter = next(model.parameters(), None)
    device = (
        torch.device("cpu")
        if parameter is None
        else parameter.device
    )
    was_training = model.training

    try:
        model.eval()
        dummy = torch.zeros(input_shape, device=device)
        with torch.inference_mode():
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return total_macs


def serialized_state_dict_bytes(model: nn.Module) -> int:
    """Measure the actual torch.save size of model-only state."""

    with TemporaryDirectory() as directory:
        path = Path(directory) / "model_state.pt"
        torch.save(model.state_dict(), path)
        return path.stat().st_size


def build_static_profile(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
) -> StaticModelProfile:
    """Build the complete static profile under the project MAC convention."""

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    macs = estimate_macs(model, input_shape)

    return StaticModelProfile(
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        parameter_bytes=parameter_bytes,
        model_state_tensor_bytes=tensor_tree_bytes(model.state_dict()),
        serialized_state_dict_bytes=serialized_state_dict_bytes(model),
        macs_per_image=macs // input_shape[0],
        flops_per_image=2 * macs // input_shape[0],
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, amp: bool):
    if not amp:
        return nullcontext()

    if device.type != "cuda":
        raise ValueError("AMP benchmark requires a CUDA device.")

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    )


def measure_inference(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple[int, int, int],
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    amp: bool,
) -> dict[str, object]:
    """Measure pure-model inference latency and CUDA peak memory."""

    if min(batch_size, warmup_iterations, measured_iterations) <= 0:
        raise ValueError("batch size and iteration counts must be positive.")

    model.eval()
    inputs = torch.randn(
        (batch_size, *input_shape),
        device=device,
    )

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            with _autocast_context(device, amp):
                model(inputs)

        _synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        elapsed_seconds: list[float] = []

        for _ in range(measured_iterations):
            _synchronize(device)
            start = time.perf_counter()
            with _autocast_context(device, amp):
                model(inputs)
            _synchronize(device)
            elapsed_seconds.append(time.perf_counter() - start)

    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None
    )
    milliseconds = [value * 1000.0 for value in elapsed_seconds]
    images_per_second = [
        batch_size / value
        for value in elapsed_seconds
    ]

    return {
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "elapsed_ms_samples": milliseconds,
        "elapsed_ms": summarize_samples(milliseconds),
        "images_per_second_samples": images_per_second,
        "images_per_second": summarize_samples(images_per_second),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def measure_training_peak_memory(
    model: nn.Module,
    device: torch.device,
    input_shape: tuple[int, int, int],
    num_classes: int,
    batch_size: int,
    warmup_iterations: int,
    amp: bool,
) -> dict[str, int | bool | None]:
    """Measure peak memory of a synthetic forward/backward/AdamW step."""

    if min(batch_size, warmup_iterations) <= 0:
        raise ValueError("batch size and warmup iterations must be positive.")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if amp and device.type == "cuda"
        else None
    )
    inputs = torch.randn((batch_size, *input_shape), device=device)
    targets = torch.randint(
        low=0,
        high=num_classes,
        size=(batch_size,),
        device=device,
    )

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    for _ in range(warmup_iterations):
        step()

    _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step()
    _synchronize(device)

    return {
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "amp": amp,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None
        ),
    }
