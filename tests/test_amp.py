"""Tests for automatic mixed-precision runtime policy."""

import pytest
import torch

from src.amp import AmpRuntime, build_amp_runtime


class ScalerProbe:
    """Record scaler calls and optionally simulate an overflow skip."""

    def __init__(self, *, skip_update: bool) -> None:
        self.skip_update = skip_update
        self.scale_value = 8.0
        self.events: list[str] = []

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        self.events.append("step")
        if not self.skip_update:
            optimizer.step()

    def update(self):
        self.events.append("update")
        if self.skip_update:
            self.scale_value /= 2.0


# AMP 关闭时必须维持原有 FP32 路径，且不创建无意义的 scaler。
def test_build_amp_runtime_preserves_disabled_fp32_path():
    """Represent disabled AMP without autocasting or gradient scaling."""

    runtime = build_amp_runtime(False, torch.device("cpu"))
    values = torch.ones(2, 2)

    with runtime.autocast():
        result = values @ values

    assert runtime.requested is False
    assert runtime.enabled is False
    assert runtime.scaler is None
    assert runtime.fallback_reason is None
    assert result.dtype == torch.float32


# 当前项目把 amp:true 定义为 CUDA FP16；CPU 请求必须显式记录回退。
def test_build_amp_runtime_records_cpu_fallback():
    """Fall back to FP32 while preserving that AMP was requested."""

    runtime = build_amp_runtime(True, torch.device("cpu"))

    assert runtime.requested is True
    assert runtime.enabled is False
    assert runtime.autocast_dtype is None
    assert runtime.scaler is None
    assert "resolved device is cpu" in runtime.fallback_reason
    assert runtime.to_metadata() == {
        "requested": True,
        "enabled": False,
        "device_type": "cpu",
        "dtype": None,
        "uses_grad_scaler": False,
        "fallback_reason": runtime.fallback_reason,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to construct the enabled AMP runtime.",
)
def test_build_amp_runtime_enables_cuda_fp16_and_scaler():
    """Pair CUDA FP16 autocast with an enabled CUDA GradScaler."""

    runtime = build_amp_runtime(True, torch.device("cuda"))

    assert runtime.requested is True
    assert runtime.enabled is True
    assert runtime.device_type == "cuda"
    assert runtime.autocast_dtype == torch.float16
    assert runtime.scaler is not None
    assert runtime.scaler.is_enabled()
    assert runtime.to_metadata() == {
        "requested": True,
        "enabled": True,
        "device_type": "cuda",
        "dtype": "float16",
        "uses_grad_scaler": True,
        "fallback_reason": None,
    }


@pytest.mark.parametrize("value", [None, 0, 1, "true", {}])
def test_build_amp_runtime_rejects_non_boolean_configuration(value):
    """Reject values YAML users could mistake for an AMP boolean."""

    with pytest.raises(ValueError, match="true or false"):
        build_amp_runtime(value, torch.device("cpu"))


def test_build_amp_runtime_rejects_unavailable_cuda(monkeypatch):
    """Reject an internally inconsistent CUDA device request."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        build_amp_runtime(True, torch.device("cuda"))


@pytest.mark.parametrize(
    ("skip_update", "expected_success"),
    [(False, True), (True, False)],
)
def test_amp_runtime_reports_whether_scaler_updated_parameters(
    skip_update,
    expected_success,
):
    """Infer an overflow skip from the dynamic scale decrease."""

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([1.0])
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler = ScalerProbe(skip_update=skip_update)
    runtime = AmpRuntime(
        requested=True,
        enabled=True,
        device_type="cuda",
        autocast_dtype=torch.float16,
        scaler=scaler,
        fallback_reason=None,
    )
    before = parameter.detach().clone()

    succeeded = runtime.step_optimizer(optimizer)

    assert succeeded is expected_success
    assert scaler.events == ["step", "update"]
    assert torch.equal(parameter, before) is skip_update
