"""Tests for pure learning-rate schedule mathematics."""

import pytest
import torch

from src.schedulers import (
    WarmupCosineScheduler,
    compute_warmup_cosine_multiplier,
)


def schedule_values(
    *,
    total_steps: int,
    warmup_steps: int,
    start_factor: float = 0.1,
    min_factor: float = 0.1,
) -> list[float]:
    """Evaluate every update in one finite schedule."""

    return [
        compute_warmup_cosine_multiplier(
            update_index=index,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            start_factor=start_factor,
            min_factor=min_factor,
        )
        for index in range(total_steps)
    ]


# 10 次更新、4 次 warmup 对应笔记中的完整数值示例。
def test_warmup_cosine_matches_defined_endpoint_contract():
    """Include warmup endpoints, avoid a duplicate peak, and end at minimum."""

    values = schedule_values(total_steps=10, warmup_steps=4)

    assert values == pytest.approx([
        0.1,
        0.4,
        0.7,
        1.0,
        0.9397114317,
        0.775,
        0.55,
        0.325,
        0.1602885683,
        0.1,
    ])
    assert values[:4] == sorted(values[:4])
    assert values[4:] == sorted(values[4:], reverse=True)
    assert values[4] < 1.0


# 无 warmup 时，cosine 自己必须同时包含峰值和最小值。
def test_cosine_without_warmup_includes_both_endpoints():
    """Start at the peak and finish at the minimum without warmup."""

    values = schedule_values(
        total_steps=5,
        warmup_steps=0,
        min_factor=0.2,
    )

    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.2)
    assert values == sorted(values, reverse=True)


# 单步边界优先使用峰值；一步 warmup 后的唯一 cosine 更新到达最小值。
def test_single_step_boundaries_have_explicit_meanings():
    """Handle schedules too short to sample two endpoints."""

    assert schedule_values(
        total_steps=1,
        warmup_steps=0,
    ) == pytest.approx([1.0])

    assert schedule_values(
        total_steps=2,
        warmup_steps=1,
        min_factor=0.2,
    ) == pytest.approx([1.0, 0.2])


# 所有合法位置都必须落在最小倍率和峰值倍率之间。
@pytest.mark.parametrize(
    ("total_steps", "warmup_steps"),
    [(2, 0), (2, 1), (7, 0), (7, 3)],
)
def test_schedule_never_leaves_configured_bounds(total_steps, warmup_steps):
    """Keep every multiplier inside the closed minimum-to-peak interval."""

    values = schedule_values(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        start_factor=0.15,
        min_factor=0.05,
    )

    assert all(0.05 <= value <= 1.0 for value in values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"update_index": True}, "update_index must be an integer"),
        ({"total_steps": 0}, "total_steps must be greater than zero"),
        ({"update_index": -1}, "update_index must be in the range"),
        ({"update_index": 5}, "update_index must be in the range"),
        ({"warmup_steps": -1}, "warmup_steps must be in the range"),
        ({"warmup_steps": 5}, "warmup_steps must be in the range"),
        ({"start_factor": 1.1}, "start_factor must be in the range"),
        ({"min_factor": -0.1}, "min_factor must be in the range"),
    ],
)
def test_schedule_rejects_ambiguous_or_invalid_inputs(kwargs, message):
    """Reject values that would make the finite schedule undefined."""

    arguments = {
        "update_index": 0,
        "total_steps": 5,
        "warmup_steps": 2,
        "start_factor": 0.1,
        "min_factor": 0.1,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        compute_warmup_cosine_multiplier(**arguments)


def make_optimizer_with_two_groups() -> torch.optim.Optimizer:
    """Create parameter groups whose peak learning rates differ by 10x."""

    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(2.0))
    return torch.optim.SGD([
        {"params": [first], "lr": 3e-4},
        {"params": [second], "lr": 3e-5},
    ])


# 构造函数应立即为 update 0 准备 LR，而不是等待第一次 step 后才设置。
def test_scheduler_initializes_update_zero_and_preserves_group_ratio():
    """Apply the initial multiplier to every optimizer parameter group."""

    optimizer = make_optimizer_with_two_groups()
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=10,
        warmup_steps=4,
        start_factor=0.1,
        min_factor=0.1,
    )

    assert scheduler.completed_updates == 0
    assert scheduler.base_lrs == pytest.approx([3e-4, 3e-5])
    assert scheduler.get_current_lrs() == pytest.approx([3e-5, 3e-6])
    assert scheduler.get_current_lrs()[0] / scheduler.get_current_lrs()[1] == (
        pytest.approx(10.0)
    )


# 每次 scheduler.step 都代表刚刚成功完成一次 optimizer update。
def test_scheduler_step_prepares_each_next_learning_rate_exactly_once():
    """Expose the pure schedule in the same order optimizer updates consume it."""

    optimizer = make_optimizer_with_two_groups()
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_steps=5,
        warmup_steps=2,
        start_factor=0.1,
        min_factor=0.2,
    )
    used_lrs = []

    for _ in range(5):
        used_lrs.append(scheduler.get_current_lrs()[0])
        optimizer.step()
        scheduler.step()

    expected_multipliers = schedule_values(
        total_steps=5,
        warmup_steps=2,
        start_factor=0.1,
        min_factor=0.2,
    )
    assert used_lrs == pytest.approx([
        3e-4 * multiplier
        for multiplier in expected_multipliers
    ])
    assert scheduler.completed_updates == 5

    with pytest.raises(RuntimeError, match="after all planned updates"):
        scheduler.step()


# state_dict 必须恢复“下一次更新该用什么 LR”，而不只是保存配置。
def test_scheduler_state_round_trip_restores_progress_and_current_lr():
    """Resume at the exact next-update position with matching group LRs."""

    source_optimizer = make_optimizer_with_two_groups()
    source = WarmupCosineScheduler(
        source_optimizer,
        total_steps=8,
        warmup_steps=3,
        start_factor=0.1,
        min_factor=0.05,
    )
    for _ in range(4):
        source_optimizer.step()
        source.step()

    target_optimizer = make_optimizer_with_two_groups()
    target = WarmupCosineScheduler(
        target_optimizer,
        total_steps=8,
        warmup_steps=3,
        start_factor=0.1,
        min_factor=0.05,
    )
    target.load_state_dict(source.state_dict())

    assert target.completed_updates == 4
    assert target.get_current_lrs() == pytest.approx(
        source.get_current_lrs()
    )

    source_optimizer.step()
    source.step()
    target_optimizer.step()
    target.step()
    assert target.get_current_lrs() == pytest.approx(
        source.get_current_lrs()
    )


# 用不同总步数构造 scheduler 后，不能静默套用不兼容的恢复状态。
def test_scheduler_rejects_mismatched_resume_configuration():
    """Reject state whose schedule identity differs from the current config."""

    source = WarmupCosineScheduler(
        make_optimizer_with_two_groups(),
        total_steps=8,
        warmup_steps=3,
        start_factor=0.1,
        min_factor=0.05,
    )
    target = WarmupCosineScheduler(
        make_optimizer_with_two_groups(),
        total_steps=9,
        warmup_steps=3,
        start_factor=0.1,
        min_factor=0.05,
    )

    with pytest.raises(ValueError, match="total_steps does not match"):
        target.load_state_dict(source.state_dict())
