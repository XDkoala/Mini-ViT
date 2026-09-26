"""Pure learning-rate schedule functions."""

import math

from torch.optim import Optimizer


def _validate_integer(value: object, name: str) -> int:
    """Require a real integer rather than silently accepting bool or float."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _validate_factor(value: object, name: str) -> float:
    """Require one finite numeric multiplier inside the unit interval."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")

    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in the range [0, 1].")
    return value


def compute_warmup_cosine_multiplier(
    update_index: int,
    total_steps: int,
    warmup_steps: int,
    start_factor: float,
    min_factor: float,
) -> float:
    """Return the LR multiplier used by one optimizer update.

    ``update_index`` is zero-based and identifies the successful optimizer
    update that will consume this multiplier. Warmup includes both its start
    and peak endpoints. When warmup is enabled, cosine excludes the duplicated
    peak and includes the final minimum endpoint.
    """

    update_index = _validate_integer(update_index, "update_index")
    total_steps = _validate_integer(total_steps, "total_steps")
    warmup_steps = _validate_integer(warmup_steps, "warmup_steps")
    start_factor = _validate_factor(start_factor, "start_factor")
    min_factor = _validate_factor(min_factor, "min_factor")

    if total_steps <= 0:
        raise ValueError("total_steps must be greater than zero.")
    if not 0 <= update_index < total_steps:
        raise ValueError(
            "update_index must be in the range [0, total_steps)."
        )
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "warmup_steps must be in the range [0, total_steps)."
        )
    # Warmup 的 W 个实际更新同时包含起点和峰值两个端点。
    if warmup_steps > 0 and update_index < warmup_steps:
        if warmup_steps == 1:
            return 1.0

        warmup_progress = update_index / (warmup_steps - 1)
        return start_factor + warmup_progress * (1.0 - start_factor)

    if warmup_steps == 0:
        # 单步训练无法同时采样峰值和最小值，优先让唯一更新使用峰值。
        if total_steps == 1:
            return 1.0
        cosine_progress = update_index / (total_steps - 1)
    else:
        # Warmup 最后一步已经使用峰值；cosine 从峰值后的第一个位置开始。
        cosine_steps = total_steps - warmup_steps
        cosine_index = update_index - warmup_steps
        cosine_progress = (cosine_index + 1) / cosine_steps

    cosine_weight = 0.5 * (
        1.0 + math.cos(math.pi * cosine_progress)
    )
    return min_factor + (1.0 - min_factor) * cosine_weight


class WarmupCosineScheduler:
    """Prepare one LR per successful optimizer update."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_steps: int,
        warmup_steps: int,
        start_factor: float,
        min_factor: float,
    ) -> None:
        if not optimizer.param_groups:
            raise ValueError("optimizer must contain at least one parameter group.")

        # 用纯函数验证调度参数，同时取得 update 0 的倍率。
        initial_multiplier = compute_warmup_cosine_multiplier(
            update_index=0,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            start_factor=start_factor,
            min_factor=min_factor,
        )

        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.start_factor = float(start_factor)
        self.min_factor = float(min_factor)
        self.base_lrs = [
            float(group["lr"])
            for group in optimizer.param_groups
        ]
        if any(
            not math.isfinite(lr) or lr < 0.0
            for lr in self.base_lrs
        ):
            raise ValueError(
                "optimizer learning rates must be finite and non-negative."
            )

        self.completed_updates = 0
        self._apply_multiplier(initial_multiplier)

    def _apply_multiplier(self, multiplier: float) -> None:
        """Apply one common multiplier while preserving group LR ratios."""

        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs,
            strict=True,
        ):
            group["lr"] = base_lr * multiplier

    def get_current_lrs(self) -> list[float]:
        """Return the learning rates prepared for the next optimizer update."""

        return [
            float(group["lr"])
            for group in self.optimizer.param_groups
        ]

    def step(self) -> None:
        """Record one successful update and prepare the following update LR."""

        if self.completed_updates >= self.total_steps:
            raise RuntimeError(
                "scheduler.step() was called after all planned updates."
            )

        self.completed_updates += 1
        if self.completed_updates == self.total_steps:
            # 最后一次更新已经使用最终倍率，不存在需要准备的下一步。
            return

        multiplier = compute_warmup_cosine_multiplier(
            update_index=self.completed_updates,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_factor=self.start_factor,
            min_factor=self.min_factor,
        )
        self._apply_multiplier(multiplier)

    def state_dict(self) -> dict[str, object]:
        """Return all state needed to resume the same schedule position."""

        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "start_factor": self.start_factor,
            "min_factor": self.min_factor,
            "base_lrs": list(self.base_lrs),
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore progress after validating the scheduler configuration."""

        required_keys = {
            "total_steps",
            "warmup_steps",
            "start_factor",
            "min_factor",
            "base_lrs",
            "completed_updates",
        }
        missing_keys = required_keys - state.keys()
        if missing_keys:
            names = ", ".join(sorted(missing_keys))
            raise ValueError(f"Scheduler state is missing keys: {names}")

        expected_configuration = {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "start_factor": self.start_factor,
            "min_factor": self.min_factor,
        }
        for name, expected_value in expected_configuration.items():
            if state[name] != expected_value:
                raise ValueError(
                    f"Scheduler state {name} does not match "
                    "the configured scheduler."
                )

        saved_base_lrs = state["base_lrs"]
        if (
            not isinstance(saved_base_lrs, list)
            or len(saved_base_lrs) != len(self.optimizer.param_groups)
            or [float(value) for value in saved_base_lrs] != self.base_lrs
        ):
            raise ValueError(
                "Scheduler state base_lrs do not match optimizer groups."
            )

        completed_updates = _validate_integer(
            state["completed_updates"],
            "completed_updates",
        )
        if not 0 <= completed_updates <= self.total_steps:
            raise ValueError(
                "completed_updates must be in the range [0, total_steps]."
            )

        self.completed_updates = completed_updates
        update_index = min(completed_updates, self.total_steps - 1)
        multiplier = compute_warmup_cosine_multiplier(
            update_index=update_index,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_factor=self.start_factor,
            min_factor=self.min_factor,
        )
        self._apply_multiplier(multiplier)
