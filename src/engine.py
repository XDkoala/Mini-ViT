"""Reusable training and evaluation loops."""

from contextlib import nullcontext
import math
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.amp import AmpRuntime
from src.gradients import (
    clip_gradient_norm,
    compute_gradient_norm,
    validate_gradient_clip_norm,
)
from src.metrics import ClassificationMetrics
from src.schedulers import WarmupCosineScheduler
from src.utils import save_checkpoint, save_history_csv

# Stage 5

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    scheduler: WarmupCosineScheduler | None = None,
    gradient_clip_norm: float | None = None,
    amp_runtime: AmpRuntime | None = None,
) -> dict[str, float]:
    """Train a model for one epoch and return aggregated metrics."""

    model.train()   # 切换为训练模式
    metrics = ClassificationMetrics()
    last_used_learning_rate: float | None = None
    gradient_norm_sum = 0.0
    gradient_norm_count = 0
    gradient_clip_count = 0
    gradient_clip_norm = validate_gradient_clip_norm(
        gradient_clip_norm
    )

    if scheduler is not None and scheduler.optimizer is not optimizer:
        raise ValueError(
            "scheduler must manage the optimizer passed to train_one_epoch."
        )

    if (
        amp_runtime is not None
        and amp_runtime.device_type != device.type
    ):
        raise ValueError(
            "amp_runtime device_type must match the training device."
        )

    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            "max_batches must be greater than zero."
        )

    # 遍历 batches
    for batch_index, (images, labels) in enumerate(loader):
        # 模型、输入和标签必须位于同一个计算设备。
        images = images.to(device)
        labels = labels.to(device)

        # 清除上一个 step 留下的梯度，避免梯度意外累加。
        optimizer.zero_grad(set_to_none=True)

        # autocast 只包住前向和 loss；关闭 AMP 时使用空 context，
        # 保持已有 FP32 运算路径不变。
        autocast_context = (
            nullcontext()
            if amp_runtime is None
            else amp_runtime.autocast()
        )
        with autocast_context:
            logits = model(images)
            loss = criterion(logits, labels)

        # AMP 会先生成 scaled gradients，再恢复为真实尺度；FP32
        # runtime 与未提供 runtime 时都执行普通 backward。
        if amp_runtime is None:
            loss.backward()
        else:
            amp_runtime.backward(loss)
            amp_runtime.unscale_gradients(optimizer)

        # 此时梯度已经生成、参数尚未更新，因此测得的是本次
        # optimizer update 真正准备使用的原始梯度规模。
        if amp_runtime is not None and amp_runtime.enabled:
            batch_gradient_norm = compute_gradient_norm(
                model.parameters(),
                error_if_nonfinite=False,
            )
            gradient_is_finite = math.isfinite(batch_gradient_norm)

            # 非有限 AMP 梯度必须保留给 GradScaler 检测并跳过更新；
            # 不能先进行裁剪，把 Inf 进一步污染为 NaN。
            if gradient_is_finite and gradient_clip_norm is not None:
                _, was_clipped = clip_gradient_norm(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )
            else:
                was_clipped = False
        else:
            gradient_is_finite = True
            if gradient_clip_norm is None:
                batch_gradient_norm = compute_gradient_norm(
                    model.parameters()
                )
                was_clipped = False
            else:
                batch_gradient_norm, was_clipped = clip_gradient_norm(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )

        # 在 step 前读取，记录本 batch 真正使用的 LR；step 后的
        # scheduler 会把 optimizer.lr 改成下一个 batch 的值。
        candidate_learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )
        update_succeeded = (
            amp_runtime.step_optimizer(optimizer)
            if amp_runtime is not None
            else _step_optimizer(optimizer)
        )

        if update_succeeded:
            if not gradient_is_finite:
                raise RuntimeError(
                    "GradScaler reported a successful update for "
                    "non-finite gradients."
                )

            last_used_learning_rate = candidate_learning_rate
            gradient_norm_sum += batch_gradient_norm
            gradient_norm_count += 1
            gradient_clip_count += int(was_clipped)

            if scheduler is not None:
                scheduler.step()

        # 累计本 batch 的 loss、正确数和样本数。
        metrics.update(
            loss=loss,
            logits=logits,
            labels=labels,
        )

        # 在当前 batch 完整处理后停止，避免预先读取额外 batch。
        if (
            max_batches is not None
            and batch_index + 1 >= max_batches
        ):
            break

    aggregate_metrics = metrics.compute()
    if last_used_learning_rate is None:
        raise RuntimeError(
            "training produced no successful optimizer updates."
        )

    result = {
        **aggregate_metrics,
        "learning_rate": last_used_learning_rate,
        # 按 optimizer update 次数平均；每个 batch 对应一次更新事件。
        "gradient_norm": gradient_norm_sum / gradient_norm_count,
    }
    if gradient_clip_norm is not None:
        result["gradient_clip_fraction"] = (
            gradient_clip_count / gradient_norm_count
        )

    return result


def _step_optimizer(optimizer: Optimizer) -> bool:
    """Run one ordinary FP32 optimizer update with a shared success contract."""

    optimizer.step()
    return True


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate a model without computing gradients or updating parameters."""

    model.eval()
    metrics = ClassificationMetrics()

    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            "max_batches must be greater than zero."
        )

    for batch_index, (images, labels) in enumerate(loader):
        # 验证数据仍需与模型位于同一个设备。
        images = images.to(device)
        labels = labels.to(device)

        # 只执行前向传播和 loss 计算，不建立反向传播计算图。
        logits = model(images)
        loss = criterion(logits, labels)

        metrics.update(
            loss=loss,
            logits=logits,
            labels=labels,
        )

        # 在当前 batch 完整处理后停止，避免预先读取额外 batch。
        if (
            max_batches is not None
            and batch_index + 1 >= max_batches
        ):
            break

    return metrics.compute()


@torch.no_grad()
def evaluate_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[
    dict[str, int | float],
    list[dict[str, int | float | bool]],
]:
    """Evaluate a classifier and retain one prediction record per sample."""

    model.eval()
    metrics = ClassificationMetrics()
    prediction_records: list[
        dict[str, int | float | bool]
    ] = []

    if max_batches is not None and max_batches <= 0:
        raise ValueError(
            "max_batches must be greater than zero."
        )

    sample_index = 0

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # 模型仍输出原始 logits；CrossEntropyLoss 不接收 Softmax 概率。
        logits = model(images)
        loss = criterion(logits, labels)

        metrics.update(
            loss=loss,
            logits=logits,
            labels=labels,
        )

        # Softmax 仅用于解释预测置信度，不参与 loss 或参数更新。
        probabilities = logits.softmax(dim=1)
        confidences, predicted_labels = probabilities.max(dim=1)

        # 每个张量只做一次 GPU→CPU 传输，避免逐样本同步设备。
        true_labels_cpu = labels.detach().cpu().tolist()
        predicted_labels_cpu = (
            predicted_labels.detach().cpu().tolist()
        )
        confidences_cpu = confidences.detach().cpu().tolist()

        for true_label, predicted_label, confidence in zip(
            true_labels_cpu,
            predicted_labels_cpu,
            confidences_cpu,
            strict=True,
        ):
            prediction_records.append({
                "sample_index": sample_index,
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "confidence": float(confidence),
                "correct": bool(predicted_label == true_label),
            })
            sample_index += 1

        # 与训练/验证循环保持相同语义：处理完当前 batch 后再停止，
        # 不为判断上限而预先读取额外 batch。
        if (
            max_batches is not None
            and batch_index + 1 >= max_batches
        ):
            break

    aggregate_metrics = metrics.compute()

    return (
        {
            "loss": aggregate_metrics["loss"],
            "accuracy": aggregate_metrics["accuracy"],
            "num_samples": len(prediction_records),
        },
        prediction_records,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    output_dir: str | Path | None = None,
    config: dict[str, object] | None = None,
    start_epoch: int = 1,
    initial_history: list[dict[str, int | float]] | None = None,
    best_val_accuracy: float = float("-inf"),
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    verbose: bool = False,
    scheduler: WarmupCosineScheduler | None = None,
    gradient_clip_norm: float | None = None,
    amp_runtime: AmpRuntime | None = None,
) -> list[dict[str, int | float]]:
    """Train and validate a model for multiple epochs."""

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if start_epoch <= 0:
        raise ValueError("start_epoch must be greater than zero.")

    if start_epoch > epochs:
        raise ValueError(
            "start_epoch must not be greater than epochs."
        )

    if (
        max_train_batches is not None
        and max_train_batches <= 0
    ):
        raise ValueError(
            "max_train_batches must be greater than zero."
        )

    if (
        max_val_batches is not None
        and max_val_batches <= 0
    ):
        raise ValueError(
            "max_val_batches must be greater than zero."
        )

    history = (
        []
        if initial_history is None
        else list(initial_history)
    )

    expected_previous_epoch = start_epoch - 1

    if history:
        if history[-1]["epoch"] != expected_previous_epoch:
            raise ValueError(
                "The last history epoch must equal start_epoch - 1."
            )
    elif start_epoch != 1:
        raise ValueError(
            "Resumed training requires existing history."
        )

    checkpoint_config = {} if config is None else config

    for epoch in range(start_epoch, epochs + 1):
        # perf_counter 只衡量经过时间，不受系统时钟校准影响。
        epoch_start_time = perf_counter()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
            scheduler=scheduler,
            gradient_clip_norm=gradient_clip_norm,
            amp_runtime=amp_runtime,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_val_batches,
        )

        # 记录训练与验证的计算耗时，不把磁盘写入波动算进去。
        epoch_time_seconds = perf_counter() - epoch_start_time

        current_val_accuracy = val_metrics["accuracy"]
        is_best = current_val_accuracy > best_val_accuracy

        if is_best:
            best_val_accuracy = current_val_accuracy

        epoch_record: dict[str, int | float] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_time_seconds": epoch_time_seconds,
            "learning_rate": train_metrics["learning_rate"],
            "gradient_norm": train_metrics["gradient_norm"],
        }
        if "gradient_clip_fraction" in train_metrics:
            epoch_record["gradient_clip_fraction"] = train_metrics[
                "gradient_clip_fraction"
            ]

        history.append(epoch_record)

        if output_dir is not None:
            experiment_dir = Path(output_dir)

            save_history_csv(
                history=history,
                path=experiment_dir / "history.csv",
            )

            # last.pt 每轮覆盖，用于从最近进度继续训练。
            save_checkpoint(
                path=experiment_dir / "last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                best_val_accuracy=best_val_accuracy,
                history=history,
                config=checkpoint_config,
                scheduler=scheduler,
                scaler=(
                    None
                    if amp_runtime is None
                    else amp_runtime.scaler
                ),
            )

            # best.pt 只在验证准确率严格刷新时覆盖。
            if is_best:
                save_checkpoint(
                    path=experiment_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_val_accuracy=best_val_accuracy,
                    history=history,
                    config=checkpoint_config,
                    scheduler=scheduler,
                    scaler=(
                        None
                        if amp_runtime is None
                        else amp_runtime.scaler
                    ),
                )

        if verbose:
            best_marker = "yes" if is_best else "no"
            clip_summary = (
                ""
                if "gradient_clip_fraction" not in train_metrics
                else (
                    "clip_fraction="
                    f"{train_metrics['gradient_clip_fraction']:.2%} | "
                )
            )
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.2%} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.2%} | "
                f"lr={train_metrics['learning_rate']:.6g} | "
                f"grad_norm={train_metrics['gradient_norm']:.4f} | "
                f"{clip_summary}"
                f"time={epoch_time_seconds:.1f}s | "
                f"best={best_marker}",
                flush=True,
            )

    return history
