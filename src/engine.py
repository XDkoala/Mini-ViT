"""Reusable training and evaluation loops."""

from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.metrics import ClassificationMetrics
from src.utils import save_checkpoint, save_history_csv

# Stage 5

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Train a model for one epoch and return aggregated metrics."""

    model.train()   # 切换为训练模式
    metrics = ClassificationMetrics()

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

        # 前向传播：由当前参数计算原始分类 logits。
        logits = model(images)
        loss = criterion(logits, labels)

        # 反向传播只负责计算梯度，step 才真正修改参数。
        loss.backward()
        optimizer.step()

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

    return metrics.compute()


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
        }

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
                )

        if verbose:
            best_marker = "yes" if is_best else "no"
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.2%} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.2%} | "
                f"time={epoch_time_seconds:.1f}s | "
                f"best={best_marker}",
                flush=True,
            )

    return history
