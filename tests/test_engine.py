"""Tests for reusable training and evaluation utilities."""

import csv
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import src.engine as engine_module
from src.engine import evaluate, fit, train_one_epoch
from src.metrics import ClassificationMetrics
from src.utils import load_checkpoint


class EvaluationProbe(nn.Module):
    """Record mode and gradient state observed during each forward call."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 2, bias=False)
        self.training_states: list[bool] = []
        self.gradient_states: list[bool] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.training_states.append(self.training)
        self.gradient_states.append(torch.is_grad_enabled())
        return self.classifier(inputs)


class CountingBatchLoader:
    """Yield fixed batches while recording how many were actually requested."""

    def __init__(self, batches) -> None:
        self.batches = batches
        self.yield_count = 0

    def __iter__(self):
        for batch in self.batches:
            self.yield_count += 1
            yield batch


# 不同大小的 batch 必须按照样本数加权，不能简单平均 batch 指标。
def test_classification_metrics_weights_batches_by_sample_count():
    """Compute epoch metrics correctly across unequal batch sizes."""

    metrics = ClassificationMetrics()

    # Batch 1：2 个样本，平均 loss=0.5，预测正确 1 个。
    batch_1_loss = torch.tensor(0.5)
    batch_1_logits = torch.tensor([
        [2.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ])
    batch_1_labels = torch.tensor([0, 1])

    metrics.update(
        loss=batch_1_loss,
        logits=batch_1_logits,
        labels=batch_1_labels,
    )

    # Batch 2：1 个样本，平均 loss=2.0，预测正确 1 个。
    batch_2_loss = torch.tensor(2.0)
    batch_2_logits = torch.tensor([
        [0.0, 3.0, 1.0],
    ])
    batch_2_labels = torch.tensor([1])

    metrics.update(
        loss=batch_2_loss,
        logits=batch_2_logits,
        labels=batch_2_labels,
    )

    result = metrics.compute()

    # 加权 loss：(0.5 × 2 + 2.0 × 1) / 3 = 1.0
    assert result["loss"] == pytest.approx(1.0)

    # 正确数：(1 + 1) / 3 = 2/3
    assert result["accuracy"] == pytest.approx(2 / 3)


# 没有加入任何样本时，compute() 应主动报告使用错误。
def test_classification_metrics_rejects_empty_compute():
    """Reject metric computation when no samples were accumulated."""

    metrics = ClassificationMetrics()

    with pytest.raises(
        ValueError,
        match="before any samples",
    ):
        metrics.compute()


# reset() 后的结果只能来自新 batch，不能混入之前的数据。
def test_classification_metrics_reset_clears_previous_state():
    """Discard all previously accumulated metric statistics."""

    metrics = ClassificationMetrics()

    # reset 前加入一个 loss=0.5、预测正确的样本。
    metrics.update(
        loss=torch.tensor(0.5),
        logits=torch.tensor([[3.0, 1.0]]),
        labels=torch.tensor([0]),
    )

    metrics.reset()

    # reset 后加入一个 loss=2.0、预测错误的样本。
    metrics.update(
        loss=torch.tensor(2.0),
        logits=torch.tensor([[3.0, 1.0]]),
        labels=torch.tensor([1]),
    )

    result = metrics.compute()

    assert result["loss"] == pytest.approx(2.0)
    assert result["accuracy"] == pytest.approx(0.0)


# 用可手算的两批数据验证训练模式、参数更新和整轮指标。
def test_train_one_epoch_updates_model_and_returns_weighted_metrics():
    """Train over two batches and return exact sample-weighted metrics."""

    device = torch.device("cpu")

    # 只有两个权重的线性分类器，初始 logits 全为 0。
    model = nn.Linear(
        in_features=1,
        out_features=2,
        bias=False,
    ).to(device)
    with torch.no_grad():
        model.weight.zero_()

    # 先切到验证模式，确认 train_one_epoch() 会主动切回训练模式。
    model.eval()

    dataset = TensorDataset(
        torch.ones(3, 1),
        torch.tensor([0, 0, 1]),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )

    weight_before = model.weight.detach().clone()

    result = train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # 第一批 loss=ln(2)；更新后第二批 loss=ln(1+e^0.1)。
    expected_loss = (
        2 * math.log(2)
        + math.log1p(math.exp(0.1))
    ) / 3

    assert model.training
    assert not torch.equal(weight_before, model.weight)
    assert result["loss"] == pytest.approx(expected_loss)
    assert result["accuracy"] == pytest.approx(2 / 3)


# 验证过程必须关闭训练行为和梯度记录，且不能修改任何参数。
def test_evaluate_disables_gradients_and_preserves_parameters():
    """Evaluate in inference mode without creating gradients or updates."""

    device = torch.device("cpu")
    model = EvaluationProbe().to(device)

    with torch.no_grad():
        model.classifier.weight.zero_()

    # 从训练模式开始，确认 evaluate() 会主动切换模式。
    model.train()
    weight_before = model.classifier.weight.detach().clone()

    dataset = TensorDataset(
        torch.ones(3, 1),
        torch.tensor([0, 1, 0]),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    result = evaluate(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
    )

    # 两个 batch 的 forward 都必须发生在 eval + no_grad 环境中。
    assert model.training_states == [False, False]
    assert model.gradient_states == [False, False]
    assert not model.training

    # 验证没有 backward/step，权重和梯度都不应改变。
    assert torch.equal(weight_before, model.classifier.weight)
    assert model.classifier.weight.grad is None

    # 全零 logits 的交叉熵为 ln(2)，argmax 预测类别恒为 0。
    assert result["loss"] == pytest.approx(math.log(2))
    assert result["accuracy"] == pytest.approx(2 / 3)


# batch 上限应同时限制数据读取次数和实际训练 forward 次数。
def test_train_one_epoch_respects_max_batches_without_prefetching_extra():
    """Train on exactly one requested batch and do not consume the next one."""

    device = torch.device("cpu")
    model = EvaluationProbe().to(device)
    loader = CountingBatchLoader([
        (torch.ones(2, 1), torch.tensor([0, 0])),
        (torch.ones(2, 1), torch.tensor([1, 1])),
    ])

    train_one_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=device,
        max_batches=1,
    )

    assert loader.yield_count == 1
    assert len(model.training_states) == 1


# 验证上限与训练上限应具有完全相同的读取语义。
def test_evaluate_respects_max_batches_without_prefetching_extra():
    """Evaluate exactly one requested batch and do not consume the next one."""

    device = torch.device("cpu")
    model = EvaluationProbe().to(device)
    loader = CountingBatchLoader([
        (torch.ones(2, 1), torch.tensor([0, 0])),
        (torch.ones(2, 1), torch.tensor([1, 1])),
    ])

    evaluate(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        max_batches=1,
    )

    assert loader.yield_count == 1
    assert len(model.training_states) == 1


@pytest.mark.parametrize("max_batches", [0, -1])
def test_epoch_loops_reject_non_positive_max_batches(max_batches):
    """Reject invalid batch limits before attempting to iterate the loader."""

    device = torch.device("cpu")
    model = nn.Linear(1, 2)
    loader = CountingBatchLoader([
        (torch.ones(1, 1), torch.tensor([0])),
    ])

    with pytest.raises(
        ValueError,
        match="max_batches must be greater than zero",
    ):
        train_one_epoch(
            model=model,
            loader=loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=device,
            max_batches=max_batches,
        )

    assert loader.yield_count == 0

    with pytest.raises(
        ValueError,
        match="max_batches must be greater than zero",
    ):
        evaluate(
            model=model,
            loader=loader,
            criterion=nn.CrossEntropyLoss(),
            device=device,
            max_batches=max_batches,
        )

    assert loader.yield_count == 0


# 两个 epoch 都应重新读取各自限定数量的训练与验证 batch。
def test_fit_applies_separate_batch_limits_on_every_epoch():
    """Limit train and validation iteration independently on each epoch."""

    device = torch.device("cpu")
    model = EvaluationProbe().to(device)
    train_loader = CountingBatchLoader([
        (torch.ones(2, 1), torch.tensor([0, 0])),
        (torch.ones(2, 1), torch.tensor([1, 1])),
    ])
    val_loader = CountingBatchLoader([
        (torch.ones(2, 1), torch.tensor([0, 1])),
        (torch.ones(2, 1), torch.tensor([1, 0])),
    ])

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=device,
        epochs=2,
        max_train_batches=1,
        max_val_batches=1,
    )

    assert len(history) == 2
    assert train_loader.yield_count == 2
    assert val_loader.yield_count == 2
    assert model.training_states == [True, False, True, False]


@pytest.mark.parametrize(
    ("max_train_batches", "max_val_batches", "error_message"),
    [
        (0, 1, "max_train_batches must be greater than zero"),
        (1, 0, "max_val_batches must be greater than zero"),
    ],
)
def test_fit_rejects_invalid_limits_before_any_parameter_update(
    max_train_batches,
    max_val_batches,
    error_message,
):
    """Validate all batch limits before reading data or changing parameters."""

    device = torch.device("cpu")
    model = nn.Linear(1, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = CountingBatchLoader([
        (torch.ones(1, 1), torch.tensor([0])),
    ])
    weight_before = model.weight.detach().clone()
    bias_before = model.bias.detach().clone()

    with pytest.raises(ValueError, match=error_message):
        fit(
            model=model,
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            device=device,
            epochs=1,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        )

    assert loader.yield_count == 0
    assert torch.equal(weight_before, model.weight)
    assert torch.equal(bias_before, model.bias)


# 两轮 fit 必须严格按“训练→验证→训练→验证”的顺序运行并记录结果。
def test_fit_runs_each_epoch_and_builds_history(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    """Run two train-validation cycles and build one record per epoch."""

    device = torch.device("cpu")
    model = EvaluationProbe().to(device)

    train_loader = DataLoader(
        TensorDataset(
            torch.ones(2, 1),
            torch.tensor([0, 0]),
        ),
        batch_size=2,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.ones(2, 1),
            torch.tensor([0, 1]),
        ),
        batch_size=2,
        shuffle=False,
    )

    output_dir = tmp_path / "experiment"
    config: dict[str, object] = {
        "seed": 42,
        "model": "evaluation_probe",
    }

    # 用确定的起止时刻验证耗时记录，避免测试依赖真实机器速度。
    clock_values = iter((100.0, 112.3, 200.0, 211.8))
    monkeypatch.setattr(
        engine_module,
        "perf_counter",
        lambda: next(clock_values),
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=device,
        epochs=2,
        output_dir=output_dir,
        config=config,
        verbose=True,
    )

    # 每轮各有一个训练 batch 和验证 batch。
    assert model.training_states == [True, False, True, False]
    assert model.gradient_states == [True, False, True, False]

    assert len(history) == 2
    assert [record["epoch"] for record in history] == [1, 2]

    expected_keys = {
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_time_seconds",
    }
    assert all(set(record) == expected_keys for record in history)
    assert history[0]["epoch_time_seconds"] == pytest.approx(12.3)
    assert history[1]["epoch_time_seconds"] == pytest.approx(11.8)

    # verbose 模式应逐轮报告核心指标、耗时及最佳模型状态。
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert len(output_lines) == 2
    assert "Epoch [001/002]" in output_lines[0]
    assert "train_loss=" in output_lines[0]
    assert "train_acc=" in output_lines[0]
    assert "val_loss=" in output_lines[0]
    assert "val_acc=" in output_lines[0]
    assert "time=12.3s" in output_lines[0]
    assert "best=yes" in output_lines[0]
    assert "Epoch [002/002]" in output_lines[1]
    assert "time=11.8s" in output_lines[1]
    assert "best=no" in output_lines[1]

    # fit 在每轮验证后结束，因此返回时模型保持验证模式。
    assert not model.training

    # 磁盘 CSV 应包含与内存 history 对应的两轮记录。
    history_path = output_dir / "history.csv"
    with history_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:
        saved_rows = list(csv.DictReader(file))

    assert len(saved_rows) == len(history)

    for saved_row, history_record in zip(
        saved_rows,
        history,
        strict=True,
    ):
        assert int(saved_row["epoch"]) == history_record["epoch"]

        for field in (
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "epoch_time_seconds",
        ):
            assert float(saved_row[field]) == pytest.approx(
                history_record[field]
            )

    last_checkpoint = torch.load(
        output_dir / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    best_checkpoint = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    # 验证准确率两轮均为 50%，last 更新到第 2 轮，best 保留第 1 轮。
    assert history[0]["val_accuracy"] == pytest.approx(0.5)
    assert history[1]["val_accuracy"] == pytest.approx(0.5)
    assert last_checkpoint["epoch"] == 2
    assert best_checkpoint["epoch"] == 1

    assert last_checkpoint["best_val_accuracy"] == pytest.approx(0.5)
    assert best_checkpoint["best_val_accuracy"] == pytest.approx(0.5)
    assert last_checkpoint["history"] == history
    assert best_checkpoint["history"] == history[:1]
    assert last_checkpoint["config"] == config
    assert best_checkpoint["config"] == config

    # 两轮训练后的 last 权重应不同于第一轮保存的 best 权重。
    assert any(
        not torch.equal(
            last_checkpoint["model_state"][name],
            best_checkpoint["model_state"][name],
        )
        for name in last_checkpoint["model_state"]
    )


@pytest.mark.parametrize("epochs", [0, -1])
def test_fit_rejects_non_positive_epochs(epochs):
    """Reject zero or negative training durations before iteration starts."""

    device = torch.device("cpu")
    model = nn.Linear(1, 2)
    loader = DataLoader(
        TensorDataset(
            torch.ones(1, 1),
            torch.tensor([0]),
        ),
        batch_size=1,
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        fit(
            model=model,
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=device,
            epochs=epochs,
        )


# 模拟进程中断：重建模型与优化器后，从 last.pt 接续到总计第 3 轮。
def test_fit_resumes_from_last_checkpoint_without_losing_history(
    tmp_path: Path,
):
    """Resume a saved run with continuous epoch numbers and history."""

    device = torch.device("cpu")
    output_dir = tmp_path / "resumed_experiment"
    config: dict[str, object] = {
        "seed": 42,
        "epochs": 3,
    }
    train_loader = DataLoader(
        TensorDataset(
            torch.ones(2, 1),
            torch.tensor([0, 0]),
        ),
        batch_size=2,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.ones(2, 1),
            torch.tensor([0, 1]),
        ),
        batch_size=2,
        shuffle=False,
    )

    # 第一次进程只完成 epoch 1。
    source_model = EvaluationProbe().to(device)
    source_optimizer = torch.optim.AdamW(
        source_model.parameters(),
        lr=1e-3,
    )
    first_history = fit(
        model=source_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=source_optimizer,
        device=device,
        epochs=1,
        output_dir=output_dir,
        config=config,
    )

    # 模拟重启：创建全新对象，再加载最近 checkpoint。
    resumed_model = EvaluationProbe().to(device)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=1e-3,
    )
    resume_state = load_checkpoint(
        path=output_dir / "last.pt",
        model=resumed_model,
        optimizer=resumed_optimizer,
        device=device,
    )

    resumed_history = fit(
        model=resumed_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=resumed_optimizer,
        device=device,
        epochs=3,
        output_dir=output_dir,
        config=resume_state["config"],
        start_epoch=resume_state["epoch"] + 1,
        initial_history=resume_state["history"],
        best_val_accuracy=resume_state["best_val_accuracy"],
    )

    assert [record["epoch"] for record in resumed_history] == [1, 2, 3]

    # fit 使用外层 list 的副本，不修改调用者传入的旧 history。
    assert [record["epoch"] for record in first_history] == [1]
    assert [
        record["epoch"]
        for record in resume_state["history"]
    ] == [1]

    with (output_dir / "history.csv").open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:
        saved_rows = list(csv.DictReader(file))

    assert [int(row["epoch"]) for row in saved_rows] == [1, 2, 3]

    last_checkpoint = torch.load(
        output_dir / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    best_checkpoint = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=True,
    )

    assert last_checkpoint["epoch"] == 3
    assert last_checkpoint["history"] == resumed_history

    # 验证准确率持续持平，恢复后不能覆盖中断前的最佳模型。
    assert best_checkpoint["epoch"] == 1


@pytest.mark.parametrize(
    (
        "epochs",
        "start_epoch",
        "initial_history",
        "error_message",
    ),
    [
        (3, 0, None, "start_epoch must be greater than zero"),
        (2, 3, None, "must not be greater than epochs"),
        (3, 2, None, "requires existing history"),
        (
            3,
            3,
            [{"epoch": 1}],
            "last history epoch must equal start_epoch - 1",
        ),
    ],
)
def test_fit_rejects_inconsistent_resume_state(
    epochs,
    start_epoch,
    initial_history,
    error_message,
):
    """Reject resume metadata that would create missing or duplicate epochs."""

    device = torch.device("cpu")
    model = nn.Linear(1, 2)
    loader = DataLoader(
        TensorDataset(
            torch.ones(1, 1),
            torch.tensor([0]),
        ),
        batch_size=1,
    )

    with pytest.raises(ValueError, match=error_message):
        fit(
            model=model,
            train_loader=loader,
            val_loader=loader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            device=device,
            epochs=epochs,
            start_epoch=start_epoch,
            initial_history=initial_history,
        )
