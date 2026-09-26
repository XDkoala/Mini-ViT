"""Tests for strict advanced-experiment aggregation."""

import csv
import json
from pathlib import Path

import pytest

import scripts.aggregate as aggregate_script
from src.config import save_config
from src.utils import save_history_csv, write_run_status


def create_run(
    root: Path,
    name: str,
    *,
    status: str = "completed",
    is_smoke_test: bool = False,
    git_commit: str | None = "abc123",
    git_is_dirty: bool | None = False,
    completed_epochs: int | None = 2,
) -> Path:
    """Create one lightweight run directory with realistic text artifacts."""

    run_dir = root / name
    config = {
        "format_version": 2,
        "experiment": {
            "name": name,
            "output_dir": str(run_dir),
        },
        "randomness": {
            "split_seed": 42,
            "train_seed": 123,
        },
        "device": "cpu",
        "data": {},
        "model": {},
        "training": {},
    }
    save_config(config, run_dir / "config.yaml")
    save_history_csv(
        [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "train_accuracy": 0.5,
                "val_loss": 0.9,
                "val_accuracy": 0.6,
                "epoch_time_seconds": 2.0,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "train_accuracy": 0.7,
                "val_loss": 0.7,
                "val_accuracy": 0.65,
                "epoch_time_seconds": 3.0,
            },
        ],
        run_dir / "history.csv",
    )
    (run_dir / "best.pt").touch()
    (run_dir / "last.pt").touch()
    write_run_status(
        run_dir / "run_status.json",
        status=status,
        experiment_id=name,
        is_smoke_test=is_smoke_test,
        git_commit=git_commit,
        git_is_dirty=git_is_dirty,
        target_epochs=2,
        completed_epochs=completed_epochs,
        error_type="RuntimeError" if status == "failed" else None,
        error_message="boom" if status == "failed" else None,
    )
    return run_dir


# 合格运行必须从状态、配置和完整 history 中提取结构化指标。
def test_aggregate_runs_summarizes_only_valid_completed_run(tmp_path: Path):
    """Summarize a clean completed run with reproducible seeds and metrics."""

    create_run(tmp_path, "valid_run")
    summaries, skipped = aggregate_script.aggregate_runs(tmp_path)

    assert skipped == []
    assert summaries == [{
        "experiment_id": "valid_run",
        "run_dir": "valid_run",
        "format_version": 2,
        "git_commit": "abc123",
        "split_seed": 42,
        "train_seed": 123,
        "target_epochs": 2,
        "best_epoch": 2,
        "best_val_accuracy": pytest.approx(0.65),
        "final_train_accuracy": pytest.approx(0.7),
        "final_val_accuracy": pytest.approx(0.65),
        "total_epoch_time_seconds": pytest.approx(5.0),
    }]


# 聚合器必须重新验证原始字段，并为每个排除决定给出原因。
@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"status": "running", "completed_epochs": None}, "not 'completed'"),
        ({"is_smoke_test": True}, "smoke test"),
        ({"git_is_dirty": True}, "worktree is dirty"),
        ({"git_commit": None}, "git_commit is missing"),
    ],
)
def test_aggregate_runs_reports_ineligible_reason(
    tmp_path: Path,
    kwargs,
    reason,
):
    """Exclude non-formal runs with a human-readable reason."""

    create_run(tmp_path, "rejected_run", **kwargs)
    summaries, skipped = aggregate_script.aggregate_runs(tmp_path)

    assert summaries == []
    assert len(skipped) == 1
    assert reason in skipped[0]["reason"]


# 即使状态合格，缺少 checkpoint 或 history 不完整也不能进入汇总。
def test_aggregate_runs_rejects_missing_or_incomplete_artifacts(tmp_path: Path):
    """Reject a formally completed run whose persisted artifacts are incomplete."""

    run_dir = create_run(tmp_path, "broken_run")
    (run_dir / "best.pt").unlink()

    summaries, skipped = aggregate_script.aggregate_runs(tmp_path)
    assert summaries == []
    assert "missing artifacts: best.pt" in skipped[0]["reason"]


# CLI 同时持久化合格表和跳过表，空表也必须保留表头。
def test_main_writes_summary_and_skipped_csv_files(tmp_path: Path):
    """Persist both aggregation outputs for auditability."""

    create_run(tmp_path, "valid_run")
    create_run(
        tmp_path,
        "smoke_run",
        is_smoke_test=True,
    )
    summary_path = tmp_path / "reports" / "summary.csv"
    skipped_path = tmp_path / "reports" / "skipped.csv"

    summaries, skipped = aggregate_script.main([
        "--input-root",
        str(tmp_path),
        "--output",
        str(summary_path),
        "--skipped-output",
        str(skipped_path),
    ])

    assert len(summaries) == 1
    assert len(skipped) == 1
    with summary_path.open(newline="", encoding="utf-8") as file:
        summary_rows = list(csv.DictReader(file))
    with skipped_path.open(newline="", encoding="utf-8") as file:
        skipped_rows = list(csv.DictReader(file))

    assert summary_rows[0]["experiment_id"] == "valid_run"
    assert skipped_rows[0]["experiment_id"] == "smoke_run"
    assert not summary_path.with_name("summary.csv.tmp").exists()
    assert not skipped_path.with_name("skipped.csv.tmp").exists()
