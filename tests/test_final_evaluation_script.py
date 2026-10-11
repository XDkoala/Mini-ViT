"""Check frozen seed completeness and honest aggregation. / 核验冻结种子及统计口径。"""

import math
import statistics

import pytest

import scripts.final_evaluation as final_script
from scripts.final_evaluation import reuse_seed42, summarize_accuracy


def test_statistics_use_all_seeds_and_sample_std():
    # Include every seed and use n-1. / 三个种子都保留，标准差分母为 n-1。
    values = [0.86, 0.85, 0.87]
    rows = [dict(train_seed=seed, test_accuracy=value)
            for seed, value in zip([42, 123, 2026], values)]
    result = summarize_accuracy(rows)
    assert result["mean_accuracy"] == pytest.approx(statistics.mean(values))
    assert result["sample_std_accuracy"] == pytest.approx(statistics.stdev(values))
    assert result["std_ddof"] == 1


@pytest.mark.parametrize("seeds", [[42, 123], [42, 123, 123], [2026, 123, 42]])
def test_reject_missing_duplicate_or_reordered_seeds(seeds):
    # Do not permit cherry-picking. / 不允许遗漏、重复或替换冻结种子。
    with pytest.raises(ValueError):
        summarize_accuracy([dict(train_seed=seed, test_accuracy=0.8) for seed in seeds])


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.1])
def test_reject_invalid_accuracy(value):
    # Invalid metrics must not become a final mean. / 非法指标不得进入最终均值。
    with pytest.raises(ValueError):
        summarize_accuracy([dict(train_seed=seed, test_accuracy=value) for seed in [42, 123, 2026]])


def make_raw():
    return {"conditions": [dict(split="test", model="advanced_minivit_final", corruption="clean",
                                severity=0, sample_count=10000, accuracy=0.8638, loss=0.43)],
            "metadata": {"git_commit": "frozen_commit"},
            "protocol": {"evaluation": {"model_batch_size": 256}}}


def test_reuse_original_test_not_validation():
    # Ignore validation and retain original identity. / 排除验证集记录，保留原始评估身份。
    raw = make_raw()
    raw["conditions"].append({**raw["conditions"][0], "split": "validation", "accuracy": 0.99})
    row = reuse_seed42(raw, 10000)
    assert row["test_accuracy"] == 0.8638
    assert row["evaluation_commit"] == "frozen_commit"
    assert row["reused"] is True


@pytest.mark.parametrize("mode", ["duplicate", "partial", "missing"])
def test_reuse_rejects_ambiguous_or_incomplete_evidence(mode):
    # Ambiguous or partial evidence fails closed. / 重复、缺失或不完整证据均拒绝。
    raw = make_raw()
    if mode == "duplicate":
        raw["conditions"] *= 2
    elif mode == "partial":
        raw["conditions"][0]["sample_count"] = 128
    else:
        raw["conditions"] = []
    with pytest.raises(ValueError):
        reuse_seed42(raw, 10000)


def test_dirty_worktree_is_rejected_before_evaluation(monkeypatch):
    # Dirty code has no exact commit identity. / 未提交代码不得冒充某个提交的正式结果。
    monkeypatch.setattr(final_script, "collect_git_metadata",
                        lambda _: {"git_commit": "abc", "git_is_dirty": True})
    monkeypatch.setattr(final_script.evaluate, "main",
                        lambda _: pytest.fail("Test images must not be read."))
    with pytest.raises(ValueError, match="clean, committed"):
        final_script.main()


def test_changed_evaluator_is_rejected_before_evaluation(monkeypatch):
    # Keep the evaluator frozen. / 评估入口源码变化时，在读取图片前拒绝运行。
    monkeypatch.setattr(final_script, "collect_git_metadata",
                        lambda _: {"git_commit": "abc", "git_is_dirty": False})
    monkeypatch.setattr(final_script.subprocess, "check_output", lambda *a, **k: "different_blob")
    monkeypatch.setattr(final_script.evaluate, "main",
                        lambda _: pytest.fail("Test images must not be read."))
    with pytest.raises(ValueError, match="source has changed"):
        final_script.main()
