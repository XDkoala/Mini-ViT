"""Complete the frozen three-seed clean-test evaluation without overwriting evidence."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts import evaluate
from src.config import resolve_random_seeds
from src.utils import collect_git_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs/advanced/stage18/final_evaluation.json"


def summarize_accuracy(rows: list[dict]) -> dict:
    """Report sample standard deviation, not a confidence interval. / 报告样本标准差。"""
    if [row["train_seed"] for row in rows] != [42, 123, 2026]:
        raise ValueError("All three frozen seeds are required, in protocol order.")
    values = [float(row["test_accuracy"]) for row in rows]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Accuracy must be finite and within [0, 1].")
    return {
        "mean_accuracy": statistics.mean(values),
        "sample_std_accuracy": statistics.stdev(values),
        "seed_count": len(values),
        "std_ddof": 1,
        "interpretation": "Training-seed variability on one fixed test set; not a confidence interval.",
    }


def reuse_seed42(raw: dict, sample_count: int) -> dict:
    """Select exactly one original clean-test row. / 只复用已冻结的完整测试记录。"""
    rows = [row for row in raw["conditions"] if
            row["split"] == "test" and row["model"] == "advanced_minivit_final"
            and row["corruption"] == "clean" and row["severity"] == 0]
    if len(rows) != 1 or rows[0]["sample_count"] != sample_count:
        raise ValueError("Expected one complete frozen seed-42 clean-test result.")
    return {
        "train_seed": 42,
        "test_accuracy": rows[0]["accuracy"],
        "test_loss": rows[0]["loss"],
        "num_samples": rows[0]["sample_count"],
        "evaluation_commit": raw["metadata"]["git_commit"],
        "reused": True,
        "evaluation_batch_size": raw["protocol"]["evaluation"]["model_batch_size"],
    }


def main() -> dict:
    """Preflight every identity before reading test images. / 先全部核验，再接触测试图片。"""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    metadata = collect_git_metadata(PROJECT_ROOT)
    if metadata["git_is_dirty"] is not False or not metadata["git_commit"]:
        raise ValueError("Final evaluation requires a clean, committed worktree.")
    blob = subprocess.check_output(
        ["git", "hash-object", "scripts/evaluate.py"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if blob != protocol["evaluator_blob"]:
        raise ValueError("Frozen evaluator source has changed.")
    if [item["train_seed"] for item in protocol["checkpoints"]] != [42, 123, 2026]:
        raise ValueError("Frozen seed list has changed.")
    if [item["reuse"] for item in protocol["checkpoints"]] != [True, False, False]:
        raise ValueError("Only seed 42 may reuse an earlier test result.")
    reuse_path = PROJECT_ROOT / protocol["reuse_path"]
    if evaluate.compute_sha256(reuse_path) != protocol["reuse_sha256"]:
        raise ValueError("Frozen Stage 15 evidence has changed.")
    rows = [reuse_seed42(json.loads(reuse_path.read_text(encoding="utf-8")), protocol["sample_count"])]
    checkpoints = {}
    for item in protocol["checkpoints"]:
        path = PROJECT_ROOT / item["path"]
        if evaluate.compute_sha256(path) != item["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: seed {item['train_seed']}")
        checkpoint = evaluate.load_evaluation_checkpoint(path, torch.device("cpu"))
        if resolve_random_seeds(checkpoint["config"]) != (protocol["split_seed"], item["train_seed"]):
            raise ValueError("Checkpoint seeds disagree with the frozen protocol.")
        checkpoints[item["train_seed"]] = checkpoint["config"]
    output = PROJECT_ROOT / protocol["output_dir"]
    summary_path = PROJECT_ROOT / protocol["summary_path"]
    if output.exists() or summary_path.exists():
        raise FileExistsError("Final evidence already exists; refusing to overwrite or silently rerun.")

    # Reserve the whole run directory. / 即使中途失败也保留目录，防止自动覆盖后重挑结果。
    output.mkdir(parents=True, exist_ok=False)
    rows[0].update({"checkpoint_path": protocol["checkpoints"][0]["path"],
                    "checkpoint_sha256": protocol["checkpoints"][0]["sha256"],
                    "evidence_path": protocol["reuse_path"],
                    "evidence_sha256": protocol["reuse_sha256"]})
    for item in protocol["checkpoints"][1:]:
        run_path = output / f"train_{item['train_seed']}"
        metrics = evaluate.main(["--checkpoint", item["path"], "--output-dir", str(run_path)])
        if metrics["is_partial_evaluation"] or metrics["num_samples"] != protocol["sample_count"]:
            raise ValueError("A partial test result cannot enter the final table.")
        if metrics["checkpoint_sha256"] != item["sha256"]:
            raise ValueError("Checkpoint changed during evaluation.")
        rows.append({**metrics, "train_seed": item["train_seed"], "reused": False,
                     "evaluation_commit": metadata["git_commit"],
                     "evaluation_batch_size": checkpoints[item["train_seed"]]["data"]["batch_size"],
                     "evidence_path": (run_path / "metrics.json").relative_to(PROJECT_ROOT).as_posix(),
                     "evidence_sha256": evaluate.compute_sha256(run_path / "metrics.json"),
                     "predictions_sha256": evaluate.compute_sha256(run_path / "predictions.csv")})
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "protocol_path": PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "protocol_sha256": evaluate.compute_sha256(PROTOCOL_PATH),
        "evaluator_blob": blob,
        "rows": rows,
        "statistics": summarize_accuracy(rows),
        "protocol_note": "Seed 42 reuses Stage 15 FP32 batch 256; new seeds use frozen checkpoint FP32 batch 128. No test-time augmentation or model reselection.",
    }
    evaluate.save_metrics_json(result, summary_path)
    stats = result["statistics"]
    print(f"Final clean test: {stats['mean_accuracy']:.3%} +/- {stats['sample_std_accuracy']:.3%} (sample std, n=3)")
    return result


if __name__ == "__main__":
    main()
