"""Aggregate only complete, reproducible experiment runs into CSV tables."""

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path

from scripts.train import PROJECT_ROOT, resolve_project_path
from src.config import (
    load_config,
    resolve_format_version,
    resolve_random_seeds,
)
from src.utils import BASE_HISTORY_FIELDS


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs" / "advanced"
REQUIRED_ARTIFACTS = (
    "config.yaml",
    "history.csv",
    "best.pt",
    "last.pt",
)
SUMMARY_FIELDS = (
    "experiment_id",
    "run_dir",
    "format_version",
    "git_commit",
    "split_seed",
    "train_seed",
    "target_epochs",
    "best_epoch",
    "best_val_accuracy",
    "final_train_accuracy",
    "final_val_accuracy",
    "total_epoch_time_seconds",
)
SKIPPED_FIELDS = (
    "experiment_id",
    "run_dir",
    "reason",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse aggregation input and output paths."""

    parser = argparse.ArgumentParser(
        description="Aggregate completed, reproducible experiment runs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing experiment run folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Eligible-run CSV path (default: INPUT_ROOT/summary.csv).",
    )
    parser.add_argument(
        "--skipped-output",
        type=Path,
        default=None,
        help=(
            "Skipped-run CSV path "
            "(default: INPUT_ROOT/summary_skipped.csv)."
        ),
    )
    return parser.parse_args(argv)


def discover_run_directories(input_root: str | Path) -> list[Path]:
    """Find directories containing any recognized top-level run artifact."""

    root = Path(input_root)
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError("input root must be a directory.")

    artifact_names = {"run_status.json", *REQUIRED_ARTIFACTS}
    directories = {
        path.parent
        for path in root.rglob("*")
        if path.is_file() and path.name in artifact_names
    }
    return sorted(directories, key=lambda path: path.as_posix())


def _read_status(path: Path) -> dict[str, object]:
    """Read one status file and require a JSON object root."""

    with path.open(encoding="utf-8") as file:
        status = json.load(file)
    if not isinstance(status, dict):
        raise ValueError("run_status.json root must be an object.")
    return status


def _ineligibility_reason(status: Mapping[str, object]) -> str | None:
    """Recompute eligibility from source fields instead of trusting its flag."""

    resolve_format_version(status)

    if status.get("status") != "completed":
        return f"status is {status.get('status')!r}, not 'completed'"
    if status.get("is_smoke_test") is not False:
        return "run is a smoke test or has an invalid smoke flag"
    if not isinstance(status.get("git_commit"), str) or not status["git_commit"]:
        return "git_commit is missing"
    if status.get("git_is_dirty") is not False:
        return "Git worktree is dirty or its state is unknown"

    target_epochs = status.get("target_epochs")
    completed_epochs = status.get("completed_epochs")
    if (
        isinstance(target_epochs, bool)
        or not isinstance(target_epochs, int)
        or target_epochs <= 0
        or completed_epochs != target_epochs
    ):
        return "completed_epochs does not equal a valid target_epochs"

    if status.get("aggregation_eligible") is not True:
        return "aggregation_eligible flag is inconsistent with source fields"
    return None


def _read_history_summary(
    path: Path,
    target_epochs: int,
) -> dict[str, int | float]:
    """Validate a complete epoch sequence and return aggregate metrics."""

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or ())
        missing_fields = set(BASE_HISTORY_FIELDS) - fieldnames
        if missing_fields:
            names = ", ".join(sorted(missing_fields))
            raise ValueError(f"history is missing fields: {names}")
        rows = list(reader)

    if not rows:
        raise ValueError("history contains no epoch rows")

    try:
        epochs = [int(row["epoch"]) for row in rows]
        val_accuracies = [float(row["val_accuracy"]) for row in rows]
        train_accuracies = [float(row["train_accuracy"]) for row in rows]
        epoch_times = [float(row["epoch_time_seconds"]) for row in rows]
    except (TypeError, ValueError) as error:
        raise ValueError("history contains a non-numeric required value") from error

    expected_epochs = list(range(1, target_epochs + 1))
    if epochs != expected_epochs:
        raise ValueError(
            "history epochs are not the complete sequence "
            f"1..{target_epochs}"
        )

    best_index = max(
        range(len(rows)),
        key=val_accuracies.__getitem__,
    )
    return {
        "best_epoch": epochs[best_index],
        "best_val_accuracy": val_accuracies[best_index],
        "final_train_accuracy": train_accuracies[-1],
        "final_val_accuracy": val_accuracies[-1],
        "total_epoch_time_seconds": sum(epoch_times),
    }


def summarize_run(
    run_dir: str | Path,
    input_root: str | Path,
) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    """Return either one eligible summary or one explicit skip record."""

    directory = Path(run_dir)
    root = Path(input_root)
    relative_dir = directory.relative_to(root).as_posix()
    status_path = directory / "run_status.json"
    experiment_id = directory.name

    def skipped(reason: str) -> tuple[None, dict[str, str]]:
        return None, {
            "experiment_id": experiment_id,
            "run_dir": relative_dir,
            "reason": reason,
        }

    if not status_path.is_file():
        return skipped("missing run_status.json")

    try:
        status = _read_status(status_path)
        experiment_id = str(status.get("experiment_id", experiment_id))
        reason = _ineligibility_reason(status)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return skipped(f"invalid run status: {error}")

    if reason is not None:
        return skipped(reason)

    missing_artifacts = [
        name
        for name in REQUIRED_ARTIFACTS
        if not (directory / name).is_file()
    ]
    if missing_artifacts:
        return skipped(
            "missing artifacts: " + ", ".join(missing_artifacts)
        )

    try:
        config = load_config(directory / "config.yaml")
        config_experiment_id = str(config["experiment"]["name"])
        if config_experiment_id != experiment_id:
            raise ValueError(
                "experiment_id does not match config experiment.name"
            )
        split_seed, train_seed = resolve_random_seeds(config)
        target_epochs = int(status["target_epochs"])
        history_summary = _read_history_summary(
            directory / "history.csv",
            target_epochs,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        return skipped(f"invalid run artifacts: {error}")

    summary: dict[str, object] = {
        "experiment_id": experiment_id,
        "run_dir": relative_dir,
        "format_version": resolve_format_version(status),
        "git_commit": status["git_commit"],
        "split_seed": split_seed,
        "train_seed": train_seed,
        "target_epochs": target_epochs,
        **history_summary,
    }
    return summary, None


def aggregate_runs(
    input_root: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Aggregate all discovered runs into eligible and skipped records."""

    summaries: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for run_dir in discover_run_directories(input_root):
        summary, skip_record = summarize_run(run_dir, input_root)
        if summary is not None:
            summaries.append(summary)
        if skip_record is not None:
            skipped.append(skip_record)
    return summaries, skipped


def write_csv(
    records: list[dict[str, object]] | list[dict[str, str]],
    path: str | Path,
    fieldnames: tuple[str, ...],
) -> None:
    """Atomically write a CSV table, including its header when empty."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary_path.replace(output_path)


def main(
    argv: list[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Run strict experiment aggregation and persist both decisions."""

    args = parse_args(argv)
    input_root = resolve_project_path(args.input_root)
    output_path = (
        resolve_project_path(args.output)
        if args.output is not None
        else input_root / "summary.csv"
    )
    skipped_path = (
        resolve_project_path(args.skipped_output)
        if args.skipped_output is not None
        else input_root / "summary_skipped.csv"
    )

    summaries, skipped = aggregate_runs(input_root)
    write_csv(summaries, output_path, SUMMARY_FIELDS)
    write_csv(skipped, skipped_path, SKIPPED_FIELDS)
    print(
        f"Aggregation finished: eligible={len(summaries)}, "
        f"skipped={len(skipped)}"
    )
    return summaries, skipped


if __name__ == "__main__":
    main()
