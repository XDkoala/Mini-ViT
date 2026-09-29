"""Train Stage 16 scratch, linear-probe, or fine-tune experiments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_scheduler  # noqa: E402
from src.amp import build_amp_runtime  # noqa: E402
from src.config import CURRENT_FORMAT_VERSION, save_config  # noqa: E402
from src.core50 import (  # noqa: E402
    build_core50_dataloaders,
    sha256_file,
)
from src.engine import fit  # noqa: E402
from src.gradients import validate_gradient_clip_norm  # noqa: E402
from src.regularization import build_classification_criteria  # noqa: E402
from src.transfer import (  # noqa: E402
    build_transfer_model,
    clone_encoder_state,
    encoder_state_is_unchanged,
    trainable_parameters,
)
from src.utils import (  # noqa: E402
    collect_run_metadata,
    load_checkpoint,
    resolve_device,
    set_seed,
    write_run_status,
)


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "advanced"
    / "stage16"
    / "scratch.yaml"
)

_REQUIRED_TOP_LEVEL = {
    "format_version",
    "experiment",
    "randomness",
    "device",
    "data",
    "model",
    "transfer",
    "training",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse formal and smoke/resume overrides."""

    parser = argparse.ArgumentParser(
        description="Train one frozen Stage 16 transfer setting."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args(argv)


def load_transfer_config(path: Path) -> dict[str, object]:
    """Load the Stage 16 schema, where manifest membership replaces split_seed."""

    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Transfer config root must be a mapping.")
    missing = _REQUIRED_TOP_LEVEL - config.keys()
    if missing:
        raise ValueError(
            "Missing transfer config keys: " + ", ".join(sorted(missing))
        )
    if config["format_version"] != CURRENT_FORMAT_VERSION:
        raise ValueError(
            f"Transfer config format_version must be {CURRENT_FORMAT_VERSION}."
        )
    for section in (
        "experiment",
        "randomness",
        "data",
        "model",
        "transfer",
        "training",
    ):
        if not isinstance(config[section], dict):
            raise ValueError(f"Transfer config section {section} must be a mapping.")

    randomness = config["randomness"]
    if set(randomness) != {"train_seed"}:
        raise ValueError(
            "Transfer randomness must contain only train_seed; "
            "split membership comes from the frozen manifest."
        )
    train_seed = randomness["train_seed"]
    if (
        isinstance(train_seed, bool)
        or not isinstance(train_seed, int)
        or not 0 <= train_seed < 2**32
    ):
        raise ValueError("randomness.train_seed must be in [0, 2**32).")

    required_data = {
        "dataset_root",
        "manifest",
        "manifest_sha256",
        "batch_size",
        "num_workers",
        "augmentation",
    }
    missing_data = required_data - config["data"].keys()
    if missing_data:
        raise ValueError(
            "Missing transfer data keys: " + ", ".join(sorted(missing_data))
        )
    required_transfer = {"mode", "source_checkpoint", "source_sha256"}
    missing_transfer = required_transfer - config["transfer"].keys()
    if missing_transfer:
        raise ValueError(
            "Missing transfer keys: " + ", ".join(sorted(missing_transfer))
        )
    return config


def resolve_project_path(value: object) -> Path:
    """Resolve a configured path against the repository root."""

    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_transfer_optimizer(
    model: torch.nn.Module,
    training_config: Mapping[str, object],
) -> torch.optim.Optimizer:
    """Build AdamW from only the parameters permitted by transfer mode."""

    if str(training_config["optimizer"]).strip().lower() != "adamw":
        raise ValueError("transfer training optimizer must be adamw.")
    return torch.optim.AdamW(
        trainable_parameters(model),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )


def _check_expected_hash(path: Path, expected: object, label: str) -> str:
    """Reject provenance drift before training starts."""

    actual = sha256_file(path)
    if actual.lower() != str(expected).strip().lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, found {actual}."
        )
    return actual


def main(argv: list[str] | None = None) -> list[dict[str, int | float]]:
    """Run one transfer experiment without ever evaluating CORe50 test."""

    args = parse_args(argv)
    config = load_transfer_config(args.config)
    experiment_config = config["experiment"]
    randomness = config["randomness"]
    data_config = config["data"]
    model_config = config["model"]
    transfer_config = config["transfer"]
    training_config = config["training"]

    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be greater than zero.")
        training_config["epochs"] = args.epochs
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be greater than zero.")
    if args.max_batches is not None and args.output_dir is None:
        raise ValueError(
            "--max-batches requires --output-dir to protect formal outputs."
        )

    train_seed = int(randomness["train_seed"])
    set_seed(train_seed)
    device = resolve_device(str(config["device"]))
    amp_runtime = build_amp_runtime(training_config.get("amp", False), device)

    dataset_root = resolve_project_path(data_config["dataset_root"])
    manifest_path = resolve_project_path(data_config["manifest"])
    manifest_sha256 = _check_expected_hash(
        manifest_path,
        data_config["manifest_sha256"],
        "CORe50 manifest",
    )

    mode = str(transfer_config["mode"])
    source_value = transfer_config["source_checkpoint"]
    source_checkpoint = (
        None
        if source_value is None
        else resolve_project_path(source_value)
    )
    if mode == "scratch":
        if transfer_config["source_sha256"] is not None:
            raise ValueError("Scratch source_sha256 must be null.")
    else:
        if source_checkpoint is None:
            raise ValueError(f"{mode} requires source_checkpoint.")
        _check_expected_hash(
            source_checkpoint,
            transfer_config["source_sha256"],
            "Source checkpoint",
        )

    if args.output_dir is not None:
        experiment_config["output_dir"] = str(args.output_dir)
    output_dir = resolve_project_path(experiment_config["output_dir"])
    is_smoke_test = args.max_batches is not None
    run_type = "smoke" if is_smoke_test else "formal"
    target_epochs = int(training_config["epochs"])

    run_metadata = collect_run_metadata(
        project_root=PROJECT_ROOT,
        device=device,
        split_seed=None,
        train_seed=train_seed,
        is_smoke_test=is_smoke_test,
    )
    config["runtime"] = {
        "is_smoke_test": is_smoke_test,
        "run_type": run_type,
        "max_train_batches": args.max_batches,
        "max_val_batches": args.max_batches,
        "manifest_sha256": manifest_sha256,
        "amp": amp_runtime.to_metadata(),
        "metadata": run_metadata,
    }
    status_arguments = {
        "path": output_dir / "run_status.json",
        "experiment_id": str(experiment_config["name"]),
        "is_smoke_test": is_smoke_test,
        "run_type": run_type,
        "git_commit": run_metadata.get("git_commit"),
        "git_is_dirty": run_metadata.get("git_is_dirty"),
        "target_epochs": target_epochs,
    }
    write_run_status(**status_arguments, status="running")

    try:
        train_loader, validation_loader, _, _ = build_core50_dataloaders(
            dataset_root,
            manifest_path,
            batch_size=int(data_config["batch_size"]),
            num_workers=int(data_config["num_workers"]),
            train_seed=train_seed,
            augmentation=str(data_config["augmentation"]),
        )
        model, initialization = build_transfer_model(
            model_config,
            mode=mode,
            device=device,
            source_checkpoint=source_checkpoint,
        )
        config["runtime"]["transfer_initialization"] = asdict(initialization)
        frozen_encoder_reference = (
            clone_encoder_state(model)
            if mode == "linear_probe"
            else None
        )

        optimizer = build_transfer_optimizer(model, training_config)
        scheduler = build_scheduler(
            optimizer,
            training_config,
            train_loader,
            args.max_batches,
        )
        train_criterion, validation_criterion = (
            build_classification_criteria(
                float(training_config.get("label_smoothing", 0.0))
            )
        )
        gradient_clip_norm = validate_gradient_clip_norm(
            training_config.get("gradient_clip_norm")
        )

        start_epoch = 1
        initial_history = None
        best_val_accuracy = float("-inf")
        if args.resume is not None:
            resume_state = load_checkpoint(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
                scaler=amp_runtime.scaler,
            )
            if (
                frozen_encoder_reference is not None
                and not encoder_state_is_unchanged(
                    model,
                    frozen_encoder_reference,
                )
            ):
                raise ValueError(
                    "Linear-probe resume checkpoint changed source encoder."
                )
            start_epoch = int(resume_state["epoch"]) + 1
            initial_history = list(resume_state["history"])
            best_val_accuracy = float(resume_state["best_val_accuracy"])

        save_config(config, output_dir / "config.yaml")
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=validation_loader,
            train_criterion=train_criterion,
            val_criterion=validation_criterion,
            optimizer=optimizer,
            device=device,
            epochs=target_epochs,
            output_dir=output_dir,
            config=config,
            start_epoch=start_epoch,
            initial_history=initial_history,
            best_val_accuracy=best_val_accuracy,
            max_train_batches=args.max_batches,
            max_val_batches=args.max_batches,
            verbose=True,
            scheduler=scheduler,
            gradient_clip_norm=gradient_clip_norm,
            amp_runtime=amp_runtime,
            batch_mixer=None,
        )
        if (
            frozen_encoder_reference is not None
            and not encoder_state_is_unchanged(model, frozen_encoder_reference)
        ):
            raise RuntimeError(
                "Linear-probe encoder changed during target training."
            )
    except BaseException as error:
        try:
            write_run_status(
                **status_arguments,
                status="failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
        except Exception as status_error:
            error.add_note(
                "Additionally failed to persist run status: "
                f"{status_error}"
            )
        raise

    write_run_status(
        **status_arguments,
        status="completed",
        completed_epochs=int(history[-1]["epoch"]),
    )
    return history


if __name__ == "__main__":
    main()
