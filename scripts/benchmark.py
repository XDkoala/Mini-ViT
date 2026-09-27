"""Run the Stage 14 static and CUDA runtime benchmark suite."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
from pathlib import Path

import torch
import yaml

from scripts.train import PROJECT_ROOT, build_model, resolve_project_path
from src.benchmarking import (
    MEBIBYTE,
    build_static_profile,
    measure_inference,
    measure_training_peak_memory,
)
from src.utils import collect_git_metadata, resolve_device


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "advanced"
    / "stage14"
    / "benchmark.yaml"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Benchmark frozen MiniViT and CNN candidates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    return parser.parse_args(argv)


def load_benchmark_config(path: Path) -> dict[str, object]:
    """Load and minimally validate the Stage 14 benchmark protocol."""

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Benchmark config root must be a mapping.")

    required = {
        "device",
        "input_shape",
        "precisions",
        "latency",
        "throughput",
        "training_memory",
        "candidates",
        "output",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(
            "Missing benchmark config keys: "
            + ", ".join(sorted(missing))
        )

    candidates = config["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list.")

    precisions = config["precisions"]
    if precisions != ["fp32", "amp"]:
        raise ValueError("precisions must be exactly [fp32, amp].")

    input_shape = config["input_shape"]
    if (
        not isinstance(input_shape, list)
        or len(input_shape) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in input_shape
        )
    ):
        raise ValueError("input_shape must contain three positive integers.")

    return config


def _load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        # 先把 checkpoint 留在 CPU；若直接 map 到 CUDA，checkpoint 中的
        # model_state 会与已构建模型各占一份显存，污染峰值测量。
        map_location="cpu",
        weights_only=True,
    )
    model = build_model(checkpoint["config"]["model"], device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _runtime_metadata(device: torch.device) -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        ),
        "mac_convention": "Conv2d + Linear + attention matmuls",
        "flop_convention": "1 MAC = 2 FLOPs",
        **collect_git_metadata(PROJECT_ROOT),
    }


def _flatten_summary(
    candidate: dict[str, object],
    precision: str,
    static: dict[str, int],
    latency: dict[str, object],
    throughput: dict[str, object],
    training_memory: dict[str, object],
    checkpoint_bytes: int,
    best_val_accuracy: float,
) -> dict[str, object]:
    latency_summary = latency["elapsed_ms"]
    throughput_summary = throughput["images_per_second"]

    return {
        "candidate": candidate["name"],
        "precision": precision,
        "best_val_accuracy": best_val_accuracy,
        "parameter_count": static["parameter_count"],
        "parameter_mib": static["parameter_bytes"] / MEBIBYTE,
        "model_state_tensor_mib": (
            static["model_state_tensor_bytes"] / MEBIBYTE
        ),
        "serialized_state_dict_mib": (
            static["serialized_state_dict_bytes"] / MEBIBYTE
        ),
        "training_checkpoint_mib": checkpoint_bytes / MEBIBYTE,
        "macs_per_image": static["macs_per_image"],
        "flops_per_image": static["flops_per_image"],
        "latency_batch_size": latency["batch_size"],
        "latency_p50_ms": latency_summary["p50"],
        "latency_p95_ms": latency_summary["p95"],
        "throughput_batch_size": throughput["batch_size"],
        "throughput_mean_images_per_second": throughput_summary["mean"],
        "throughput_p50_images_per_second": throughput_summary["p50"],
        "inference_peak_allocated_mib": (
            latency["peak_allocated_bytes"] / MEBIBYTE
        ),
        "inference_peak_reserved_mib": (
            latency["peak_reserved_bytes"] / MEBIBYTE
        ),
        "training_batch_size": training_memory["batch_size"],
        "training_peak_allocated_mib": (
            training_memory["peak_allocated_bytes"] / MEBIBYTE
        ),
        "training_peak_reserved_mib": (
            training_memory["peak_reserved_bytes"] / MEBIBYTE
        ),
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Run all frozen benchmark candidates and persist raw measurements."""

    args = parse_args(argv)
    config_path = (
        args.config
        if args.config.is_absolute()
        else PROJECT_ROOT / args.config
    )
    config = load_benchmark_config(config_path)
    device = resolve_device(str(config["device"]))

    if device.type != "cuda":
        raise RuntimeError(
            "Stage 14 formal benchmark requires CUDA for FP32/AMP comparison."
        )

    input_shape = tuple(int(value) for value in config["input_shape"])
    latency_config = config["latency"]
    throughput_config = config["throughput"]
    training_config = config["training_memory"]
    raw_results: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for candidate in config["candidates"]:
        checkpoint_path = resolve_project_path(candidate["checkpoint"])
        static_model, checkpoint = _load_checkpoint_model(
            checkpoint_path,
            torch.device("cpu"),
        )
        static_profile = build_static_profile(
            static_model,
            (1, *input_shape),
        ).to_dict()
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        del static_model

        for precision in config["precisions"]:
            amp = precision == "amp"
            model, _ = _load_checkpoint_model(checkpoint_path, device)
            latency = measure_inference(
                model=model,
                device=device,
                input_shape=input_shape,
                batch_size=int(latency_config["batch_size"]),
                warmup_iterations=int(latency_config["warmup_iterations"]),
                measured_iterations=int(
                    latency_config["measured_iterations"]
                ),
                amp=amp,
            )
            del model
            _release_cuda()

            model, _ = _load_checkpoint_model(checkpoint_path, device)
            throughput = measure_inference(
                model=model,
                device=device,
                input_shape=input_shape,
                batch_size=int(throughput_config["batch_size"]),
                warmup_iterations=int(
                    throughput_config["warmup_iterations"]
                ),
                measured_iterations=int(
                    throughput_config["measured_iterations"]
                ),
                amp=amp,
            )
            del model
            _release_cuda()

            model, checkpoint_for_training = _load_checkpoint_model(
                checkpoint_path,
                device,
            )
            num_classes = int(
                checkpoint_for_training["config"]["model"]["num_classes"]
            )
            training_memory = measure_training_peak_memory(
                model=model,
                device=device,
                input_shape=input_shape,
                num_classes=num_classes,
                batch_size=int(training_config["batch_size"]),
                warmup_iterations=int(training_config["warmup_iterations"]),
                amp=amp,
            )
            del model
            _release_cuda()

            record = {
                "candidate": candidate["name"],
                "checkpoint": str(candidate["checkpoint"]),
                "precision": precision,
                "best_val_accuracy": best_val_accuracy,
                "static": static_profile,
                "latency": latency,
                "throughput": throughput,
                "training_memory": training_memory,
            }
            raw_results.append(record)
            summary_rows.append(
                _flatten_summary(
                    candidate=candidate,
                    precision=precision,
                    static=static_profile,
                    latency=latency,
                    throughput=throughput,
                    training_memory=training_memory,
                    checkpoint_bytes=checkpoint_path.stat().st_size,
                    best_val_accuracy=best_val_accuracy,
                )
            )
            print(
                f"{candidate['name']} {precision}: "
                f"latency_p50={latency['elapsed_ms']['p50']:.3f}ms, "
                "throughput_mean="
                f"{throughput['images_per_second']['mean']:.1f} img/s"
            )

    output_config = config["output"]
    json_path = resolve_project_path(output_config["json"])
    csv_path = resolve_project_path(output_config["csv"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "protocol": config,
        "environment": _runtime_metadata(device),
        "results": raw_results,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json_path.relative_to(PROJECT_ROOT).as_posix())
    print(csv_path.relative_to(PROJECT_ROOT).as_posix())
    return payload


if __name__ == "__main__":
    main()
