"""Generate reproducible Stage 17 validation-only attention comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as vision_functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.predict import METHOD_LABELS, project_path  # noqa: E402
from src.core50 import load_manifest, sha256_file  # noqa: E402
from src.corruptions import apply_corruption, validate_corruption  # noqa: E402
from src.data import CIFAR10_MEAN, CIFAR10_STD, CIFAR10_VAL_SIZE, split_train_val_indices  # noqa: E402
from src.inference import InferenceRuntime, load_inference_runtime, predict_batch  # noqa: E402
from src.utils import collect_git_metadata  # noqa: E402
from src.visualization import plot_attention_comparison  # noqa: E402


def validation_samples(runtime: InferenceRuntime, limit: int) -> list[dict[str, object]]:
    """Read only the registered validation split, with stable source IDs."""

    data_config = runtime.config["data"]
    samples = []
    if runtime.dataset == "cifar10":
        # CIFAR's official training source contains our frozen validation split.
        # 只打开官方 train 文件，再取项目原先固定的 validation 索引。
        source = CIFAR10(project_path(data_config["data_dir"]), train=True, download=False)
        if tuple(source.classes) != runtime.class_names:
            raise ValueError("Official CIFAR class ordering differs from inference mapping.")
        split_seed = runtime.config.get("randomness", {}).get(
            "split_seed", runtime.config.get("seed")
        )
        if split_seed is None:
            raise ValueError("CIFAR checkpoint has no frozen split seed.")
        _, indices = split_train_val_indices(len(source), CIFAR10_VAL_SIZE, int(split_seed))
        for position, index in enumerate(indices[:limit]):
            image, label = source[index]
            samples.append({"source_index": index, "validation_position": position,
                            "true_label": int(label), "image": image})
    else:
        manifest = project_path(data_config["manifest"])
        if sha256_file(manifest) != data_config["manifest_sha256"]:
            raise ValueError("CORe50 manifest differs from the checkpoint's frozen manifest.")
        records = [record for record in load_manifest(manifest) if record.split == "validation"]
        dataset_root = project_path(data_config["dataset_root"])
        for position, record in enumerate(records[:limit]):
            with Image.open(dataset_root / record.relative_path) as image:
                rgb = image.convert("RGB")
                rgb.load()
            samples.append({"source_index": position, "validation_position": position,
                            "relative_path": record.relative_path,
                            "true_label": record.category_label, "image": rgb})
    return samples


def select_cases(
    runtime: InferenceRuntime,
    samples: list[dict[str, object]],
    *,
    count: int,
    batch_size: int,
) -> tuple[dict[str, list[dict[str, object]]], int]:
    """Pick the first clean correct/wrong cases with distinct true classes."""

    if count <= 0 or batch_size <= 0:
        raise ValueError("count and batch_size must be positive.")
    selected = {"correct": [], "wrong": []}
    seen = {"correct": set(), "wrong": set()}
    scanned = 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        inputs = torch.stack([runtime.prepare_image(sample["image"])[0] for sample in chunk])
        predictions = predict_batch(runtime, inputs).probabilities.argmax(dim=1)
        for sample, prediction in zip(chunk, predictions.tolist(), strict=True):
            scanned += 1
            label = int(sample["true_label"])
            status = "correct" if prediction == label else "wrong"
            if len(selected[status]) < count and label not in seen[status]:
                selected[status].append(sample)
                seen[status].add(label)
            if all(len(cases) == count for cases in selected.values()):
                return selected, scanned
    raise ValueError("Not enough distinct-class correct/wrong cases within the fixed scan limit.")


def generate_candidate(
    runtime: InferenceRuntime,
    name: str,
    selected: dict[str, list[dict[str, object]]],
    corruption: dict[str, object],
    output_dir: Path,
) -> list[dict[str, object]]:
    """Render clean and corrupted versions of the same selected images."""

    summary_rows = []
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for status, samples in selected.items():
        inputs, metadata = [], []
        for case_index, sample in enumerate(samples):
            normalized, pixels = runtime.prepare_image(sample["image"])
            image_path = sample_dir / f"{name}_{status}_{case_index + 1}.png"
            sample["image"].save(image_path)
            corrupted = apply_corruption(
                pixels, str(corruption["name"]), int(corruption["severity"]),
                sample_index=int(sample["source_index"]),
                corruption_seed=int(corruption["seed"]),
            )
            # Corrupt pixels before normalization, matching Stage 15's order.
            # 先在 [0,1] 像素空间加扰动，再用正式 mean/std 归一化。
            corrupted_normalized = vision_functional.normalize(
                corrupted, mean=CIFAR10_MEAN, std=CIFAR10_STD
            )
            for condition, tensor in (("clean", normalized),
                                      (f"{corruption['name']} S{corruption['severity']}", corrupted_normalized)):
                inputs.append(tensor)
                metadata.append({
                    "source_index": int(sample["source_index"]),
                    "validation_position": int(sample["validation_position"]),
                    "relative_path": sample.get("relative_path"),
                    "image": image_path.relative_to(PROJECT_ROOT).as_posix(),
                    "image_sha256": sha256_file(image_path),
                    "true_label": int(sample["true_label"]),
                    "true_class": runtime.class_names[int(sample["true_label"])],
                    "selected_by_clean_status": status,
                    "condition": condition,
                    "input_sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
                })
        normalized_batch = torch.stack(inputs)
        predictions = predict_batch(runtime, normalized_batch, attention="both")
        for index, row in enumerate(metadata):
            confidence, predicted = predictions.probabilities[index].max(dim=0)
            row.update({
                "predicted_label": int(predicted),
                "predicted_class": runtime.class_names[int(predicted)],
                "confidence": float(confidence),
                "correct": int(predicted) == row["true_label"],
                "raw_patch_weights": {
                    method: values[index].tolist()
                    for method, values in predictions.patch_weights.items()
                },
            })
        figure = plot_attention_comparison(
            normalized_batch,
            {METHOD_LABELS[method]: maps for method, maps in predictions.heatmaps.items()},
            metadata,
            title=f"{name}: selected clean {status} | same images after corruption",
        )
        path = output_dir / f"{name}_{status}_comparison.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        for row in metadata:
            row["comparison_image"] = path.relative_to(PROJECT_ROOT).as_posix()
        summary_rows.extend(metadata)
    return summary_rows


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/advanced/stage17/diagnostics.yaml")
    args = parser.parse_args(argv)
    config_path = project_path(args.config)
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    selection = config["selection"]
    if selection["rule"] != "first_correct_and_wrong_with_distinct_true_labels":
        raise ValueError("Unknown case selection rule.")
    limit = int(selection["max_samples"])
    if limit <= 0:
        raise ValueError("max_samples must be positive.")
    corruption = config["corruption"]
    validate_corruption(str(corruption["name"]), int(corruption["severity"]))
    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "source_sha256": {
            path: sha256_file(PROJECT_ROOT / path)
            for path in ("src/models/minivit.py", "src/models/__init__.py",
                         "src/inference.py", "src/visualization.py",
                         "scripts/predict.py", "scripts/visualize_rollout.py")
        },
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                        "cuda": torch.version.cuda},
        "git": collect_git_metadata(PROJECT_ROOT),
        "split": "validation",
        "test_images_read": False,
        "selection": selection,
        "corruption": corruption,
        "interpretation": "Per-image relative maps; descriptive, not causal. Cases are illustrative, not aggregate metrics.",
        "candidates": [],
    }
    for candidate in config["candidates"]:
        runtime = load_inference_runtime(
            project_path(candidate["checkpoint"]), device=config["device"],
            expected_sha256=candidate["sha256"],
        )
        samples = validation_samples(runtime, limit)
        selected, scanned = select_cases(
            runtime, samples, count=int(selection["cases_per_status"]),
            batch_size=int(selection["batch_size"]),
        )
        cases = generate_candidate(runtime, candidate["name"], selected, corruption, output_dir)
        result["candidates"].append({
            "name": candidate["name"], **runtime.metadata(),
            "selection_scanned_samples": scanned, "cases": cases,
        })
        print(f"{candidate['name']}: {scanned} validation samples scanned, {len(cases)} clean/corrupted rows saved.")
    (output_dir / "diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    main()
