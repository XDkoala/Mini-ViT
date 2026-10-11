"""Tests for checkpoint-specific inference and preprocessing contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.core50 import CORE50_CATEGORY_NAMES, build_core50_transforms
from src.data import CIFAR10_CLASS_NAMES, build_transforms
from src.inference import load_inference_runtime, predict_batch, top_k_predictions
from src.models import build_model


def write_checkpoint(path: Path, dataset: str = "cifar10", model_name: str = "minivit") -> dict:
    """Create a small but valid 32px classifier checkpoint."""

    model_config = {"name": model_name, "num_classes": 10, "in_channels": 3}
    if model_name == "minivit":
        model_config.update(image_size=32, patch_size=8, embed_dim=12, depth=2,
                            num_heads=3, mlp_dim=24, dropout=0.1)
    else:
        model_config.update(channels=[4, 8], dropout=0.1)
    config = {"model": model_config, "data": {"data_dir": "data"},
              "experiment": {"name": "inference_test"}, "seed": 42}
    if dataset == "core50":
        config["data"] = {"dataset_root": "not_needed_for_inference",
                          "manifest": "not_needed.csv", "manifest_sha256": "unused"}
        config["transfer"] = {"mode": "fine_tune", "source_checkpoint": "not_needed.pt"}
    model = build_model(model_config, torch.device("cpu"))
    torch.save({"config": config, "model_state": model.state_dict(), "epoch": 7}, path)
    return config


@pytest.mark.parametrize("dataset", ["cifar10", "core50"])
def test_checkpoint_mapping_and_preprocessing_match_evaluation(tmp_path, dataset):
    """Use each checkpoint's actual label ordering and evaluation transform."""

    # 两个任务虽均为十类，但类别语义与缩放规则不能混用。
    path = tmp_path / "best.pt"
    write_checkpoint(path, dataset)
    runtime = load_inference_runtime(path, device="cpu")
    expected_names = CIFAR10_CLASS_NAMES if dataset == "cifar10" else CORE50_CATEGORY_NAMES
    assert runtime.class_names == expected_names
    size = 32 if dataset == "cifar10" else 128
    image = Image.fromarray(np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3))
    transform = build_transforms("none")[1] if dataset == "cifar10" else build_core50_transforms("none")[1]
    normalized, pixels = runtime.prepare_image(image)
    assert torch.equal(normalized, transform(image))
    assert pixels.shape == (3, 32, 32)
    assert not runtime.model.training


def test_attention_modes_preserve_logits_and_raw_weights(tmp_path):
    """All attention modes share the same restored prediction."""

    # 返回注意力只改变辅助输出，不能改变分类结果。
    path = tmp_path / "best.pt"
    write_checkpoint(path)
    runtime = load_inference_runtime(path, device="cpu")
    images = torch.randn(2, 3, 32, 32)
    plain = predict_batch(runtime, images)
    last = predict_batch(runtime, images, attention="last")
    rollout = predict_batch(runtime, images, attention="rollout")
    both = predict_batch(runtime, images, attention="both")
    for output in (last, rollout, both):
        assert torch.allclose(plain.logits, output.logits)
        assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(2))
    assert torch.equal(last.patch_weights["last"], both.patch_weights["last"])
    assert torch.equal(rollout.patch_weights["rollout"], both.patch_weights["rollout"])
    assert both.heatmaps["rollout"].shape == (2, 32, 32)
    assert both.patch_weights["rollout"].shape == (2, 16)


def test_top_k_sorts_and_breaks_ties_by_class_index():
    """Rank softmax scores deterministically and reject invalid k."""

    # 相同分数按原标签顺序排列；k 越界必须报错。
    probabilities = torch.tensor([0.1, 0.4, 0.4, 0.1])
    result = top_k_predictions(probabilities, ("a", "b", "c", "d"), 3)
    assert [row["class"] for row in result] == ["b", "c", "a"]
    with pytest.raises(ValueError, match="top-k"):
        top_k_predictions(probabilities, ("a", "b", "c", "d"), 5)


def test_loader_rejects_wrong_hash_and_wrong_class_mapping(tmp_path):
    """Fail before inference when checkpoint identity or label count is wrong."""

    # 哈希和分类头维度绑定报告中的类别映射。
    path = tmp_path / "best.pt"
    write_checkpoint(path)
    with pytest.raises(ValueError, match="SHA-256"):
        load_inference_runtime(path, expected_sha256="bad")
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["config"]["model"]["num_classes"] = 5
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="class mapping"):
        load_inference_runtime(path)


def test_cnn_predicts_but_rejects_attention(tmp_path):
    """CNN checkpoints use the CLI's plain prediction path."""

    # CNN 能分类，但没有本项目 MiniViT 的注意力接口。
    path = tmp_path / "best.pt"
    write_checkpoint(path, model_name="cnn")
    runtime = load_inference_runtime(path, device="cpu")
    images = torch.zeros(1, 3, 32, 32)
    assert predict_batch(runtime, images).logits.shape == (1, 10)
    with pytest.raises(ValueError, match="MiniViT"):
        predict_batch(runtime, images, attention="rollout")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_cuda_prediction_and_rollout_agree(tmp_path):
    """Verify device portability of classification and rollout calculations."""

    # 在有 CUDA 的机器上实际计算并比较 CPU/GPU，不能只检查设备名字。
    torch.manual_seed(17)
    path = tmp_path / "best.pt"
    write_checkpoint(path)
    images = torch.randn(2, 3, 32, 32)
    cpu = predict_batch(load_inference_runtime(path, device="cpu"), images, attention="both")
    cuda = predict_batch(load_inference_runtime(path, device="cuda"), images, attention="both")
    # Default cuDNN may use TF32 convolution; allow its measured rounding error.
    # 默认 cuDNN 可用 TF32 卷积，允许实测量级的舍入误差，并仍要求类别一致。
    assert torch.allclose(cpu.logits, cuda.logits, atol=5e-4, rtol=1e-4)
    assert torch.equal(cpu.probabilities.argmax(-1), cuda.probabilities.argmax(-1))
    assert torch.allclose(cpu.patch_weights["rollout"], cuda.patch_weights["rollout"], atol=1e-5, rtol=1e-4)
