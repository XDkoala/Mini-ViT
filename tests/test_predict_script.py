"""End-to-end tests for the single-image prediction CLI."""

import json
import subprocess
import sys

import pytest
from PIL import Image

from scripts.predict import main
from tests.test_inference import write_checkpoint


def test_predict_writes_checkpoint_bound_report_and_both_maps(tmp_path):
    """Produce a readable report and labeled attention image for one input."""

    # 命令行必须真正保存分类结果、模型身份和两种方法的原始权重。
    checkpoint = tmp_path / "best.pt"
    write_checkpoint(checkpoint, dataset="core50")
    image_path = tmp_path / "image.png"
    Image.new("RGB", (128, 128), (80, 100, 120)).save(image_path)
    output = tmp_path / "prediction"
    report = main(["--checkpoint", str(checkpoint), "--image", str(image_path),
                   "--attention", "both", "--device", "cpu", "--output-dir", str(output)])
    saved = json.loads((output / "prediction.json").read_text())
    assert saved == report
    assert report["dataset"] == "core50"
    assert report["class_names"][0] == "plug_adapters"
    assert len(report["top_k"]) == 5
    assert set(report["raw_patch_weights"]) == {"last", "rollout"}
    assert (output / "attention.png").is_file()
    assert len(report["checkpoint_sha256"]) == 64


def test_predict_rejects_invalid_image_and_top_k(tmp_path):
    """Explain invalid inputs before saving an incomplete report."""

    # 错误输入不能生成一份看起来正常的预测报告。
    checkpoint = tmp_path / "best.pt"
    write_checkpoint(checkpoint)
    image_path = tmp_path / "image.png"
    Image.new("RGB", (32, 32)).save(image_path)
    with pytest.raises(ValueError, match="top-k"):
        main(["--checkpoint", str(checkpoint), "--image", str(image_path), "--top-k", "11"])
    image_path.write_text("not an image")
    with pytest.raises(OSError):
        main(["--checkpoint", str(checkpoint), "--image", str(image_path)])


def test_predict_direct_cli_has_readable_missing_path_error():
    """Run the direct entry point and avoid a traceback for a missing image."""

    # 直接 python scripts/predict.py 也应能导入 src，并给出清楚的路径错误。
    result = subprocess.run(
        [sys.executable, "scripts/predict.py", "--checkpoint", "missing.pt",
         "--image", "missing.png"], capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Image not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_predict_preserves_an_input_named_attention_png(tmp_path):
    """Prevent an explicitly chosen output folder from overwriting the input."""

    # 输入恰好叫 attention.png 时，不能被输出的叠图覆盖。
    checkpoint = tmp_path / "best.pt"
    write_checkpoint(checkpoint)
    image_path = tmp_path / "attention.png"
    Image.new("RGB", (32, 32)).save(image_path)
    before = image_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        main(["--checkpoint", str(checkpoint), "--image", str(image_path),
              "--attention", "both", "--output-dir", str(tmp_path)])
    assert image_path.read_bytes() == before
