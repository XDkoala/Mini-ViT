"""Tests for YAML configuration loading and structure validation."""

from pathlib import Path

import pytest

from src.config import load_config, save_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 正式 MiniViT 配置必须能读取，并保留关键字段的正确 Python 类型。
def test_load_config_reads_minivit_baseline():
    """Load the real baseline configuration with expected values and types."""

    config = load_config(
        PROJECT_ROOT / "configs" / "minivit.yaml"
    )

    assert config["seed"] == 42
    assert config["device"] == "auto"

    experiment = config["experiment"]
    data = config["data"]
    model = config["model"]
    training = config["training"]

    assert experiment["name"] == "minivit_baseline"
    assert experiment["output_dir"] == "outputs/minivit_baseline"
    assert data["batch_size"] == 128
    assert data["num_workers"] == 0
    assert model["embed_dim"] == 192
    assert model["depth"] == 6
    assert model["num_heads"] == 6
    assert training["epochs"] == 100
    assert training["optimizer"] == "adamw"
    assert training["learning_rate"] == pytest.approx(0.0003)
    assert isinstance(training["learning_rate"], float)


@pytest.mark.parametrize(
    ("yaml_text", "error_message"),
    [
        (
            "",
            "configuration root must be a mapping",
        ),
        (
            "- one\n- two\n",
            "configuration root must be a mapping",
        ),
        (
            """
experiment: {}
seed: 42
data: {}
model: {}
training: {}
""",
            "Missing required configuration keys: device",
        ),
        (
            """
experiment: {}
seed: 42
device: auto
data: {}
model: minivit
training: {}
""",
            "Configuration section 'model' must be a mapping",
        ),
    ],
)
def test_load_config_rejects_invalid_structure(
    tmp_path: Path,
    yaml_text: str,
    error_message: str,
):
    """Reject empty, non-mapping, incomplete, or malformed structures."""

    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        yaml_text,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=error_message,
    ):
        load_config(config_path)


# 不存在的配置文件应保留 pathlib 提供的明确错误类型。
def test_load_config_preserves_file_not_found_error(tmp_path: Path):
    """Raise FileNotFoundError when the requested YAML file is absent."""

    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


# 有效配置写出后再读回，数据内容和 Python 类型必须保持一致。
def test_save_config_round_trips_effective_configuration(tmp_path: Path):
    """Create parent directories and preserve nested effective settings."""

    config: dict[str, object] = {
        "experiment": {
            "name": "阶段五冒烟",
            "output_dir": "outputs/stage5_smoke",
        },
        "seed": 42,
        "device": "auto",
        "data": {
            "data_dir": "data",
            "batch_size": 128,
            "num_workers": 0,
        },
        "model": {
            "embed_dim": 192,
            "depth": 6,
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.0003,
        },
        "runtime": {
            "is_smoke_test": True,
            "max_train_batches": 2,
            "max_val_batches": 2,
        },
    }
    output_path = tmp_path / "nested" / "experiment" / "config.yaml"

    result = save_config(config, output_path)

    assert result is None
    assert output_path.is_file()
    assert load_config(output_path) == config
    assert "阶段五冒烟" in output_path.read_text(encoding="utf-8")
