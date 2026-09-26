"""Tests for YAML configuration loading and structure validation."""

from pathlib import Path

import pytest

from src.config import (
    CURRENT_FORMAT_VERSION,
    load_config,
    resolve_format_version,
    resolve_random_seeds,
    save_config,
)


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
    assert data["augmentation"] == "basic"
    assert model["name"] == "minivit"
    assert model["embed_dim"] == 192
    assert model["depth"] == 6
    assert model["num_heads"] == 6
    assert training["epochs"] == 100
    assert training["optimizer"] == "adamw"
    assert training["learning_rate"] == pytest.approx(0.0003)
    assert isinstance(training["learning_rate"], float)
    assert resolve_random_seeds(config) == (42, 42)


# CNN 正式配置应复用相同实验协议，并仅替换模型结构和输出目录。
def test_load_config_reads_cnn_baseline():
    """Load the CNN baseline configuration with its model-selection field."""

    config = load_config(
        PROJECT_ROOT / "configs" / "cnn.yaml"
    )

    assert config["experiment"]["name"] == "cnn_baseline"
    assert config["experiment"]["output_dir"] == "outputs/cnn_baseline"
    assert config["seed"] == 42
    assert config["data"]["batch_size"] == 128
    assert config["data"]["augmentation"] == "basic"
    assert config["model"] == {
        "name": "cnn",
        "in_channels": 3,
        "num_classes": 10,
        "channels": [96, 192, 384, 512],
        "dropout": 0.1,
    }
    assert config["training"]["epochs"] == 100
    assert config["training"]["optimizer"] == "adamw"


# 五份消融配置必须与 MiniViT baseline 仅存在目标变量及实验身份差异。
@pytest.mark.parametrize(
    ("filename", "experiment_name", "model_override", "data_override"),
    [
        (
            "patch_p2.yaml",
            "ablation_patch_p2_d6_aug_seed42",
            {"patch_size": 2},
            {},
        ),
        (
            "patch_p8.yaml",
            "ablation_patch_p8_d6_aug_seed42",
            {"patch_size": 8},
            {},
        ),
        (
            "depth_d3.yaml",
            "ablation_depth_p4_d3_aug_seed42",
            {"depth": 3},
            {},
        ),
        (
            "depth_d9.yaml",
            "ablation_depth_p4_d9_aug_seed42",
            {"depth": 9},
            {},
        ),
        (
            "no_augmentation.yaml",
            "ablation_aug_p4_d6_noaug_seed42",
            {},
            {"augmentation": "none"},
        ),
    ],
)
def test_ablation_configs_change_only_the_intended_variable(
    filename,
    experiment_name,
    model_override,
    data_override,
):
    """Keep every controlled field equal to the baseline configuration."""

    baseline = load_config(
        PROJECT_ROOT / "configs" / "minivit.yaml"
    )
    ablation = load_config(
        PROJECT_ROOT / "configs" / "ablations" / filename
    )

    expected_model = {
        **baseline["model"],
        **model_override,
    }
    expected_data = {
        **baseline["data"],
        **data_override,
    }

    assert ablation["experiment"] == {
        "name": experiment_name,
        "output_dir": f"outputs/{experiment_name}",
    }
    assert ablation["seed"] == baseline["seed"]
    assert ablation["device"] == baseline["device"]
    assert ablation["data"] == expected_data
    assert ablation["model"] == expected_model
    assert ablation["training"] == baseline["training"]


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


def make_minimal_config() -> dict[str, object]:
    """Create the non-randomness portion shared by schema tests."""

    return {
        "experiment": {},
        "device": "cpu",
        "data": {},
        "model": {},
        "training": {},
    }


# v2 配置必须分别保留数据划分种子和训练随机种子。
def test_load_config_accepts_v2_randomness_without_mutating_it(
    tmp_path: Path,
):
    """Load and resolve explicit split/training seeds."""

    config_path = tmp_path / "advanced.yaml"
    config_path.write_text(
        """
experiment: {}
device: cpu
data: {}
model: {}
training: {}
randomness:
  split_seed: 42
  train_seed: 123
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["randomness"] == {
        "split_seed": 42,
        "train_seed": 123,
    }
    assert "seed" not in config
    assert resolve_random_seeds(config) == (42, 123)


# 两套 schema 同时存在时没有可靠的优先级，必须主动拒绝。
def test_resolve_random_seeds_rejects_ambiguous_or_missing_schema():
    """Require exactly one legacy or advanced randomness schema."""

    base = make_minimal_config()

    with pytest.raises(ValueError, match="not both"):
        resolve_random_seeds({
            **base,
            "seed": 42,
            "randomness": {
                "split_seed": 42,
                "train_seed": 123,
            },
        })

    with pytest.raises(ValueError, match="either 'seed' or 'randomness'"):
        resolve_random_seeds(base)


# v2 不允许缺省任一 seed，否则程序会再次产生隐式耦合。
@pytest.mark.parametrize(
    ("randomness", "missing_name"),
    [
        ({"train_seed": 123}, "split_seed"),
        ({"split_seed": 42}, "train_seed"),
    ],
)
def test_resolve_random_seeds_requires_both_v2_fields(
    randomness,
    missing_name,
):
    """Reject incomplete version 2 randomness mappings."""

    with pytest.raises(ValueError, match=missing_name):
        resolve_random_seeds({
            **make_minimal_config(),
            "randomness": randomness,
        })


# bool、浮点数、负数与超过 PyTorch 支持范围的值都应尽早报错。
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("seed", 1.5),
        ("randomness.split_seed", -1),
        ("randomness.train_seed", 2**32),
    ],
)
def test_resolve_random_seeds_rejects_invalid_values(field, value):
    """Validate seed type and the unsigned 32-bit range."""

    if field == "seed":
        config = {
            **make_minimal_config(),
            "seed": value,
        }
    else:
        config = {
            **make_minimal_config(),
            "randomness": {
                "split_seed": 42,
                "train_seed": 123,
            },
        }
        section, key = field.split(".")
        config[section][key] = value

    with pytest.raises(ValueError, match=field):
        resolve_random_seeds(config)


def test_load_config_rejects_non_mapping_randomness(tmp_path: Path):
    """Reject a scalar advanced randomness section."""

    config_path = tmp_path / "invalid_randomness.yaml"
    config_path.write_text(
        """
experiment: {}
device: cpu
data: {}
model: {}
training: {}
randomness: 42
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'randomness'.*mapping"):
        load_config(config_path)


# 没有版本字段的历史配置按 v1 读取；新产物显式标记当前版本。
def test_resolve_format_version_supports_legacy_and_current_configs():
    """Default missing versions to v1 and accept the current version."""

    assert resolve_format_version({}) == 1
    assert resolve_format_version({
        "format_version": CURRENT_FORMAT_VERSION,
    }) == CURRENT_FORMAT_VERSION


# bool 虽是 int 子类但不应被当作版本号，未知版本也不能猜测解析。
@pytest.mark.parametrize("version", [True, "2", 0, 3])
def test_resolve_format_version_rejects_invalid_or_unknown_values(version):
    """Reject malformed and unsupported artifact format versions."""

    with pytest.raises(ValueError, match="format_version"):
        resolve_format_version({"format_version": version})
