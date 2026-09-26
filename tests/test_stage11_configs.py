"""Contract tests for the Stage 11 optimization ablation matrix."""

from copy import deepcopy
from pathlib import Path

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "advanced" / "stage11"
CONFIG_PATHS = {
    "o0": CONFIG_DIR / "o0_fixed_lr.yaml",
    "o1": CONFIG_DIR / "o1_cosine.yaml",
    "o2": CONFIG_DIR / "o2_warmup_cosine.yaml",
    "o3": CONFIG_DIR / "o3_warmup_cosine_clip.yaml",
}


def load_matrix() -> dict[str, dict[str, object]]:
    """Load all formal Stage 11 configurations through the public loader."""

    return {
        name: load_config(path)
        for name, path in CONFIG_PATHS.items()
    }


# 模型、数据、随机种子和基础优化器必须完全相同，避免混入其他变量。
def test_stage11_matrix_freezes_all_shared_experiment_factors():
    """Keep every non-optimization factor identical across O0 through O3."""

    matrix = load_matrix()
    reference = matrix["o0"]

    for config in matrix.values():
        assert config["format_version"] == 2
        assert config["randomness"] == reference["randomness"]
        assert config["device"] == reference["device"]
        assert config["data"] == reference["data"]
        assert config["model"] == reference["model"]
        assert config["training"]["epochs"] == 100
        assert config["training"]["optimizer"] == "adamw"
        assert config["training"]["learning_rate"] == 0.0003
        assert config["training"]["weight_decay"] == 0.05
        assert config["training"]["amp"] is False


# O0→O1 只加入 cosine；O1→O2 只加入 warmup；O2→O3 只加入 clipping。
def test_stage11_matrix_changes_only_the_intended_optimization_factor():
    """Encode the cumulative O0-to-O3 ablation without hidden differences."""

    matrix = load_matrix()

    assert matrix["o0"]["training"]["scheduler"] == {"name": "none"}
    assert matrix["o0"]["training"]["gradient_clip_norm"] is None

    assert matrix["o1"]["training"]["scheduler"] == {
        "name": "warmup_cosine",
        "warmup_epochs": 0,
        "start_factor": 1.0,
        "min_factor": 0.1,
    }
    assert matrix["o1"]["training"]["gradient_clip_norm"] is None

    assert matrix["o2"]["training"]["scheduler"] == {
        "name": "warmup_cosine",
        "warmup_epochs": 5,
        "start_factor": 0.1,
        "min_factor": 0.1,
    }
    assert matrix["o2"]["training"]["gradient_clip_norm"] is None

    o3_training = deepcopy(matrix["o3"]["training"])
    o3_training["gradient_clip_norm"] = None
    assert o3_training == matrix["o2"]["training"]
    assert matrix["o3"]["training"]["gradient_clip_norm"] == 1.0


# 正式运行必须拥有不同 ID 和输出目录，不能互相覆盖。
def test_stage11_matrix_uses_unique_formal_output_directories():
    """Give every ablation an isolated traceable experiment identity."""

    matrix = load_matrix()
    names = {
        config["experiment"]["name"]
        for config in matrix.values()
    }
    output_dirs = {
        config["experiment"]["output_dir"]
        for config in matrix.values()
    }

    assert len(names) == len(matrix)
    assert len(output_dirs) == len(matrix)
    assert all(
        str(path).startswith("outputs/advanced/stage11/")
        for path in output_dirs
    )
