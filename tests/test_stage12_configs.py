"""Contract tests for Stage 12 regularization screening configurations."""

from copy import deepcopy
from pathlib import Path

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "advanced" / "stage12" / "screening"
CONFIG_PATHS = {
    "ls005": CONFIG_DIR / "label_smoothing_005.yaml",
    "ls010": CONFIG_DIR / "label_smoothing_010.yaml",
    "mixup02": CONFIG_DIR / "mixup_alpha02.yaml",
    "mixup04": CONFIG_DIR / "mixup_alpha04.yaml",
    "cutmix05": CONFIG_DIR / "cutmix_alpha05.yaml",
    "cutmix10": CONFIG_DIR / "cutmix_alpha10.yaml",
}


def load_matrix() -> dict[str, dict[str, object]]:
    """Load every Stage 12 screening candidate through the public loader."""

    return {name: load_config(path) for name, path in CONFIG_PATHS.items()}


def shared_training_config(config: dict[str, object]) -> dict[str, object]:
    """Remove the one intended regularization factor from a copied config."""

    training = deepcopy(config["training"])
    training.pop("label_smoothing")
    training.pop("batch_mixing")
    return training


# 六组候选必须共享模型、数据、seed 与优化协议，只筛正则化强度。
def test_stage12_screening_matrix_freezes_shared_factors():
    """Keep all non-regularization factors identical across candidates."""

    matrix = load_matrix()
    reference = matrix["ls005"]

    for config in matrix.values():
        assert config["format_version"] == 2
        assert config["experiment"]["run_type"] == "screening"
        assert config["randomness"] == reference["randomness"]
        assert config["device"] == reference["device"]
        assert config["data"] == reference["data"]
        assert config["model"] == reference["model"]
        assert shared_training_config(config) == shared_training_config(reference)
        assert config["training"]["epochs"] == 30
        assert config["training"]["scheduler"]["warmup_epochs"] == 2
        assert config["training"]["amp"] is True


# 每对候选只改变预注册强度，且三种方法不能在同一配置中叠加。
def test_stage12_screening_matrix_encodes_exact_candidate_values():
    """Represent the six pre-registered single-method candidates exactly."""

    matrix = load_matrix()

    assert matrix["ls005"]["training"]["label_smoothing"] == 0.05
    assert matrix["ls010"]["training"]["label_smoothing"] == 0.10
    for name in ("ls005", "ls010"):
        assert matrix[name]["training"]["batch_mixing"] == {"name": "none"}

    for name, alpha in (("mixup02", 0.2), ("mixup04", 0.4)):
        assert matrix[name]["training"]["label_smoothing"] == 0.0
        assert matrix[name]["training"]["batch_mixing"] == {
            "name": "mixup",
            "alpha": alpha,
        }

    for name, alpha in (("cutmix05", 0.5), ("cutmix10", 1.0)):
        assert matrix[name]["training"]["label_smoothing"] == 0.0
        assert matrix[name]["training"]["batch_mixing"] == {
            "name": "cutmix",
            "alpha": alpha,
        }


# 候选必须写入隔离目录，防止运行之间相互覆盖。
def test_stage12_screening_matrix_uses_unique_output_directories():
    """Give each screening candidate a unique traceable identity."""

    matrix = load_matrix()
    names = {config["experiment"]["name"] for config in matrix.values()}
    output_dirs = {
        config["experiment"]["output_dir"]
        for config in matrix.values()
    }

    assert len(names) == len(matrix)
    assert len(output_dirs) == len(matrix)
    assert all(
        str(path).startswith("outputs/advanced/stage12/screening/")
        for path in output_dirs
    )
