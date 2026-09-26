"""Configuration loading and basic structure validation."""

from collections.abc import Mapping
from pathlib import Path

import yaml

# Stage 5

REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "device",
    "data",
    "model",
    "training",
}

MAPPING_SECTIONS = (
    "experiment",
    "data",
    "model",
    "training",
)

SEED_UPPER_BOUND = 2**32
LEGACY_FORMAT_VERSION = 1
CURRENT_FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = {
    LEGACY_FORMAT_VERSION,
    CURRENT_FORMAT_VERSION,
}


def resolve_format_version(
    config: Mapping[str, object],
) -> int:
    """Return the artifact format version, treating a missing field as v1."""

    version = config.get(
        "format_version",
        LEGACY_FORMAT_VERSION,
    )

    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("format_version must be an integer.")

    if version not in SUPPORTED_FORMAT_VERSIONS:
        supported = ", ".join(
            str(item)
            for item in sorted(SUPPORTED_FORMAT_VERSIONS)
        )
        raise ValueError(
            "Unsupported format_version "
            f"{version}; supported versions: {supported}."
        )

    return version


def _validate_seed(
    value: object,
    field_name: str,
) -> int:
    """Validate one seed and return it as an integer."""

    # bool 是 int 的子类，但把 true/false 当随机种子通常是配置笔误。
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{field_name} must be an integer."
        )

    if not 0 <= value < SEED_UPPER_BOUND:
        raise ValueError(
            f"{field_name} must be in the range "
            f"[0, {SEED_UPPER_BOUND})."
        )

    return value


def resolve_random_seeds(
    config: Mapping[str, object],
) -> tuple[int, int]:
    """Resolve legacy or advanced randomness configuration.

    Version 1 configurations contain one top-level ``seed``. It controls
    both the dataset split and training randomness so historical runs keep
    their original behavior. Version 2 configurations use an explicit
    ``randomness`` mapping with independent ``split_seed`` and
    ``train_seed`` values.
    """

    has_legacy_seed = "seed" in config
    has_randomness = "randomness" in config

    if has_legacy_seed and has_randomness:
        raise ValueError(
            "Configuration must use either 'seed' or "
            "'randomness', not both."
        )

    if not has_legacy_seed and not has_randomness:
        raise ValueError(
            "Configuration must define either 'seed' "
            "or 'randomness'."
        )

    if has_legacy_seed:
        legacy_seed = _validate_seed(
            config["seed"],
            "seed",
        )
        return legacy_seed, legacy_seed

    randomness = config["randomness"]
    if not isinstance(randomness, Mapping):
        raise ValueError(
            "Configuration section 'randomness' "
            "must be a mapping."
        )

    required_seed_keys = {
        "split_seed",
        "train_seed",
    }
    missing_seed_keys = required_seed_keys - randomness.keys()

    if missing_seed_keys:
        missing_names = ", ".join(
            sorted(missing_seed_keys)
        )
        raise ValueError(
            "Missing randomness keys: "
            f"{missing_names}"
        )

    split_seed = _validate_seed(
        randomness["split_seed"],
        "randomness.split_seed",
    )
    train_seed = _validate_seed(
        randomness["train_seed"],
        "randomness.train_seed",
    )

    return split_seed, train_seed


def load_config(
    path: str | Path,
) -> dict[str, object]:
    """Load a YAML configuration and validate its top-level structure."""

    config_path = Path(path)

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The configuration root must be a mapping."
        )

    missing_keys = (
        REQUIRED_TOP_LEVEL_KEYS
        - config.keys()
    )

    if missing_keys:
        missing_names = ", ".join(
            sorted(missing_keys)
        )
        raise ValueError(
            f"Missing required configuration keys: {missing_names}"
        )

    for section_name in MAPPING_SECTIONS:
        if not isinstance(
            config[section_name],
            dict,
        ):
            raise ValueError(
                f"Configuration section "
                f"'{section_name}' must be a mapping."
            )

    # 在读取阶段尽早拒绝模糊或非法的随机性配置。
    # 函数只校验并解析，不修改原始 config，因此旧 YAML 可原样 round-trip。
    resolve_format_version(config)
    resolve_random_seeds(config)

    return config


def save_config(
    config: dict[str, object],
    path: str | Path,
) -> None:
    """Write an effective experiment configuration to YAML."""

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )
