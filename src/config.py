"""Configuration loading and basic structure validation."""

from pathlib import Path

import yaml

# Stage 5

REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "seed",
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
