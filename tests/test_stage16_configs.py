"""Contracts for the frozen Stage 16 formal transfer matrix."""

from copy import deepcopy
from pathlib import Path

from scripts.train_transfer import load_transfer_config


CONFIG_DIR = Path("configs/advanced/stage16")
CONFIG_NAMES = ("scratch", "linear_probe", "fine_tune")


def _load_all() -> dict[str, dict[str, object]]:
    return {
        name: load_transfer_config(CONFIG_DIR / f"{name}.yaml")
        for name in CONFIG_NAMES
    }


def test_stage16_configs_are_valid_and_match_their_modes() -> None:
    configs = _load_all()

    assert {
        name: config["transfer"]["mode"]
        for name, config in configs.items()
    } == {name: name for name in CONFIG_NAMES}
    assert all(config["randomness"] == {"train_seed": 42} for config in configs.values())
    assert all(config["training"]["epochs"] == 50 for config in configs.values())


def test_stage16_matrix_changes_only_identity_and_transfer_mode() -> None:
    configs = _load_all()
    normalized = []
    for config in configs.values():
        copy = deepcopy(config)
        copy.pop("experiment")
        copy.pop("transfer")
        normalized.append(copy)

    assert normalized[0] == normalized[1] == normalized[2]


def test_stage16_pretrained_configs_share_frozen_source() -> None:
    configs = _load_all()
    scratch = configs["scratch"]["transfer"]
    probe = configs["linear_probe"]["transfer"]
    fine_tune = configs["fine_tune"]["transfer"]

    assert scratch["source_checkpoint"] is None
    assert scratch["source_sha256"] is None
    assert probe["source_checkpoint"] == fine_tune["source_checkpoint"]
    assert probe["source_sha256"] == fine_tune["source_sha256"]
    assert probe["source_sha256"] == (
        "d7d1f42c5de6bb742a1196e70d16b49cdb45b79f986c69f65477fd65e7db7309"
    )


def test_stage16_configs_pin_manifest_identity() -> None:
    configs = _load_all()
    expected = "d12f46f688851a08287f0220c85f49c5d2ee05e0c084b81066d9b79d101633d0"

    assert all(config["data"]["manifest_sha256"] == expected for config in configs.values())
    assert all("split_seed" not in config["randomness"] for config in configs.values())
