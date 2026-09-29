"""Tests for the Stage 16 CORe50 preparation CLI."""

import subprocess
import sys

from scripts.prepare_core50 import parse_args


def test_prepare_core50_cli_defaults_enable_formal_verification() -> None:
    args = parse_args([])

    assert args.dataset_root.name == "core50_128x128"
    assert args.archive.name == "core50_128x128.zip"
    assert args.manifest.name == "stage16_core50.csv"
    assert args.summary.name == "dataset_summary.json"
    assert args.skip_image_verification is False


def test_prepare_core50_script_can_run_directly() -> None:
    """Catch project-root import failures in the documented CLI form."""

    result = subprocess.run(
        [sys.executable, "scripts/prepare_core50.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Prepare the frozen Stage 16" in result.stdout
