"""Golden-file test for the Slime rollout-status patcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from modal_training_gym.frameworks.slime.modal_helpers.patches import (
    patch_rollout_status_reporting as patcher,
)

TESTDATA = Path(__file__).parent / "testdata" / "slime"


@pytest.fixture(scope="session")
def slime_inputs() -> dict[str, str]:
    inputs = sorted(TESTDATA.glob("*.input"))
    assert inputs
    return {path.name.removesuffix(".input"): path.read_text() for path in inputs}


def test_missing_patch_target_is_skipped(tmp_path, capsys):
    missing = tmp_path / "train_async.py"
    patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping rollout-status patch" in capsys.readouterr().out


def test_patch_matches_golden(slime_inputs, tmp_path, request, capsys):
    rewrite_goldens = request.config.getoption("--rewrite")
    for name, source in slime_inputs.items():
        golden_path = TESTDATA / f"{name}.output"
        work = tmp_path / name
        work.write_text(source)
        patcher._patch_file(work)
        actual = work.read_text()
        assert "WARNING: Could not patch" not in capsys.readouterr().out

        if rewrite_goldens:
            golden_path.write_text(actual)
            continue

        assert golden_path.exists(), (
            f"Golden output file does not exist: {golden_path}. "
            "Regenerate and review the expected patch output with "
            "uv run pytest tests/test_substep_times.py --rewrite."
        )
        expected = golden_path.read_text()
        assert actual == expected, (
            f"golden mismatch for {name}; rerun with --rewrite to accept"
        )
