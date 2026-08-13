"""Golden-file tests for the miles rollout-status and advantage patchers."""

from __future__ import annotations

from pathlib import Path

import pytest

from modal_training_gym.frameworks.miles.modal_helpers.patches import (
    patch_advantage_distribution as advantage_patcher,
    patch_rollout_status_reporting as rollout_patcher,
)

TESTDATA = Path(__file__).parent / "testdata" / "miles"

# The async driver dispatches generation through futures and has no offload
# calls, so these rollout-patch anchors are expected to miss there.
EXPECTED_ASYNC_MISSES = "generate_rollouts, offload_rollout, offload_train"


@pytest.fixture(scope="session")
def miles_inputs() -> dict[str, str]:
    inputs = sorted(TESTDATA.glob("*.input"))
    assert inputs
    return {path.name.removesuffix(".input"): path.read_text() for path in inputs}


def test_missing_patch_targets_are_skipped(tmp_path, capsys):
    missing = tmp_path / "train.py"
    rollout_patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping rollout-status patch" in capsys.readouterr().out

    advantage_patcher._patch_file(missing)
    assert not missing.exists()
    assert "not found, skipping advantage-distribution patch" in capsys.readouterr().out


def _apply_patcher(name: str, work: Path) -> None:
    if name == "log_utils.py":
        advantage_patcher._patch_file(work)
    else:
        rollout_patcher._patch_file(work)


def test_patch_matches_golden(miles_inputs, tmp_path, request, capsys):
    rewrite_goldens = request.config.getoption("--rewrite")
    for name, source in miles_inputs.items():
        golden_path = TESTDATA / f"{name}.output"
        work = tmp_path / name
        work.write_text(source)
        _apply_patcher(name, work)
        actual = work.read_text()

        out = capsys.readouterr().out
        if name == "train_async.py":
            assert f"Could not patch train_async.py for: {EXPECTED_ASYNC_MISSES}" in out
        else:
            assert "WARNING: Could not" not in out

        if rewrite_goldens:
            golden_path.write_text(actual)
            continue

        assert golden_path.exists(), (
            f"Golden output file does not exist: {golden_path}. "
            "Regenerate and review the expected patch output with "
            "uv run pytest tests/test_miles_patches.py --rewrite."
        )
        expected = golden_path.read_text()
        assert actual == expected, (
            f"golden mismatch for {name}; rerun with --rewrite to accept"
        )
