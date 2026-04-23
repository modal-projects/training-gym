"""Unit tests for ``TrainResult`` and its wiring into each framework launcher.

Intentionally avoids any real Modal API calls — these tests exercise the
pure-Python construction + path-resolution logic, which is where regressions
are most likely to land.

Run from the repo root:

    uv run python tests/test_train_result.py
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch


def test_train_result_defaults_and_dataclass_fields() -> None:
    from modal_training_gym.common.train_result import TrainResult

    tr = TrainResult(
        app_name="demo",
        framework="slime",
        checkpoints_volume_name="demo-checkpoints",
        checkpoints_mount_path="/checkpoints",
        base_model="Qwen/Qwen3-4B",
    )
    assert tr.app_name == "demo"
    assert tr.framework == "slime"
    assert tr.checkpoints_volume_name == "demo-checkpoints"
    assert tr.checkpoints_mount_path == "/checkpoints"
    assert tr.base_model == "Qwen/Qwen3-4B"
    assert tr.checkpoint_subpath == ""
    assert tr.iteration_prefix == ""
    assert tr.volume_version is None
    assert tr.extra == {}


def test_latest_checkpoint_path_without_iterations_returns_mount() -> None:
    """Frameworks with no iteration prefix: ``latest_checkpoint_path``
    returns the mount root (or subpath) without hitting the Modal API.
    """
    from modal_training_gym.common.train_result import TrainResult

    tr = TrainResult(
        app_name="demo",
        framework="miles",
        checkpoints_volume_name="demo-checkpoints",
        checkpoints_mount_path="/checkpoints",
        base_model="Qwen/Qwen3-4B",
    )
    # No iteration_prefix and no subpath — return the mount itself.
    assert tr.latest_checkpoint_path() == "/checkpoints"

    tr2 = TrainResult(
        app_name="demo",
        framework="ms-swift",
        checkpoints_volume_name="demo-checkpoints",
        checkpoints_mount_path="/checkpoints",
        base_model="Qwen/Qwen3-4B",
        checkpoint_subpath="demo_train_123",
    )
    assert tr2.latest_checkpoint_path() == "/checkpoints/demo_train_123"


def test_latest_checkpoint_path_with_iterations_sorts_lexicographically() -> None:
    from modal_training_gym.common.train_result import TrainResult

    tr = TrainResult(
        app_name="demo",
        framework="slime",
        checkpoints_volume_name="demo-checkpoints",
        checkpoints_mount_path="/checkpoints",
        base_model="Qwen/Qwen3-4B",
        iteration_prefix="iter_",
    )

    fake_vol = MagicMock()
    fake_vol.iterdir.return_value = iter(
        [
            _fake_entry("iter_0000020"),
            _fake_entry("iter_0000100"),
            _fake_entry("args.json"),
            _fake_entry("iter_0000050"),
        ]
    )
    with patch.object(TrainResult, "volume", return_value=fake_vol):
        assert tr.latest_checkpoint_path() == "/checkpoints/iter_0000100"


def test_latest_checkpoint_path_raises_when_empty() -> None:
    from modal_training_gym.common.train_result import TrainResult

    tr = TrainResult(
        app_name="demo",
        framework="slime",
        checkpoints_volume_name="demo-checkpoints",
        checkpoints_mount_path="/checkpoints",
        base_model="Qwen/Qwen3-4B",
        iteration_prefix="iter_",
    )
    fake_vol = MagicMock()
    fake_vol.iterdir.return_value = iter([])
    with patch.object(TrainResult, "volume", return_value=fake_vol):
        try:
            tr.latest_checkpoint_path()
        except FileNotFoundError as e:
            msg = str(e)
            assert "demo-checkpoints" in msg
            assert "iter_" in msg
        else:
            raise AssertionError("Expected FileNotFoundError")


def test_build_serve_app_passes_volume_and_path() -> None:
    """``build_serve_app`` wires through the checkpoints volume name and
    picks a sensible default ``served_model_name``.
    """
    from modal_training_gym.common import train_result as tr_mod

    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "fake-app"

    tr = tr_mod.TrainResult(
        app_name="demo",
        framework="slime",
        checkpoints_volume_name="demo-checkpoints",
        checkpoints_mount_path="/checkpoints",
        base_model="Qwen/Qwen3-4B",
    )

    with patch(
        "modal_training_gym.common.serve_vllm.build_vllm_serve_app",
        side_effect=fake_build,
    ):
        # Pass an explicit checkpoint_path so we don't hit the (un-mocked)
        # volume in this test.
        got = tr.build_serve_app(checkpoint_path="/checkpoints/iter_0000050")

    assert got == "fake-app"
    assert captured["app_name"] == "demo-serve"
    assert captured["model_path"] == "/checkpoints/iter_0000050"
    assert captured["served_model_name"] == "demo"
    assert captured["checkpoints_volume"] == "demo-checkpoints"
    assert captured["checkpoints_mount_path"] == "/checkpoints"


def test_each_framework_attaches_train_result() -> None:
    """Smoke test: every ``build_*_app`` exposes ``app.train_result``
    with a consistent shape.
    """
    # Import lazily — these pull in modal, which is already a project dep.
    from modal_training_gym.common.dataset import HuggingFaceDataset
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.train_result import TrainResult

    # SLIME ----------------------------------------------------------------
    from modal_training_gym.frameworks.slime import SlimeConfig

    class _GSM8K(HuggingFaceDataset):
        hf_repo = "zhuzilin/gsm8k"
        input_key = "messages"
        label_key = "label"

    slime_app = SlimeConfig(
        model=Qwen3_4B(),
        dataset=_GSM8K(data_root="/data"),
    ).build_app()
    result = slime_app.train_result
    assert isinstance(result, TrainResult)
    assert result.framework == "slime"
    assert result.checkpoints_volume_name.endswith("-checkpoints")
    assert result.checkpoints_mount_path == "/checkpoints"
    assert result.base_model == "Qwen/Qwen3-4B"
    assert result.iteration_prefix == "iter_"


def _fake_entry(name: str) -> Any:
    """Build a ``FileEntry``-shaped object whose ``path`` field matches
    what ``modal.Volume.iterdir`` yields for a flat listing.
    """
    entry = MagicMock()
    entry.path = name
    return entry


def main() -> int:
    tests = [
        test_train_result_defaults_and_dataclass_fields,
        test_latest_checkpoint_path_without_iterations_returns_mount,
        test_latest_checkpoint_path_with_iterations_sorts_lexicographically,
        test_latest_checkpoint_path_raises_when_empty,
        test_build_serve_app_passes_volume_and_path,
        test_each_framework_attaches_train_result,
    ]
    failures: list[tuple[str, str]] = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001 — test harness
            import traceback

            failures.append((t.__name__, traceback.format_exc()))
            print(f"  FAIL  {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)}/{len(tests)} test(s) failed.")
        for name, tb in failures:
            print(f"\n{name}:\n{tb}")
        return 1
    print(f"\nAll {len(tests)} test(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
