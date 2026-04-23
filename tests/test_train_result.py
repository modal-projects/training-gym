"""Unit tests for :class:`TrainResult`.

Intentionally avoids real Modal API calls — these tests exercise the
pure-Python construction, serialization, and path-resolution logic,
which is where regressions are most likely to land. Modal-connected
methods (``volume``, ``save``, ``load``) are patched.

Run from the repo root:

    uv run python tests/test_train_result.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, patch


def _make() -> Any:
    from modal_training_gym.common.train_result import TrainResult

    return TrainResult(
        app_name="demo",
        framework="ms-swift",
        run_id="train_1700000000",
        checkpoint_dir="/checkpoints/demo_train_1700000000",
        base_model="Qwen/Qwen3-4B",
        checkpoints_volume_name="demo-checkpoints",
        checkpoints_mount_path="/checkpoints",
        iteration_prefix="iter_",
    )


def test_dataclass_round_trip_via_asdict() -> None:
    """``save`` persists the result via ``asdict`` and ``load`` must be
    able to reconstruct the exact same instance."""
    from modal_training_gym.common.train_result import TrainResult

    tr = _make()
    data = asdict(tr)
    assert data["run_id"] == "train_1700000000"
    assert data["framework"] == "ms-swift"
    assert data["iteration_prefix"] == "iter_"
    assert data["extra"] == {}
    # Round-trip reconstruction
    assert TrainResult(**data) == tr


def test_latest_checkpoint_path_without_iterations_returns_dir() -> None:
    """Frameworks without iteration subdirs return ``checkpoint_dir``
    as-is without touching Modal."""
    from modal_training_gym.common.train_result import TrainResult

    tr = TrainResult(
        app_name="demo",
        framework="miles",
        run_id="demo-1700000000",
        checkpoint_dir="/checkpoints/demo-1700000000",
        base_model="Qwen/Qwen3-4B",
        checkpoints_volume_name="demo-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )
    assert tr.latest_checkpoint_path() == "/checkpoints/demo-1700000000"


def test_latest_checkpoint_path_with_iterations_sorts_lexicographically() -> None:
    from modal_training_gym.common.train_result import TrainResult

    tr = _make()
    fake_vol = MagicMock()
    fake_vol.iterdir.return_value = iter(
        [
            _fake_entry("demo_train_1700000000/iter_0000020"),
            _fake_entry("demo_train_1700000000/iter_0000100"),
            _fake_entry("demo_train_1700000000/args.json"),
            _fake_entry("demo_train_1700000000/iter_0000050"),
        ]
    )
    with patch.object(TrainResult, "volume", return_value=fake_vol):
        assert (
            tr.latest_checkpoint_path()
            == "/checkpoints/demo_train_1700000000/iter_0000100"
        )


def test_latest_checkpoint_path_raises_when_empty() -> None:
    from modal_training_gym.common.train_result import TrainResult

    tr = _make()
    fake_vol = MagicMock()
    fake_vol.iterdir.return_value = iter([])
    with patch.object(TrainResult, "volume", return_value=fake_vol):
        try:
            tr.latest_checkpoint_path()
        except FileNotFoundError as e:
            msg = str(e)
            assert "iter_" in msg
            assert "demo_train_1700000000" in msg
        else:
            raise AssertionError("Expected FileNotFoundError")


def test_volume_always_uses_version_2() -> None:
    """Per the user's requirement: ``TrainResult.volume()`` must always
    pass ``version=2``."""
    import modal_training_gym.common.train_result as tr_mod

    tr = _make()
    captured: dict[str, Any] = {}

    class _FakeVolume:
        @classmethod
        def from_name(cls, name: str, **kwargs: Any) -> "_FakeVolume":
            captured["name"] = name
            captured["kwargs"] = kwargs
            return cls()

    with patch.object(tr_mod, "__builtins__", dict(tr_mod.__builtins__)):
        # The import inside ``.volume()`` grabs ``modal.Volume``. Patch that.
        with patch.dict(
            "sys.modules",
            {"modal": MagicMock(Volume=_FakeVolume)},
        ):
            tr.volume()

    assert captured["name"] == "demo-checkpoints"
    assert captured["kwargs"] == {"create_if_missing": True, "version": 2}


def test_save_writes_to_store_keyed_by_run_id() -> None:
    """``save`` writes an ``asdict`` serialization into the shared
    modal.Dict under this instance's ``run_id``."""
    tr = _make()
    store: dict[str, Any] = {}

    class _FakeDict:
        @classmethod
        def from_name(cls, name: str, **kwargs: Any) -> "_FakeDict":
            _FakeDict.last_name = name  # type: ignore[attr-defined]
            return cls()

        def __setitem__(self, key: str, value: Any) -> None:
            store[key] = value

    with patch.dict(
        "sys.modules",
        {"modal": MagicMock(Dict=_FakeDict)},
    ):
        tr.save()

    assert _FakeDict.last_name == "demo-train-results"  # type: ignore[attr-defined]
    assert "train_1700000000" in store
    assert store["train_1700000000"]["checkpoint_dir"] == (
        "/checkpoints/demo_train_1700000000"
    )


def test_load_returns_latest_when_run_id_is_none() -> None:
    """``load(app_name)`` picks the lexicographically-largest key."""
    from modal_training_gym.common.train_result import TrainResult

    stored = {
        "train_1700000000": asdict(_make()),
        "train_1800000000": {
            **asdict(_make()),
            "run_id": "train_1800000000",
            "checkpoint_dir": "/checkpoints/demo_train_1800000000",
        },
    }

    class _FakeDict:
        @classmethod
        def from_name(cls, name: str, **kwargs: Any) -> "_FakeDict":
            return cls()

        def keys(self) -> Any:
            return list(stored.keys())

        def __contains__(self, key: str) -> bool:
            return key in stored

        def __getitem__(self, key: str) -> Any:
            return stored[key]

    with patch.dict("sys.modules", {"modal": MagicMock(Dict=_FakeDict)}):
        got = TrainResult.load("demo")
    assert got.run_id == "train_1800000000"
    assert got.checkpoint_dir == "/checkpoints/demo_train_1800000000"


def test_load_raises_when_no_runs_saved() -> None:
    from modal_training_gym.common.train_result import TrainResult

    class _FakeDict:
        @classmethod
        def from_name(cls, name: str, **kwargs: Any) -> "_FakeDict":
            return cls()

        def keys(self) -> Any:
            return []

    with patch.dict("sys.modules", {"modal": MagicMock(Dict=_FakeDict)}):
        try:
            TrainResult.load("demo")
        except LookupError as e:
            assert "demo" in str(e)
        else:
            raise AssertionError("Expected LookupError")


def test_build_serve_app_wires_volume_and_run_id() -> None:
    """``build_serve_app`` forwards the checkpoints volume and picks a
    sensible default ``served_model_name`` that embeds ``run_id``."""
    captured: dict[str, Any] = {}

    def fake_build(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "fake-app"

    tr = _make()
    with patch(
        "modal_training_gym.common.serve_vllm.build_vllm_serve_app",
        side_effect=fake_build,
    ):
        # Explicit checkpoint_path so we don't hit the (un-mocked) volume.
        got = tr.build_serve_app(
            checkpoint_path="/checkpoints/demo_train_1700000000/iter_0000050"
        )

    assert got == "fake-app"
    assert captured["app_name"] == "demo-train_1700000000-serve"
    assert captured["served_model_name"] == "demo-train_1700000000"
    assert captured["checkpoints_volume"] == "demo-checkpoints"
    assert captured["checkpoints_mount_path"] == "/checkpoints"


def test_each_framework_has_train_result_construction_site() -> None:
    """Smoke test: every launcher imports ``TrainResult`` and references
    it in its ``train``/``train_multi_node`` function. Catches the case
    where a launcher edit accidentally drops the publish step.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    launchers = {
        "slime": repo / "modal_training_gym/frameworks/slime/launcher.py",
        "ms-swift": repo / "modal_training_gym/frameworks/ms_swift/launcher.py",
        "miles": repo / "modal_training_gym/frameworks/miles/launcher.py",
        "harbor": repo / "modal_training_gym/frameworks/harbor/launcher.py",
    }
    for fw, path in launchers.items():
        text = path.read_text()
        assert "TrainResult(" in text, f"{fw} launcher never constructs a TrainResult"
        assert "result.save()" in text, (
            f"{fw} launcher constructs but never persists TrainResult"
        )


def _fake_entry(path: str) -> Any:
    entry = MagicMock()
    entry.path = path
    return entry


def main() -> int:
    tests = [
        test_dataclass_round_trip_via_asdict,
        test_latest_checkpoint_path_without_iterations_returns_dir,
        test_latest_checkpoint_path_with_iterations_sorts_lexicographically,
        test_latest_checkpoint_path_raises_when_empty,
        test_volume_always_uses_version_2,
        test_save_writes_to_store_keyed_by_run_id,
        test_load_returns_latest_when_run_id_is_none,
        test_load_raises_when_no_runs_saved,
        test_build_serve_app_wires_volume_and_run_id,
        test_each_framework_has_train_result_construction_site,
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
