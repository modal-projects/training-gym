"""Launches that go sideways should say what happened."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from modal_training_gym.common import launcher_helpers


class _FakeVolume:
    def __init__(self, name: str, *, exists: bool, created: bool = False) -> None:
        self.name = name
        self._exists = exists
        self.created = created

    def hydrate(self) -> None:
        if not self._exists:
            raise RuntimeError("not found")


def _fake_volume_class(monkeypatch: pytest.MonkeyPatch, *, exists: bool) -> dict:
    calls: dict[str, Any] = {"created": False}

    class _Volume:
        @staticmethod
        def from_name(name: str, create_if_missing: bool = False) -> _FakeVolume:
            if create_if_missing:
                calls["created"] = True
            return _FakeVolume(name, exists=exists)

    monkeypatch.setattr(launcher_helpers, "Volume", _Volume)
    return calls


def test_existing_volume_is_reused_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _fake_volume_class(monkeypatch, exists=True)

    volume = launcher_helpers.volume_for("slime-qwen3_6_27b_recipe-checkpoints")

    assert volume.name == "slime-qwen3_6_27b_recipe-checkpoints"
    assert calls["created"] is False
    assert capsys.readouterr().out == ""


def test_creating_a_volume_says_so_and_names_the_rename_trap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Volume names derive from the recipe class, so a rename starts over empty.

    That silently orphans a cached conversion — tens of GB and real GPU time —
    so creating a volume has to be visible rather than implicit.
    """
    calls = _fake_volume_class(monkeypatch, exists=False)

    launcher_helpers.volume_for("slime-qwen3_6_27b_recipe_agentic-checkpoints")

    assert calls["created"] is True
    out = capsys.readouterr().out
    assert "Creating Modal Volume" in out
    assert "renamed" in out


def test_a_launch_that_never_spawns_does_not_raise_unboundlocalerror() -> None:
    """A detached app whose client disconnects exits `app.run()` without raising.

    The spawn is then skipped, and dereferencing `function_call` afterwards used
    to surface as `UnboundLocalError`, hiding the real cause entirely.
    """
    source = Path(
        Path(__file__).parents[1] / "modal_training_gym/common/train.py"
    ).read_text()

    assert "function_call = None" in source
    guard = source.index("if function_call is None:")
    assert guard < source.index("run_record.function_call_id = function_call.object_id")
    assert "lost its connection" in source[guard : guard + 800]
