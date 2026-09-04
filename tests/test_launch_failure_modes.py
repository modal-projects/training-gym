"""Launches that go sideways should say what happened."""

from __future__ import annotations

from pathlib import Path


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
