from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from modal_training_gym.common.launcher_helpers import (
    mount_caller_source,
    ship_callable,
)


class RecordingImage:
    def __init__(self) -> None:
        self.operations: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def add_local_file(self, *args: object, **kwargs: object) -> RecordingImage:
        self.operations.append(("add_local_file", args, kwargs))
        return self


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mount_caller_source_flattens_flat_script(tmp_path: Path) -> None:
    script = tmp_path / "flat.py"
    script.write_text("VALUE = 1\n")

    image = mount_caller_source(RecordingImage(), str(script))

    assert image.operations == [
        (
            "add_local_file",
            (str(script),),
            {"remote_path": "/root/flat.py", "copy": True},
        )
    ]


def test_mount_caller_source_skips_when_caller_script_is_none() -> None:
    image = mount_caller_source(RecordingImage(), None)

    assert image.operations == []


def test_ship_callable_flattens_sibling_file(tmp_path: Path) -> None:
    caller = tmp_path / "caller.py"
    caller.write_text("")
    helper = tmp_path / "reward.py"
    helper.write_text("def score() -> int:\n    return 1\n")
    module = _load_module(helper, "reward")
    paths: list[str] = []

    image = ship_callable(
        RecordingImage(),
        module.score,
        caller_script=str(caller),
        fallback_name="score",
        set_path=paths.append,
    )

    assert paths == ["reward.score"]
    assert image.operations == [
        (
            "add_local_file",
            (str(helper),),
            {"remote_path": "/root/reward.py", "copy": True},
        )
    ]
