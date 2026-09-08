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

    def add_local_dir(self, *args: object, **kwargs: object) -> RecordingImage:
        self.operations.append(("add_local_dir", args, kwargs))
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


def test_mount_caller_source_flattens_tutorials_main(tmp_path: Path) -> None:
    main = tmp_path / "tutorials" / "main.py"
    main.parent.mkdir()
    main.write_text("VALUE = 1\n")

    image = mount_caller_source(RecordingImage(), str(main))

    assert image.operations == [
        (
            "add_local_file",
            (str(main),),
            {"remote_path": "/root/main.py", "copy": True},
        )
    ]


def test_mount_caller_source_skips_when_caller_script_is_none() -> None:
    image = mount_caller_source(RecordingImage(), None)

    assert image.operations == []


def test_mount_caller_source_ships_tutorial_package(tmp_path: Path) -> None:
    package = tmp_path / "tutorials" / "nested"
    package.mkdir(parents=True)
    main = package / "main.py"
    main.write_text("from .helper import score\n")
    (package / "helper.py").write_text("def score() -> int:\n    return 1\n")

    image = mount_caller_source(RecordingImage(), str(main))

    assert image.operations == [
        (
            "add_local_dir",
            (package,),
            {
                "remote_path": "/root/tutorials/nested",
                "copy": True,
                "ignore": ["**/__pycache__", "**/*.pyc"],
            },
        )
    ]


def test_mount_caller_source_ships_tutorial_package_from_helper(tmp_path: Path) -> None:
    package = tmp_path / "tutorials" / "nested"
    package.mkdir(parents=True)
    (package / "main.py").write_text("from .helper import score\n")
    helper = package / "helper.py"
    helper.write_text("def score() -> int:\n    return 1\n")

    image = mount_caller_source(RecordingImage(), str(helper))

    assert image.operations == [
        (
            "add_local_dir",
            (package,),
            {
                "remote_path": "/root/tutorials/nested",
                "copy": True,
                "ignore": ["**/__pycache__", "**/*.pyc"],
            },
        )
    ]


def test_ship_callable_uses_package_path_when_caller_is_helper(tmp_path: Path) -> None:
    package = tmp_path / "tutorials" / "nested"
    package.mkdir(parents=True)
    main = package / "main.py"
    main.write_text("def train() -> None:\n    return None\n")
    helper = package / "helper.py"
    helper.write_text("def score() -> int:\n    return 1\n")
    module = _load_module(main, "tutorials.nested.main")
    paths: list[str] = []

    image = ship_callable(
        RecordingImage(),
        module.train,
        caller_script=str(helper),
        fallback_name="train",
        set_path=paths.append,
    )

    assert paths == ["tutorials.nested.main.train"]
    assert image.operations == []


def test_ship_callable_uses_package_path_for_sibling(tmp_path: Path) -> None:
    package = tmp_path / "tutorials" / "nested"
    package.mkdir(parents=True)
    main = package / "main.py"
    main.write_text("")
    helper = package / "helper.py"
    helper.write_text("def score() -> int:\n    return 1\n")
    module = _load_module(helper, "tutorials.nested.helper")
    paths: list[str] = []

    image = ship_callable(
        RecordingImage(),
        module.score,
        caller_script=str(main),
        fallback_name="score",
        set_path=paths.append,
    )

    assert paths == ["tutorials.nested.helper.score"]
    assert image.operations == []


def test_ship_callable_still_flattens_unrelated_file(tmp_path: Path) -> None:
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
