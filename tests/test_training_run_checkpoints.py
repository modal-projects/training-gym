from __future__ import annotations

from dataclasses import dataclass

import pytest

import modal_training_gym.common.checkpoint as checkpoint_mod
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.launcher_helpers import init_training_run_record
from modal_training_gym.common.models import Qwen3_5_4B
from modal_training_gym.common.run import (
    CHECKPOINT_LOCATION_METADATA_KEY,
    TrainingRun,
    set_checkpoint_location,
)
from modal_training_gym.common.train_result import TrainResult


@dataclass
class _DirEntry:
    path: str
    mtime: float = 1.0
    is_directory: bool = False

    def is_dir(self) -> bool:
        return self.is_directory


class _ListingVolume:
    def __init__(
        self,
        tree: dict[str, list[_DirEntry]],
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.tree = tree
        self.files = files or {}

    def iterdir(self, path: str, *, recursive: bool = False):
        del recursive
        key = path.rstrip("/") or "."
        return list(self.tree.get(key, []))

    def read_file(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]


def _complete_iter_files(name: str) -> list[_DirEntry]:
    child_key = f"run-1/{name}"
    return [
        _DirEntry(f"{child_key}/common.pt"),
        _DirEntry(f"{child_key}/shard.distcp"),
        _DirEntry(f"{child_key}/.metadata"),
    ]


def _complete_hf_files(name: str) -> list[_DirEntry]:
    return [
        _DirEntry(f"run-1/{name}/config.json"),
        _DirEntry(f"run-1/{name}/model.safetensors"),
        _DirEntry(f"run-1/{name}/{checkpoint_mod._CONVERT_COMPLETE_MARKER}"),
    ]


def _run_with_checkpoint_location() -> TrainingRun:
    run = TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
    )
    set_checkpoint_location(
        run,
        checkpoint_dir="/checkpoints/run-1",
        checkpoints_volume_name="slime-slime4brecipe-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )
    return run


def _train_result() -> TrainResult:
    return TrainResult(
        app_name="slime",
        framework=Framework.SLIME,
        training_run_id="run-1",
        checkpoint_dir="/checkpoints/run-1",
        checkpoints_volume_name="slime-slime4brecipe-checkpoints",
        checkpoints_mount_path="/checkpoints",
        model_config=Qwen3_5_4B(),
    )


def test_live_run_lists_complete_iter_at_or_before_tracker(
    monkeypatch, fake_volume
) -> None:
    run = _run_with_checkpoint_location()
    volume = _ListingVolume(
        {
            "run-1": [
                _DirEntry("iter_0000001", is_directory=True),
                _DirEntry("iter_0000002", is_directory=True),
                _DirEntry("iter_0000003", is_directory=True),
                _DirEntry("iter_0000004_hf", is_directory=True),
            ],
            "run-1/iter_0000001": _complete_iter_files("iter_0000001"),
            "run-1/iter_0000002": [
                _DirEntry("run-1/iter_0000002/common.pt"),
                _DirEntry("run-1/iter_0000002/shard.distcp"),
            ],
            "run-1/iter_0000003": _complete_iter_files("iter_0000003"),
            "run-1/iter_0000004_hf": _complete_hf_files("iter_0000004_hf"),
        },
        files={"run-1/latest_checkpointed_iteration.txt": b"1\n"},
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )

    assert [checkpoint.name for checkpoint in run.checkpoints()] == ["iter_0000001"]
    assert run.latest_checkpoint() is not None
    assert run.latest_checkpoint().name == "iter_0000001"


def test_live_run_hides_checkpoints_without_tracker(monkeypatch, fake_volume) -> None:
    run = _run_with_checkpoint_location()
    volume = _ListingVolume(
        {
            "run-1": [_DirEntry("iter_0000001", is_directory=True)],
            "run-1/iter_0000001": _complete_iter_files("iter_0000001"),
        }
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )

    assert run.checkpoints() == []


def test_train_result_lists_complete_dirs_without_tracker(monkeypatch) -> None:
    volume = _ListingVolume(
        {
            "run-1": [_DirEntry("iter_0000001", is_directory=True)],
            "run-1/iter_0000001": _complete_iter_files("iter_0000001"),
        }
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )

    assert [checkpoint.name for checkpoint in _train_result().checkpoints()] == [
        "iter_0000001"
    ]


def test_train_result_lists_hf_and_latest_checkpoint_stays_megatron(
    monkeypatch,
) -> None:
    volume = _ListingVolume(
        {
            "run-1": [
                _DirEntry("iter_0000001", is_directory=True),
                _DirEntry("iter_0000001_hf", is_directory=True),
                _DirEntry("iter_0000002_hf", is_directory=True),
            ],
            "run-1/iter_0000001": _complete_iter_files("iter_0000001"),
            "run-1/iter_0000001_hf": _complete_hf_files("iter_0000001_hf"),
            "run-1/iter_0000002_hf": [
                _DirEntry("run-1/iter_0000002_hf/model.safetensors")
            ],
        }
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )
    result = _train_result()

    checkpoints = result.checkpoints()

    assert [
        (checkpoint.name, checkpoint.checkpoint_type) for checkpoint in checkpoints
    ] == [
        ("iter_0000001", checkpoint_mod.CheckpointType.megatron),
        ("iter_0000001_hf", checkpoint_mod.CheckpointType.hf),
    ]
    assert result.latest_checkpoint() is not None
    assert result.latest_checkpoint().name == "iter_0000001"
    assert result.model.model_path == "/checkpoints/run-1/iter_0000001"


def test_checkpoint_reads_do_not_create_missing_volume(
    monkeypatch, fake_volume
) -> None:
    calls: list[bool] = []

    def _missing_volume(_name: str, *, create_if_missing: bool):
        calls.append(create_if_missing)
        raise RuntimeError("missing-volume")

    monkeypatch.setattr(checkpoint_mod.Volume, "from_name", _missing_volume)

    with pytest.raises(RuntimeError, match="missing-volume"):
        _run_with_checkpoint_location().checkpoints()

    result = TrainResult(
        app_name="slime",
        framework=Framework.SLIME,
        training_run_id="run-1",
        checkpoints_volume_name="checkpoints",
    )

    with pytest.raises(RuntimeError, match="missing-volume"):
        result.volume()

    assert calls == [False, False]


def test_listing_without_checkpoint_location_is_empty(fake_volume) -> None:
    run = TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
    )

    assert run.checkpoints() == []


def test_latest_checkpoint_sees_location_written_after_launch(
    monkeypatch, fake_volume
) -> None:
    live = TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
    )
    stored = _run_with_checkpoint_location()
    stored.save()
    volume = _ListingVolume(
        {
            "run-1": [_DirEntry("iter_0000001", is_directory=True)],
            "run-1/iter_0000001": _complete_iter_files("iter_0000001"),
        },
        files={"run-1/latest_checkpointed_iteration.txt": b"1\n"},
    )
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )
    handle = id(live)

    checkpoint = live.latest_checkpoint()

    assert id(live) == handle
    assert checkpoint is not None
    assert checkpoint.name == "iter_0000001"


@pytest.mark.anyio
async def test_direct_train_initialization_persists_checkpoint_location(
    fake_volume,
) -> None:
    run, _, _ = await init_training_run_record(
        training_run_id="run-1",
        modal_app_id="ap-test",
        modal_app_url="https://modal.com/apps/test",
        framework=Framework.SLIME,
        initializing_status=None,
        config_summary={},
        metric_cfg=None,
        metric_entity="",
        framework_status_token="token",
        checkpoint_dir="/checkpoints/run-1",
        checkpoints_volume_name="slime-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )

    stored = TrainingRun.from_id("run-1")
    expected = {
        "checkpoint_dir": "/checkpoints/run-1",
        "checkpoints_volume_name": "slime-checkpoints",
        "checkpoints_mount_path": "/checkpoints",
    }

    assert (run.metadata or {})[CHECKPOINT_LOCATION_METADATA_KEY] == expected
    assert (stored.metadata or {})[CHECKPOINT_LOCATION_METADATA_KEY] == expected
    assert "checkpoint_dir" not in stored.model_dump()
