import importlib.util
import sys
import types
from pathlib import Path

import modal
import pytest

from modal_training_gym.common import checkpoint as checkpoint_mod
from modal_training_gym.common import torch_dist_checkpoint as torch_dist
from modal_training_gym.common.megatron_patches import patch_checkpoint_commit
from modal_training_gym.common.checkpoint import (
    Checkpoint,
    CheckpointType,
    convert_megatron_checkpoint_to_hf,
    volume_relative_path,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.launcher_helpers import compute_save_root
from modal_training_gym.common.models import ModelConfig


class _CheckpointVolume:
    def __init__(self, names: list[str]):
        self.names = names

    def read_file(self, path: str):
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if name not in self.names:
            raise FileNotFoundError(path)
        return []


def _megatron_checkpoint() -> Checkpoint:
    return Checkpoint(
        checkpoint_type=CheckpointType.megatron,
        name="iter_10",
        path="/checkpoints/run/iter_10",
        timestamp=1.0,
        training_run_id="run",
        app_name="app",
        checkpoints_volume_name="gym-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )


def _patch_volume(monkeypatch, volume: _CheckpointVolume) -> None:
    monkeypatch.setattr(
        checkpoint_mod.Volume, "from_name", lambda *args, **kwargs: volume
    )


def test_convert_megatron_checkpoint_to_hf_returns_hf_checkpoints_unchanged() -> None:
    checkpoint = Checkpoint(
        checkpoint_type=CheckpointType.hf,
        name="iter_0000010_hf",
        path="/checkpoints/run/iter_0000010_hf",
        timestamp=1.0,
        checkpoints_volume_name="gym-checkpoints",
    )

    result = convert_megatron_checkpoint_to_hf(
        checkpoint, ModelConfig(model_name="Qwen/Qwen3-4B")
    )

    assert result is checkpoint


def test_save_root_must_stay_inside_checkpoint_volume() -> None:
    with pytest.raises(TrainingGymConfigError, match="outside Volume mount"):
        compute_save_root(
            "/tmp/checkpoints",
            recipe_default_save_root="/checkpoints",
            mounted_save_root="/checkpoints",
            training_run_id="run-1",
        )


@pytest.mark.parametrize("training_run_id", ["../outside", "/tmp/outside"])
def test_training_run_id_cannot_escape_checkpoint_volume(
    training_run_id: str,
) -> None:
    with pytest.raises(TrainingGymConfigError, match="outside Volume mount"):
        compute_save_root(
            "/checkpoints",
            recipe_default_save_root="/checkpoints",
            mounted_save_root="/checkpoints",
            training_run_id=training_run_id,
        )


def test_relative_checkpoint_path_is_rejected() -> None:
    with pytest.raises(TrainingGymConfigError, match="must be absolute POSIX paths"):
        volume_relative_path("run/iter_10", "/checkpoints")


def test_convert_reuses_when_marker_present(monkeypatch) -> None:
    _patch_volume(
        monkeypatch,
        _CheckpointVolume(
            [
                "config.json",
                "model.safetensors",
                checkpoint_mod._CONVERT_COMPLETE_MARKER,
            ]
        ),
    )

    result = convert_megatron_checkpoint_to_hf(
        _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
    )

    assert result.checkpoint_type is CheckpointType.hf
    assert result.path == "/checkpoints/run/iter_10_hf"
    assert result.name == "iter_10_hf"


def test_convert_requires_existing_checkpoint_volume(monkeypatch) -> None:
    lookups: list[tuple[str, bool]] = []

    def lookup(name: str, *, create_if_missing: bool):
        lookups.append((name, create_if_missing))
        raise RuntimeError("missing-volume")

    monkeypatch.setattr(checkpoint_mod.Volume, "from_name", lookup)

    with pytest.raises(RuntimeError, match="missing-volume"):
        convert_megatron_checkpoint_to_hf(
            _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
        )

    assert lookups == [("gym-checkpoints", False)]


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["config.json", "model.safetensors"],
    ],
    ids=["missing", "unmarked"],
)
def test_convert_runs_without_marker(monkeypatch, names: list[str]) -> None:
    _patch_volume(monkeypatch, _CheckpointVolume(names))
    monkeypatch.setattr(
        modal,
        "App",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("convert")),
    )

    with pytest.raises(RuntimeError, match="convert"):
        convert_megatron_checkpoint_to_hf(
            _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
        )


def _apply_commit_wraps(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    checkpointing = tmp_path / "checkpointing.py"
    checkpointing.write_text("def save_checkpoint(iteration):\n    return iteration\n")
    async_utils = tmp_path / "async_utils.py"
    async_utils.write_text(
        "def is_empty_async_queue():\n"
        "    return True\n"
        "def maybe_finalize_async_save(blocking=False):\n"
        "    return blocking\n"
    )
    monkeypatch.setattr(patch_checkpoint_commit, "_CHECKPOINTING", checkpointing)
    monkeypatch.setattr(patch_checkpoint_commit, "_ASYNC_UTILS", async_utils)
    patch_checkpoint_commit.main()
    return checkpointing, async_utils


def _load_patched(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_commit_wraps_save_and_async_finalize_once(
    monkeypatch, tmp_path: Path
) -> None:
    checkpointing, async_utils = _apply_commit_wraps(monkeypatch, tmp_path)
    patched_save = checkpointing.read_text()
    patched_finalize = async_utils.read_text()

    compile(patched_save, str(checkpointing), "exec")
    compile(patched_finalize, str(async_utils), "exec")
    assert patched_save.count(patch_checkpoint_commit._SAVE_MARKER) == 1
    assert 'if not getattr(get_args(), "async_save", False):' in patched_save
    assert "commit_latest_checkpoint_volume()" in patched_save
    assert patched_finalize.count(patch_checkpoint_commit._FINALIZE_MARKER) == 1
    assert "had_pending = not is_empty_async_queue()" in patched_finalize
    assert "if had_pending:" in patched_finalize
    assert "commit_latest_checkpoint_volume()" in patched_finalize

    patch_checkpoint_commit.main()

    assert checkpointing.read_text() == patched_save
    assert async_utils.read_text() == patched_finalize


def test_save_wrap_commits_only_for_sync_save(monkeypatch, tmp_path: Path) -> None:
    checkpointing, _ = _apply_commit_wraps(monkeypatch, tmp_path)
    events: list[str] = []
    args = types.SimpleNamespace(async_save=True)
    megatron = types.ModuleType("megatron")
    megatron_training = types.ModuleType("megatron.training")
    megatron_training.get_args = lambda: args
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", megatron_training)
    monkeypatch.setattr(
        torch_dist, "commit_latest_checkpoint_volume", lambda: events.append("commit")
    )

    save_mod = _load_patched(checkpointing, "patched_save_checkpoint")
    assert save_mod.save_checkpoint(1) == 1
    assert events == []

    args.async_save = False
    assert save_mod.save_checkpoint(2) == 2
    assert events == ["commit"]


def test_save_wrap_does_not_swallow_get_args_failure(
    monkeypatch, tmp_path: Path
) -> None:
    checkpointing, _ = _apply_commit_wraps(monkeypatch, tmp_path)
    megatron = types.ModuleType("megatron")
    megatron_training = types.ModuleType("megatron.training")

    def boom():
        raise RuntimeError("args unset")

    megatron_training.get_args = boom
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.training", megatron_training)

    save_mod = _load_patched(checkpointing, "patched_save_checkpoint_args")
    with pytest.raises(RuntimeError, match="args unset"):
        save_mod.save_checkpoint(1)


def test_finalize_wrap_skips_commit_when_queue_empty(
    monkeypatch, tmp_path: Path
) -> None:
    _, async_utils = _apply_commit_wraps(monkeypatch, tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        torch_dist, "commit_latest_checkpoint_volume", lambda: events.append("commit")
    )

    finalize_mod = _load_patched(async_utils, "patched_finalize_empty")
    assert finalize_mod.maybe_finalize_async_save(True) is True
    assert events == []


def test_finalize_wrap_commits_when_queue_had_work(monkeypatch, tmp_path: Path) -> None:
    _, async_utils = _apply_commit_wraps(monkeypatch, tmp_path)
    events: list[str] = []
    monkeypatch.setattr(
        torch_dist, "commit_latest_checkpoint_volume", lambda: events.append("commit")
    )

    finalize_mod = _load_patched(async_utils, "patched_finalize_pending")
    finalize_mod.is_empty_async_queue = lambda: False
    assert finalize_mod.maybe_finalize_async_save(True) is True
    assert events == ["commit"]
