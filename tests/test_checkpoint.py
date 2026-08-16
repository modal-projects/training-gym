from types import SimpleNamespace

import modal
import pytest

from modal_training_gym.common import checkpoint as checkpoint_mod
from modal_training_gym.common.checkpoint import (
    Checkpoint,
    CheckpointType,
    convert_megatron_checkpoint_to_hf,
)
from modal_training_gym.common.models import ModelConfig


class _CheckpointVolume:
    def __init__(self, names: list[str]):
        self.names = names

    def iterdir(self, path: str, recursive: bool = False):
        return [SimpleNamespace(path=name) for name in self.names]


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


def test_convert_reuses_a_complete_hf_sibling(monkeypatch) -> None:
    _patch_volume(
        monkeypatch,
        _CheckpointVolume(["config.json", "model.safetensors"]),
    )

    result = convert_megatron_checkpoint_to_hf(
        _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
    )

    assert result.checkpoint_type is CheckpointType.hf
    assert result.path == "/checkpoints/run/iter_10_hf"
    assert result.name == "iter_10_hf"


def test_convert_runs_when_the_hf_sibling_is_incomplete(monkeypatch) -> None:
    _patch_volume(monkeypatch, _CheckpointVolume(["config.json"]))
    monkeypatch.setattr(
        modal,
        "App",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("convert")),
    )

    with pytest.raises(RuntimeError, match="convert"):
        convert_megatron_checkpoint_to_hf(
            _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
        )
