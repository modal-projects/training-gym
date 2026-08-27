import modal
import pytest

from modal_training_gym.common import checkpoint as checkpoint_mod
from modal_training_gym.common.checkpoint import (
    Checkpoint,
    CheckpointType,
    convert_megatron_checkpoint_to_hf,
    list_checkpoints,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.train_result import TrainResult


class _CheckpointVolume:
    def __init__(self, names: list[str]):
        self.names = names

    def read_file(self, path: str):
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if name not in self.names:
            raise FileNotFoundError(path)
        return []

    def iterdir(self, *args, **kwargs):
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


def test_list_checkpoints_explains_that_run_must_be_terminal(monkeypatch) -> None:
    def missing_result(training_run_id: str) -> TrainResult:
        raise KeyError(training_run_id)

    monkeypatch.setattr(
        checkpoint_mod.TrainResult,
        "from_training_run_id",
        staticmethod(missing_result),
    )

    with pytest.raises(
        TrainingGymConfigError,
        match=r"Rollout progress is not terminal run completion.*"
        r"TrainingRun\.from_id\('still-running'\)\.result\(\)",
    ):
        list_checkpoints("still-running")


def test_checkpoint_discovery_does_not_create_source_volume(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def from_name(name: str, *, create_if_missing: bool):
        calls.append((name, create_if_missing))
        return _CheckpointVolume([])

    monkeypatch.setattr(checkpoint_mod.Volume, "from_name", from_name)
    monkeypatch.setattr(
        checkpoint_mod.TrainResult,
        "from_training_run_id",
        staticmethod(
            lambda training_run_id: TrainResult(
                app_name="app",
                framework=Framework.SLIME,
                training_run_id=training_run_id,
                checkpoint_dir="/checkpoints/run",
                checkpoints_volume_name="recipe-checkpoints",
                checkpoints_mount_path="/checkpoints",
            )
        ),
    )

    assert list_checkpoints("complete-run") == []
    assert calls == [("recipe-checkpoints", False)]


def test_checkpoint_conversion_does_not_create_source_volume(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []
    volume = _CheckpointVolume([])

    def from_name(name: str, *, create_if_missing: bool):
        calls.append((name, create_if_missing))
        return volume

    monkeypatch.setattr(checkpoint_mod.Volume, "from_name", from_name)
    monkeypatch.setattr(
        modal,
        "App",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("convert")),
    )

    with pytest.raises(RuntimeError, match="convert"):
        convert_megatron_checkpoint_to_hf(
            _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
        )

    assert calls == [
        ("gym-checkpoints", False),
        ("huggingface-cache", True),
        ("gym-checkpoints", False),
    ]


def test_train_result_volume_does_not_create_source_volume(monkeypatch) -> None:
    volume = object()
    calls: list[tuple[str, bool]] = []

    def from_name(name: str, *, create_if_missing: bool):
        calls.append((name, create_if_missing))
        return volume

    monkeypatch.setattr(modal.Volume, "from_name", from_name)
    result = TrainResult(
        app_name="app",
        framework=Framework.SLIME,
        training_run_id="complete-run",
        checkpoints_volume_name="recipe-checkpoints",
    )

    assert result.volume() is volume
    assert calls == [("recipe-checkpoints", False)]


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


def test_convert_reuses_when_marker_present(monkeypatch) -> None:
    _patch_volume(
        monkeypatch,
        _CheckpointVolume(
            ["config.json", "model.safetensors", ".training_gym_convert_complete"]
        ),
    )

    result = convert_megatron_checkpoint_to_hf(
        _megatron_checkpoint(), ModelConfig(model_name="Qwen/Qwen3-4B")
    )

    assert result.checkpoint_type is CheckpointType.hf
    assert result.path == "/checkpoints/run/iter_10_hf"
    assert result.name == "iter_10_hf"


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
