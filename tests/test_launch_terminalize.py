from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modal_training_gym.common.errors import TrainingGymError
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.train import TrainConfig


@pytest.fixture
def launch(monkeypatch):
    saved = []
    stop = Mock()
    convert = Mock()
    app = SimpleNamespace(
        name="test-app",
        app_id="ap-test",
        download=SimpleNamespace(remote=Mock()),
        train=SimpleNamespace(
            spawn=Mock(return_value=SimpleNamespace(object_id="fc-test"))
        ),
    )

    @contextmanager
    def run(**kwargs):
        try:
            yield app
        except (KeyboardInterrupt, ConnectionError):
            pass

    app.run = run
    config = SimpleNamespace(
        framework="miles",
        recipe=SimpleNamespace(megatron_to_hf_mode=""),
        _generate_training_run_id=lambda: "run-test",
        _build_status_display=lambda *args: Mock(),
        _build_config_summary=lambda *args: {},
        _initializing_status=lambda: None,
        _build_run_metadata=lambda: {},
        _build_app=Mock(return_value=app),
    )
    monkeypatch.setattr(TrainingRun, "save", lambda self: saved.append(self))
    monkeypatch.setattr(
        "modal_training_gym.cli.setup.ensure_dashboard_deployed", lambda: None
    )
    monkeypatch.setattr(
        "modal_training_gym.common.config.get_framework_status_url", lambda: ""
    )
    monkeypatch.setattr("modal_training_gym.common.train.vol_put", lambda *args: None)
    monkeypatch.setattr(
        "modal_training_gym.common.status_reporter.enqueue_framework_status",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("modal_training_gym.common.modal_lifecycle.stop_app", stop)
    monkeypatch.setattr(
        "modal_training_gym.common.train._convert_checkpoint_on_cache_miss", convert
    )
    return SimpleNamespace(
        call=lambda: TrainConfig.launch(config, show_output=False),
        config=config,
        app=app,
        saved=saved,
        stop=stop,
        stages={
            "download": app.download.remote,
            "conversion": convert,
            "spawn": app.train.spawn,
        },
    )


def test_success(launch):
    run = launch.call()
    assert run.function_call_id == "fc-test"
    launch.stop.assert_not_called()


@pytest.mark.parametrize("stage", ["download", "conversion", "spawn"])
@pytest.mark.parametrize("error_type", [RuntimeError, ConnectionError])
def test_setup_failure_stops_app_and_terminalizes(launch, stage, error_type):
    error = error_type("setup failed")
    target = launch.stages[stage]
    target.side_effect = error
    with pytest.raises(TrainingGymError) as caught:
        launch.call()
    assert str(caught.value) == (
        'Setup of Modal app "test-app" was interrupted before training could begin. '
        "Relaunch your TrainConfig to try again."
    )
    assert caught.value.__cause__ is error
    launch.stop.assert_called_once_with("ap-test")
    run = launch.saved[-1]
    assert run.status is TrainingRunStatus.FAILED
    assert run.completed_at == run.ended_at
    assert run.duration_seconds >= 0


@pytest.mark.parametrize("stage", ["download", "conversion", "spawn"])
def test_keyboard_interrupt_preserves_app(launch, stage, capsys):
    target = launch.stages[stage]
    target.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        launch.call()
    launch.stop.assert_not_called()
    assert launch.saved[-1].status is TrainingRunStatus.RUNNING
    assert launch.saved[-1].ended_at is None
    assert "The app was left running." in capsys.readouterr().out


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("build failed"), TrainingRunStatus.FAILED),
        (KeyboardInterrupt(), TrainingRunStatus.STOPPED),
    ],
)
def test_build_failure_terminalizes(launch, error, status):
    launch.config._build_app.side_effect = error
    with pytest.raises(type(error)):
        launch.call()
    launch.stop.assert_not_called()
    assert launch.saved[-1].status is status
    assert launch.saved[-1].completed_at is not None


def test_context_entry_failure_stops_app(launch):
    @contextmanager
    def run(**kwargs):
        raise RuntimeError("entry failed")
        yield

    launch.app.run = run
    with pytest.raises(TrainingGymError):
        launch.call()
    launch.stop.assert_called_once_with("ap-test")
    assert launch.saved[-1].status is TrainingRunStatus.FAILED


def test_terminal_save_failure_preserves_launch_error(launch, monkeypatch, capsys):
    def save(run):
        if run.status is TrainingRunStatus.FAILED:
            raise OSError("metadata unavailable")

    monkeypatch.setattr(TrainingRun, "save", save)
    launch.app.download.remote.side_effect = RuntimeError("download failed")
    with pytest.raises(TrainingGymError):
        launch.call()
    launch.stop.assert_called_once_with("ap-test")
    assert "could not save run" in capsys.readouterr().out
