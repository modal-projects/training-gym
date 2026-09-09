from __future__ import annotations

import pytest
from modal.exception import NotFoundError

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun, TrainingRunStatus


def _run(status: TrainingRunStatus) -> TrainingRun:
    return TrainingRun(
        training_run_id="run-1",
        framework=Framework.SLIME,
        config={},
        status=status,
    )


def test_done_is_false_while_running(fake_volume):
    assert _run(TrainingRunStatus.RUNNING).done() is False


@pytest.mark.parametrize(
    "status",
    [
        TrainingRunStatus.COMPLETED,
        TrainingRunStatus.FAILED,
        TrainingRunStatus.STOPPED,
        TrainingRunStatus.CANCELLED,
    ],
)
def test_done_is_true_for_terminal_status(status: TrainingRunStatus, fake_volume):
    assert _run(status).done() is True


def test_done_sees_status_written_after_launch(fake_volume):
    live = _run(TrainingRunStatus.RUNNING)
    _run(TrainingRunStatus.COMPLETED).save()
    handle = id(live)

    assert live.done() is True
    assert id(live) == handle
    assert live.status is TrainingRunStatus.COMPLETED


@pytest.mark.parametrize("exc", [KeyError("run-1"), NotFoundError("run-1")])
def test_reload_swallows_missing_run(exc: Exception, monkeypatch, fake_volume):
    live = _run(TrainingRunStatus.RUNNING)

    def missing(_run_id: str, **_kwargs):
        raise exc

    monkeypatch.setattr(TrainingRun, "from_id", missing)
    assert live.done() is False
    assert live.status is TrainingRunStatus.RUNNING


def test_reload_reraises_non_miss_errors(monkeypatch, fake_volume):
    live = _run(TrainingRunStatus.RUNNING)

    def boom(_run_id: str, **_kwargs):
        raise RuntimeError("corrupt metadata")

    monkeypatch.setattr(TrainingRun, "from_id", boom)
    with pytest.raises(RuntimeError, match="corrupt metadata"):
        live.done()


def test_context_manager_stops_app(monkeypatch):
    stopped: list[str] = []
    monkeypatch.setattr(
        "modal_training_gym.common.modal_lifecycle.stop_app", stopped.append
    )
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"

    with run as entered:
        assert entered is run
        assert stopped == []

    assert stopped == ["ap-1"]


def test_exit_closes_after_exception(monkeypatch):
    stopped: list[str] = []
    monkeypatch.setattr(
        "modal_training_gym.common.modal_lifecycle.stop_app", stopped.append
    )
    run = _run(TrainingRunStatus.RUNNING)
    run.modal_app_id = "ap-1"

    with pytest.raises(RuntimeError, match="boom"):
        with run:
            raise RuntimeError("boom")

    assert stopped == ["ap-1"]


def test_close_is_idempotent(monkeypatch):
    stopped: list[str] = []
    monkeypatch.setattr(
        "modal_training_gym.common.modal_lifecycle.stop_app", stopped.append
    )
    run = _run(TrainingRunStatus.COMPLETED)
    run.modal_app_id = "ap-1"

    run.close()
    run.close()

    assert stopped == ["ap-1"]


def test_done_does_not_stop_app(monkeypatch, fake_volume):
    stopped: list[str] = []
    monkeypatch.setattr(
        "modal_training_gym.common.modal_lifecycle.stop_app", stopped.append
    )
    run = _run(TrainingRunStatus.COMPLETED)
    run.modal_app_id = "ap-1"

    assert run.done() is True
    assert stopped == []
