"""``launch()`` must not leave a ``running`` record behind when the Modal app never starts."""

from __future__ import annotations

from modal_training_gym.common.run import TrainingRun, TrainingRunStatus
from modal_training_gym.common.train import _terminalize_unlaunched_run


def _record(monkeypatch, saved: list[TrainingRun]) -> TrainingRun:
    monkeypatch.setattr(TrainingRun, "save", lambda self: saved.append(self))
    return TrainingRun(
        training_run_id="run-1",
        framework="miles",
        config={},
        created_at=100,
        started_at=100,
    )


def test_exception_marks_run_failed(monkeypatch):
    saved: list[TrainingRun] = []
    run = _record(monkeypatch, saved)

    _terminalize_unlaunched_run(run, RuntimeError("image build failed"))

    assert saved == [run]
    assert run.status is TrainingRunStatus.FAILED
    assert run.error_message == "RuntimeError: image build failed"
    assert run.ended_at is not None and run.completed_at == run.ended_at
    assert run.duration_seconds == run.ended_at - 100
    assert run.metadata == {"terminal_reason": "launch_failed_before_modal_app"}


def test_keyboard_interrupt_marks_run_stopped(monkeypatch):
    saved: list[TrainingRun] = []
    run = _record(monkeypatch, saved)

    _terminalize_unlaunched_run(run, KeyboardInterrupt())

    assert saved == [run]
    assert run.status is TrainingRunStatus.STOPPED
    assert run.error_message is None
    assert run.metadata == {"terminal_reason": "launch_interrupted"}


def test_save_failure_is_swallowed(monkeypatch, capsys):
    run = _record(monkeypatch, [])

    def _boom(self):
        raise RuntimeError("volume unavailable")

    monkeypatch.setattr(TrainingRun, "save", _boom)

    _terminalize_unlaunched_run(run, RuntimeError("nope"))

    assert run.status is TrainingRunStatus.FAILED
    assert "could not mark run-1 as failed" in capsys.readouterr().out
