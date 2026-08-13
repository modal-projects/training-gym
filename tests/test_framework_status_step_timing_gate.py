"""Step timing is slime-only: miles emits no step_event markers, so recording
its plain phase updates would misfile them as substep entries.
"""

import pytest

import modal_training_gym.common.run as run_module
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import FrameworkStatusUpdate, TrainingRun


def _run(framework: Framework) -> TrainingRun:
    return TrainingRun(
        training_run_id="run-under-test",
        framework=framework,
        config=None,
    )


@pytest.mark.parametrize(
    ("framework", "phase", "expect_recorded"),
    [
        (Framework.SLIME, "generate_rollouts", True),
        (Framework.MILES, "generate_rollouts", False),
    ],
)
def test_step_times_recorded_only_for_slime(
    monkeypatch, framework, phase, expect_recorded
):
    calls: list[tuple] = []
    monkeypatch.setattr(
        run_module, "record_step_time_event", lambda *a, **kw: calls.append(a)
    )
    monkeypatch.setattr(run_module, "_step_times_dict", dict)

    status = _run(framework).apply_framework_status(
        FrameworkStatusUpdate(
            training_run_id="run-under-test",
            phase=phase,
            progress_current=3,
        )
    )

    assert status is not None
    assert bool(calls) is expect_recorded
