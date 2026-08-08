from __future__ import annotations

from modal_training_gym.common import step_timing


def _phase(start: float, end: float, *, invocations=None) -> dict:
    return {
        "count": 1,
        "total_duration_s": end - start,
        "longest_duration_s": end - start,
        "first_start_s": start,
        "last_end_s": end,
        "invocations": invocations or [[start, end]],
    }


def test_measured_run_times_excludes_not_in_step_phases(monkeypatch):
    monkeypatch.setattr(
        step_timing,
        "load_run",
        lambda _run_id: {
            0: [
                {
                    "role": "driver",
                    "lane_start_unix_s": 100.0,
                    "phases": {
                        "train_models": _phase(10.0, 20.0),
                        "evaluate_rollouts": _phase(0.0, 5.0),
                        "checkpoint_save": _phase(20.0, 23.0),
                    },
                }
            ]
        },
    )

    step_times, _substep_times = step_timing.measured_run_times("run")

    assert step_times == {"1": {"duration_s": 10}}
