from modal_training_gym.common.substep_timing import (
    RoleTiming,
    TimingCaptureStatus,
    TimingInterval,
    TimingPhase,
    TimingRole,
    aggregate_timing_intervals,
)
from modal_training_gym.common.training_rollout import select_rollout_summaries
from modal_training_gym.frameworks.slime.substep_timing import StepTimingCollector


def test_aggregate_timing_intervals_tracks_active_time() -> None:
    intervals = [
        TimingInterval(TimingPhase.CUSTOM_REWARD, 10.0, 1.0, 4.0),
        TimingInterval(TimingPhase.CUSTOM_REWARD, 11.0, 2.0, 5.0),
        TimingInterval(TimingPhase.CUSTOM_REWARD, 15.0, 8.0, 9.0),
    ]

    timing = aggregate_timing_intervals(intervals)[0]

    assert timing.duration_s == 5.0
    assert timing.count == 3


def test_collector_replaces_a_retried_role_before_step_closes() -> None:
    collector = StepTimingCollector()
    first = RoleTiming(
        role=TimingRole.ACTOR,
        status=TimingCaptureStatus.UNAVAILABLE,
    )
    retried = RoleTiming(
        role=TimingRole.ACTOR,
        status=TimingCaptureStatus.NOT_RUN,
    )

    assert collector.record_role_timing(4, first.model_dump_json())
    assert collector.record_role_timing(4, retried.model_dump_json())
    assert (
        collector.read_step_timings(4)[TimingRole.ACTOR.value]
        == retried.model_dump_json()
    )

    collector.close_step(4)

    assert not collector.record_role_timing(4, first.model_dump_json())
    assert collector.read_step_timings(4) == {}


def test_rollout_selection_uses_retry_boundaries_and_legacy_prefix() -> None:
    summaries = [
        {"rollout_id": 0},
        {"rollout_id": 1},
        {"rollout_id": 2, "training_attempt": 1},
        {"rollout_id": 2, "training_attempt": 2},
        {"rollout_id": 3, "training_attempt": 2},
    ]

    selected = select_rollout_summaries(summaries, [(2, 2, 4)])

    assert [(row["rollout_id"], row.get("training_attempt")) for row in selected] == [
        (0, None),
        (1, None),
        (2, 2),
        (3, 2),
    ]
