from modal_training_gym.common.substep_timing import (
    RoleTiming,
    TimelineGroup,
    TimingActivityKind,
    TimingCaptureStatus,
    TimingInterval,
    TimingPhase,
    TimingRole,
    aggregate_timing_intervals,
)
from modal_training_gym.common.training_rollout import select_rollout_summaries
from modal_training_gym.frameworks.slime.substep_timing import (
    RoleTimingRecorder,
    StepTimingCollector,
)


def test_aggregate_timing_intervals_tracks_active_time() -> None:
    intervals = [
        TimingInterval(TimingPhase.CUSTOM_REWARD, 10.0, 1.0, 4.0),
        TimingInterval(TimingPhase.CUSTOM_REWARD, 11.0, 2.0, 5.0),
        TimingInterval(TimingPhase.CUSTOM_REWARD, 15.0, 8.0, 9.0),
    ]

    timing = aggregate_timing_intervals(intervals)[0]

    assert timing.duration_s == 5.0
    assert timing.count == 3
    assert [
        (interval.started_at_unix_s, interval.duration_s)
        for interval in timing.intervals
    ] == [(10.0, 4.0), (15.0, 1.0)]


def test_role_timing_includes_framework_activity_presentation() -> None:
    recorder = RoleTimingRecorder(TimingRole.ACTOR)
    with recorder.phase(TimingPhase.TRAIN_MODEL):
        pass
    with recorder.phase(TimingPhase.FORWARD_BACKWARD):
        pass

    phases = {phase.phase: phase for phase in recorder.result().phases}

    assert phases[TimingPhase.TRAIN_MODEL].timeline_group is TimelineGroup.TRAINING
    assert phases[TimingPhase.TRAIN_MODEL].activity_kind is TimingActivityKind.ACTIVITY
    assert phases[TimingPhase.TRAIN_MODEL].display_name == "Train model"
    assert phases[TimingPhase.FORWARD_BACKWARD].parent_phase is TimingPhase.TRAIN_MODEL


def test_collector_rejects_a_late_execution_after_retry_starts() -> None:
    collector = StepTimingCollector()
    first_sequence, _ = collector.begin_role_timing(4, TimingRole.ACTOR.value)
    first = RoleTiming(
        role=TimingRole.ACTOR,
        status=TimingCaptureStatus.UNAVAILABLE,
        execution_sequence=first_sequence,
    )
    assert collector.record_role_timing(4, first_sequence, first.model_dump_json())
    retry_sequence, _ = collector.begin_role_timing(4, TimingRole.ACTOR.value)
    retried = RoleTiming(
        role=TimingRole.ACTOR,
        status=TimingCaptureStatus.NOT_RUN,
        execution_sequence=retry_sequence,
    )
    assert collector.record_role_timing(4, retry_sequence, retried.model_dump_json())
    assert not collector.record_role_timing(4, first_sequence, first.model_dump_json())
    assert (
        collector.read_step_timings(4)[TimingRole.ACTOR.value]
        == retried.model_dump_json()
    )

    collector.close_step(4)

    assert not collector.record_role_timing(
        4, retry_sequence, retried.model_dump_json()
    )
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
