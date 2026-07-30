from __future__ import annotations

import contextvars
import importlib
import json
import os
import time
from collections.abc import Awaitable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from modal_training_gym.common.substep_timing import (
    SUBSTEP_TIMING_PROTOCOL,
    RoleTiming,
    SubstepTiming,
    TimingActivityKind,
    TimingCaptureStatus,
    TimingCollectorClient,
    TimingInterval,
    TimingLease,
    TimingPhase,
    TimingRole,
    TimelineGroup,
    aggregate_timing_intervals,
)


_TIMING_ENABLED_ENV = "TRAINING_GYM_SUBSTEP_TIMING"
_TRAINING_ATTEMPT_ENV = "TRAINING_GYM_TRAINING_ATTEMPT"
_CURRENT_RECORDER: contextvars.ContextVar[RoleTimingRecorder | None] = (
    contextvars.ContextVar("training_gym_role_timing", default=None)
)

_PHASE_PRESENTATION = {
    TimingPhase.FULL_STEP: (
        TimelineGroup.SYSTEM,
        TimingActivityKind.CONTAINER,
        "Full step",
        None,
    ),
    TimingPhase.WAIT_FOR_ROLLOUT: (
        TimelineGroup.SYSTEM,
        TimingActivityKind.WAIT,
        "Wait for rollout",
        None,
    ),
    TimingPhase.OFFLOAD_ROLLOUT: (
        TimelineGroup.SYSTEM,
        TimingActivityKind.ACTIVITY,
        "Offload rollout",
        None,
    ),
    TimingPhase.TRAIN_MODELS: (
        TimelineGroup.SYSTEM,
        TimingActivityKind.CONTAINER,
        "Train models",
        None,
    ),
    TimingPhase.CHECKPOINT_SAVE: (
        TimelineGroup.SYSTEM,
        TimingActivityKind.ACTIVITY,
        "Checkpoint save",
        None,
    ),
    TimingPhase.TRAINING_CLEANUP: (
        TimelineGroup.SYSTEM,
        TimingActivityKind.ACTIVITY,
        "Training cleanup",
        None,
    ),
    TimingPhase.WAIT_FOR_NEXT_ROLLOUT: (
        TimelineGroup.SYSTEM,
        TimingActivityKind.WAIT,
        "Wait for next rollout",
        None,
    ),
    TimingPhase.WEIGHT_SYNC: (
        TimelineGroup.SYSTEM,
        TimingActivityKind.ACTIVITY,
        "Weight sync",
        None,
    ),
    TimingPhase.EVALUATE_ROLLOUTS_BEFORE: (
        TimelineGroup.ROLLOUT,
        TimingActivityKind.ACTIVITY,
        "Eval (before)",
        None,
    ),
    TimingPhase.EVALUATE_ROLLOUTS_AFTER: (
        TimelineGroup.ROLLOUT,
        TimingActivityKind.ACTIVITY,
        "Eval (after)",
        None,
    ),
    TimingPhase.GENERATE_ROLLOUTS: (
        TimelineGroup.ROLLOUT,
        TimingActivityKind.ACTIVITY,
        "Generate rollouts",
        None,
    ),
    TimingPhase.CUSTOM_REWARD: (
        TimelineGroup.ROLLOUT,
        TimingActivityKind.ACTIVITY,
        "Custom reward",
        TimingPhase.GENERATE_ROLLOUTS,
    ),
    TimingPhase.CUSTOM_REWARD_POST_PROCESS: (
        TimelineGroup.ROLLOUT,
        TimingActivityKind.ACTIVITY,
        "Reward post-process",
        TimingPhase.GENERATE_ROLLOUTS,
    ),
    TimingPhase.TRAIN_MODEL: (
        TimelineGroup.TRAINING,
        TimingActivityKind.ACTIVITY,
        "Train model",
        None,
    ),
    TimingPhase.FORWARD_BACKWARD: (
        TimelineGroup.TRAINING,
        TimingActivityKind.ACTIVITY,
        "Forward/backward",
        TimingPhase.TRAIN_MODEL,
    ),
    TimingPhase.OPTIMIZER_STEP: (
        TimelineGroup.TRAINING,
        TimingActivityKind.ACTIVITY,
        "Optimizer step",
        TimingPhase.TRAIN_MODEL,
    ),
}


def _ray():
    return importlib.import_module("ray")


@dataclass
class RoleTimingRecorder:
    role: TimingRole
    execution_sequence: int | None = None
    clock_offset_s: float = 0.0
    clock_uncertainty_s: float | None = None
    intervals: list[TimingInterval] = field(default_factory=list)
    full_step: AbstractContextManager[None] | None = None
    primary_phase: AbstractContextManager[None] | None = None

    @contextmanager
    def phase(self, phase: TimingPhase):
        started_at_unix_s = time.time() + self.clock_offset_s
        started_at_monotonic_s = time.monotonic()
        try:
            yield
        finally:
            self.intervals.append(
                TimingInterval(
                    phase=phase,
                    started_at_unix_s=started_at_unix_s,
                    started_at_monotonic_s=started_at_monotonic_s,
                    ended_at_monotonic_s=time.monotonic(),
                )
            )

    def result(self) -> RoleTiming:
        phases = []
        for phase in aggregate_timing_intervals(self.intervals):
            group, kind, display_name, parent_phase = _PHASE_PRESENTATION[phase.phase]
            phases.append(
                phase.model_copy(
                    update={
                        "timeline_group": group,
                        "activity_kind": kind,
                        "display_name": display_name,
                        "parent_phase": parent_phase,
                    }
                )
            )
        return RoleTiming(
            role=self.role,
            status=(
                TimingCaptureStatus.CAPTURED
                if phases
                else TimingCaptureStatus.UNAVAILABLE
            ),
            phases=tuple(phases),
            execution_sequence=self.execution_sequence,
            clock_uncertainty_s=self.clock_uncertainty_s,
        )


class StepTimingCollector:
    def __init__(self) -> None:
        self._execution_sequences: dict[tuple[int, str], int] = {}
        self._role_timings: dict[tuple[int, str], tuple[int, str]] = {}
        self._closed_rollouts: set[int] = set()

    def begin_role_timing(self, rollout_id: int, role: str) -> tuple[int, float]:
        if rollout_id in self._closed_rollouts:
            raise RuntimeError("rollout timing is already closed")
        key = (rollout_id, TimingRole(role).value)
        sequence = self._execution_sequences.get(key, 0) + 1
        self._execution_sequences[key] = sequence
        return sequence, time.time()

    def synchronize_clock(self) -> float:
        return time.time()

    def record_role_timing(
        self,
        rollout_id: int,
        execution_sequence: int,
        timing_json: str,
    ) -> bool:
        timing = RoleTiming.model_validate_json(timing_json)
        key = (rollout_id, timing.role.value)
        if rollout_id in self._closed_rollouts:
            return False
        if (
            timing.execution_sequence != execution_sequence
            or execution_sequence != self._execution_sequences.get(key)
        ):
            return False
        existing = self._role_timings.get(key)
        if existing == (execution_sequence, timing_json):
            return True
        self._role_timings[key] = (execution_sequence, timing_json)
        return True

    def read_step_timings(self, rollout_id: int) -> dict[str, str]:
        return {
            role: timing_json
            for (stored_rollout_id, role), (
                _,
                timing_json,
            ) in self._role_timings.items()
            if stored_rollout_id == rollout_id
        }

    def close_step(self, rollout_id: int) -> None:
        self._closed_rollouts.add(rollout_id)
        self._role_timings = {
            key: timing
            for key, timing in self._role_timings.items()
            if key[0] != rollout_id
        }
        self._execution_sequences = {
            key: sequence
            for key, sequence in self._execution_sequences.items()
            if key[0] != rollout_id
        }


def _enabled(args: object | None = None) -> bool:
    if os.environ.get(_TIMING_ENABLED_ENV) != "1":
        return False
    return (
        args is None or getattr(args, "training_gym_timing_collector", None) is not None
    )


def _role(value: object) -> TimingRole:
    normalized = getattr(value, "value", value)
    return TimingRole(str(normalized).lower())


def _clock_calibration(
    local_time_s: float,
    started_at_monotonic_s: float,
    collector_time_s: float,
) -> tuple[float, float]:
    round_trip_s = time.monotonic() - started_at_monotonic_s
    return collector_time_s - (local_time_s + round_trip_s / 2), round_trip_s / 2


@dataclass(frozen=True)
class RayTimingCollectorClient:
    actor: object

    def begin_role(
        self,
        rollout_id: int,
        role: TimingRole,
    ) -> TimingLease:
        ray = _ray()
        local_time_s = time.time()
        started_at_monotonic_s = time.monotonic()
        execution_sequence, collector_time_s = ray.get(
            getattr(self.actor, "begin_role_timing").remote(
                rollout_id,
                role.value,
            )
        )
        clock_offset_s, clock_uncertainty_s = _clock_calibration(
            local_time_s,
            started_at_monotonic_s,
            collector_time_s,
        )
        return TimingLease(
            execution_sequence=execution_sequence,
            clock_offset_s=clock_offset_s,
            clock_uncertainty_s=clock_uncertainty_s,
        )

    def record_role(
        self,
        rollout_id: int,
        lease: TimingLease,
        timing: RoleTiming,
    ) -> bool:
        return bool(
            _ray().get(
                getattr(self.actor, "record_role_timing").remote(
                    rollout_id,
                    lease.execution_sequence,
                    timing.model_dump_json(),
                )
            )
        )

    def read_step(
        self,
        rollout_id: int,
    ) -> dict[TimingRole, RoleTiming]:
        recorded = _ray().get(
            getattr(self.actor, "read_step_timings").remote(rollout_id)
        )
        return {
            TimingRole(role): RoleTiming.model_validate_json(timing_json)
            for role, timing_json in recorded.items()
        }

    def close_step(self, rollout_id: int) -> None:
        _ray().get(getattr(self.actor, "close_step").remote(rollout_id))

    def synchronize_clock(self) -> tuple[float, float]:
        ray = _ray()
        synchronize_clock = getattr(self.actor, "synchronize_clock")
        calibrations = []
        for _ in range(5):
            local_time_s = time.time()
            started_at_monotonic_s = time.monotonic()
            collector_time_s = ray.get(synchronize_clock.remote())
            calibrations.append(
                _clock_calibration(
                    local_time_s,
                    started_at_monotonic_s,
                    collector_time_s,
                )
            )
        return min(calibrations, key=lambda calibration: calibration[1])


def start_role_timing(
    args: object,
    rollout_id: int,
    role: object,
) -> (
    tuple[
        RoleTimingRecorder,
        contextvars.Token[RoleTimingRecorder | None],
        TimingLease,
    ]
    | None
):
    if not _enabled(args):
        return None
    try:
        timing_role = _role(role)
        collector: TimingCollectorClient = getattr(
            args,
            "training_gym_timing_collector",
        )
        lease = collector.begin_role(rollout_id, timing_role)
        recorder = RoleTimingRecorder(
            timing_role,
            execution_sequence=lease.execution_sequence,
            clock_offset_s=lease.clock_offset_s,
            clock_uncertainty_s=lease.clock_uncertainty_s,
        )
        token = _CURRENT_RECORDER.set(recorder)
        recorder.full_step = recorder.phase(TimingPhase.FULL_STEP)
        recorder.full_step.__enter__()
        if recorder.role in {TimingRole.ACTOR, TimingRole.CRITIC}:
            recorder.primary_phase = recorder.phase(TimingPhase.TRAIN_MODEL)
            recorder.primary_phase.__enter__()
        return recorder, token, lease
    except Exception:
        return None


def finish_role_timing(
    args: object,
    rollout_id: int,
    state: tuple[
        RoleTimingRecorder,
        contextvars.Token[RoleTimingRecorder | None],
        TimingLease,
    ]
    | None,
) -> None:
    if state is None:
        return
    recorder, token, lease = state
    if recorder.primary_phase is not None:
        recorder.primary_phase.__exit__(None, None, None)
    if recorder.full_step is not None:
        recorder.full_step.__exit__(None, None, None)
    _CURRENT_RECORDER.reset(token)
    collector = getattr(args, "training_gym_timing_collector", None)
    if collector is None or recorder.execution_sequence is None:
        return
    try:
        collector.record_role(rollout_id, lease, recorder.result())
    except Exception:
        return


def begin_phase(phase: str):
    recorder = _CURRENT_RECORDER.get()
    if recorder is None:
        return None
    timing = recorder.phase(TimingPhase(phase))
    timing.__enter__()
    return timing


def finish_phase(timing: AbstractContextManager[None] | None) -> None:
    if timing is not None:
        timing.__exit__(None, None, None)


@dataclass
class DriverTimingSession:
    args: object
    collector: TimingCollectorClient | None = None
    clock_offset_s: float = 0.0
    clock_uncertainty_s: float | None = None
    first_rollout_id: int = 0
    rollout_id_stop_exclusive: int = 0
    recorder: RoleTimingRecorder | None = None
    recorder_token: contextvars.Token[RoleTimingRecorder | None] | None = None
    full_step: AbstractContextManager[None] | None = None
    active_phase: AbstractContextManager[None] | None = None

    def configure(self, first_rollout_id: int, rollout_id_stop_exclusive: int) -> None:
        self.first_rollout_id = first_rollout_id
        self.rollout_id_stop_exclusive = rollout_id_stop_exclusive
        setattr(self.args, "first_rollout_id", first_rollout_id)
        setattr(
            self.args,
            "rollout_id_stop_exclusive",
            rollout_id_stop_exclusive,
        )
        setattr(self.args, "training_gym_timing_boundary_ready", True)

    def configure_rollout_manager(self, rollout_manager: object) -> None:
        if self.collector is None:
            return
        try:
            ray = _ray()
            configure_timing = getattr(rollout_manager, "configure_training_gym_timing")
            ray.get(
                configure_timing.remote(
                    self.first_rollout_id,
                    self.rollout_id_stop_exclusive,
                )
            )
        except Exception:
            self.collector = None

    def start_step(self) -> None:
        if self.collector is None:
            return
        try:
            self.recorder = RoleTimingRecorder(
                TimingRole.DRIVER,
                clock_offset_s=self.clock_offset_s,
                clock_uncertainty_s=self.clock_uncertainty_s,
            )
            self.recorder_token = _CURRENT_RECORDER.set(self.recorder)
            self.full_step = self.recorder.phase(TimingPhase.FULL_STEP)
            self.full_step.__enter__()
        except Exception:
            self.recorder = None

    def transition(self, phase: str) -> None:
        if self.active_phase is not None:
            finish_phase(self.active_phase)
        try:
            self.active_phase = begin_phase(phase)
        except ValueError:
            self.active_phase = None

    def finish_active_phase(self) -> None:
        finish_phase(self.active_phase)
        self.active_phase = None

    def publish_step(
        self,
        rollout_id: int,
        actor_ran: bool,
        critic_ran: bool,
        rollout_ran: bool = True,
    ) -> None:
        try:
            self._publish_step(
                rollout_id,
                actor_ran,
                critic_ran,
                rollout_ran,
            )
        except Exception:
            self.recorder = None

    def _publish_step(
        self,
        rollout_id: int,
        actor_ran: bool,
        critic_ran: bool,
        rollout_ran: bool,
    ) -> None:
        if self.collector is None or self.recorder is None:
            return
        self.finish_active_phase()
        if self.full_step is not None:
            self.full_step.__exit__(None, None, None)
        if self.recorder_token is not None:
            _CURRENT_RECORDER.reset(self.recorder_token)
        roles = {TimingRole.DRIVER: self.recorder.result()}
        try:
            recorded = self.collector.read_step(rollout_id)
        except Exception:
            recorded = {}
        for role in (TimingRole.ROLLOUT, TimingRole.ACTOR, TimingRole.CRITIC):
            ran = {
                TimingRole.ROLLOUT: rollout_ran,
                TimingRole.ACTOR: actor_ran,
                TimingRole.CRITIC: critic_ran,
            }[role]
            if not ran:
                roles[role] = RoleTiming(
                    role=role,
                    status=TimingCaptureStatus.NOT_RUN,
                )
            elif role in recorded:
                roles[role] = recorded[role]
            else:
                roles[role] = RoleTiming(
                    role=role,
                    status=TimingCaptureStatus.UNAVAILABLE,
                )
        driver_step = next(
            phase
            for phase in roles[TimingRole.DRIVER].phases
            if phase.phase is TimingPhase.FULL_STEP
        )
        timing = SubstepTiming(
            training_run_id=os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", ""),
            training_attempt=int(os.environ[_TRAINING_ATTEMPT_ENV]),
            first_rollout_id=self.first_rollout_id,
            rollout_id_stop_exclusive=self.rollout_id_stop_exclusive,
            rollout_id=rollout_id,
            source_rollout_id=rollout_id,
            training_rollout_id=rollout_id,
            started_at_unix_s=driver_step.started_at_unix_s,
            duration_s=driver_step.duration_s,
            roles=tuple(roles.values()),
        )
        if _post_timing(timing):
            try:
                self.collector.close_step(rollout_id)
            except Exception:
                pass
        self.recorder = None


def create_driver_timing(args: object) -> DriverTimingSession:
    global _DRIVER_SESSION
    session = DriverTimingSession(args)
    _DRIVER_SESSION = session
    if not _enabled():
        return session
    try:
        ray = _ray()
        collector_type = ray.remote(num_cpus=0, max_restarts=0)(StepTimingCollector)
        session.collector = RayTimingCollectorClient(collector_type.remote())
        (
            session.clock_offset_s,
            session.clock_uncertainty_s,
        ) = session.collector.synchronize_clock()
        setattr(args, "training_gym_timing_collector", session.collector)
    except Exception:
        session.collector = None
    return session


_DRIVER_SESSION: DriverTimingSession | None = None


def transition_driver_phase(phase: str, step_event: str = "") -> None:
    if _DRIVER_SESSION is None:
        return
    mapped = {
        "generate_rollouts": TimingPhase.WAIT_FOR_ROLLOUT,
        "compute_log_probs": TimingPhase.TRAIN_MODELS,
        "optimizer_step": TimingPhase.TRAIN_MODELS,
        "checkpoint_save": TimingPhase.CHECKPOINT_SAVE,
        "offload_rollout": TimingPhase.OFFLOAD_ROLLOUT,
        "offload_train": TimingPhase.TRAINING_CLEANUP,
        "weight_sync": TimingPhase.WEIGHT_SYNC,
        "evaluate_rollouts": (
            TimingPhase.EVALUATE_ROLLOUTS_BEFORE
            if step_event == "eval_begin"
            else TimingPhase.EVALUATE_ROLLOUTS_AFTER
        ),
    }.get(phase)
    if mapped is not None:
        try:
            _DRIVER_SESSION.transition(mapped.value)
        except Exception:
            return


async def timed_await(phase: str, awaitable: Awaitable[object]) -> object:
    timing = begin_phase(phase)
    try:
        return await awaitable
    finally:
        finish_phase(timing)


def _timing_url() -> str:
    status_url = os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_URL", "").strip()
    if status_url.endswith("/api/framework-status"):
        return status_url.removesuffix("/api/framework-status") + "/api/timing-events"
    return ""


def _post_timing(timing: SubstepTiming) -> bool:
    url = _timing_url()
    if not url:
        return False
    request = Request(
        url,
        data=timing.model_dump_json().encode(),
        headers={
            "Authorization": (
                "Bearer " + os.environ.get("TRAINING_GYM_FRAMEWORK_STATUS_TOKEN", "")
            ),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for timeout_s in (2.0, 5.0, 10.0):
        try:
            with urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read())
            return payload.get("status") in {"stored", "duplicate", "stale"}
        except (HTTPError, OSError, URLError, ValueError):
            continue
    return False


def supports_substep_timing(status_url: str) -> bool:
    if not status_url.endswith("/api/framework-status"):
        return False
    url = status_url.removesuffix("/api/framework-status") + "/api/timing-events"
    try:
        with urlopen(url, timeout=2.0) as response:
            payload = json.loads(response.read())
    except (HTTPError, OSError, URLError, ValueError):
        return False
    return payload.get("protocol") == SUBSTEP_TIMING_PROTOCOL
