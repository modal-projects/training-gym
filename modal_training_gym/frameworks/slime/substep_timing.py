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
    TimingCaptureStatus,
    TimingInterval,
    TimingPhase,
    TimingRole,
    aggregate_timing_intervals,
)


_TIMING_ENABLED_ENV = "TRAINING_GYM_SUBSTEP_TIMING"
_TRAINING_ATTEMPT_ENV = "TRAINING_GYM_TRAINING_ATTEMPT"
_CURRENT_RECORDER: contextvars.ContextVar[RoleTimingRecorder | None] = (
    contextvars.ContextVar("training_gym_role_timing", default=None)
)


def _ray():
    return importlib.import_module("ray")


@dataclass
class RoleTimingRecorder:
    role: TimingRole
    intervals: list[TimingInterval] = field(default_factory=list)
    full_step: AbstractContextManager[None] | None = None
    primary_phase: AbstractContextManager[None] | None = None

    @contextmanager
    def phase(self, phase: TimingPhase):
        started_at_unix_s = time.time()
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
        phases = aggregate_timing_intervals(self.intervals)
        return RoleTiming(
            role=self.role,
            status=(
                TimingCaptureStatus.CAPTURED
                if phases
                else TimingCaptureStatus.UNAVAILABLE
            ),
            phases=phases,
        )


class StepTimingCollector:
    def __init__(self) -> None:
        self._role_timings: dict[tuple[int, str], str] = {}
        self._closed_rollouts: set[int] = set()

    def record_role_timing(self, rollout_id: int, timing_json: str) -> bool:
        timing = RoleTiming.model_validate_json(timing_json)
        key = (rollout_id, timing.role.value)
        if rollout_id in self._closed_rollouts:
            return False
        existing = self._role_timings.get(key)
        if existing == timing_json:
            return True
        self._role_timings[key] = timing_json
        return True

    def read_step_timings(self, rollout_id: int) -> dict[str, str]:
        return {
            role: timing
            for (stored_rollout_id, role), timing in self._role_timings.items()
            if stored_rollout_id == rollout_id
        }

    def close_step(self, rollout_id: int) -> None:
        self._closed_rollouts.add(rollout_id)
        self._role_timings = {
            key: timing
            for key, timing in self._role_timings.items()
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


def start_role_timing(
    args: object,
    rollout_id: int,
    role: object,
) -> tuple[RoleTimingRecorder, contextvars.Token[RoleTimingRecorder | None]] | None:
    if not _enabled(args):
        return None
    try:
        recorder = RoleTimingRecorder(_role(role))
        token = _CURRENT_RECORDER.set(recorder)
        recorder.full_step = recorder.phase(TimingPhase.FULL_STEP)
        recorder.full_step.__enter__()
        if recorder.role in {TimingRole.ACTOR, TimingRole.CRITIC}:
            recorder.primary_phase = recorder.phase(TimingPhase.TRAIN_MODEL)
            recorder.primary_phase.__enter__()
        return recorder, token
    except Exception:
        return None


def finish_role_timing(
    args: object,
    rollout_id: int,
    state: tuple[RoleTimingRecorder, contextvars.Token[RoleTimingRecorder | None]]
    | None,
) -> None:
    if state is None:
        return
    recorder, token = state
    if recorder.primary_phase is not None:
        recorder.primary_phase.__exit__(None, None, None)
    if recorder.full_step is not None:
        recorder.full_step.__exit__(None, None, None)
    _CURRENT_RECORDER.reset(token)
    collector = getattr(args, "training_gym_timing_collector", None)
    if collector is None:
        return
    try:
        ray = _ray()
        record_role_timing = getattr(collector, "record_role_timing")
        ray.get(
            record_role_timing.remote(
                rollout_id,
                recorder.result().model_dump_json(),
            )
        )
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
    collector: object | None = None
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
            self.recorder = RoleTimingRecorder(TimingRole.DRIVER)
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
            ray = _ray()
            read_step_timings = getattr(self.collector, "read_step_timings")
            recorded = ray.get(read_step_timings.remote(rollout_id))
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
            elif role.value in recorded:
                roles[role] = RoleTiming.model_validate_json(recorded[role.value])
            else:
                roles[role] = RoleTiming(
                    role=role,
                    status=TimingCaptureStatus.UNAVAILABLE,
                )
        timing = SubstepTiming(
            training_run_id=os.environ.get("TRAINING_GYM_TRAINING_RUN_ID", ""),
            training_attempt=int(os.environ[_TRAINING_ATTEMPT_ENV]),
            first_rollout_id=self.first_rollout_id,
            rollout_id_stop_exclusive=self.rollout_id_stop_exclusive,
            rollout_id=rollout_id,
            started_at_unix_s=min(
                phase.started_at_unix_s
                for role_timing in roles.values()
                for phase in role_timing.phases
            ),
            duration_s=next(
                phase.duration_s
                for phase in roles[TimingRole.DRIVER].phases
                if phase.phase is TimingPhase.FULL_STEP
            ),
            roles=tuple(roles.values()),
        )
        if _post_timing(timing):
            try:
                ray = _ray()
                close_step = getattr(self.collector, "close_step")
                ray.get(close_step.remote(rollout_id))
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
        collector_type = ray.remote(num_cpus=0, max_restarts=1)(StepTimingCollector)
        session.collector = collector_type.remote()
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
