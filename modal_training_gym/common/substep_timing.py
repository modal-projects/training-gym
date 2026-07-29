from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SUBSTEP_TIMING_PROTOCOL = "substep_timing"
SUBSTEP_TIMING_SCHEMA_VERSION = 1


class TimingRole(str, Enum):
    DRIVER = "driver"
    ROLLOUT = "rollout"
    ACTOR = "actor"
    CRITIC = "critic"


class TimingCaptureStatus(str, Enum):
    CAPTURED = "captured"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"


class TimingPhase(str, Enum):
    FULL_STEP = "full_step"
    WAIT_FOR_ROLLOUT = "wait_for_rollout"
    OFFLOAD_ROLLOUT = "offload_rollout"
    TRAIN_MODELS = "train_models"
    CHECKPOINT_SAVE = "checkpoint_save"
    TRAINING_CLEANUP = "training_cleanup"
    WAIT_FOR_NEXT_ROLLOUT = "wait_for_next_rollout"
    WEIGHT_SYNC = "weight_sync"
    EVALUATE_ROLLOUTS_BEFORE = "evaluate_rollouts_before"
    EVALUATE_ROLLOUTS_AFTER = "evaluate_rollouts_after"
    GENERATE_ROLLOUTS = "generate_rollouts"
    CUSTOM_REWARD = "custom_reward"
    CUSTOM_REWARD_POST_PROCESS = "custom_reward_post_process"
    TRAIN_MODEL = "train_model"
    FORWARD_BACKWARD = "forward_backward"
    OPTIMIZER_STEP = "optimizer_step"


class TimingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("training_run_id", check_fields=False)
    @classmethod
    def validate_training_run_id(cls, value: str) -> str:
        if "/" in value or value in {".", ".."}:
            raise ValueError("training_run_id is not path-safe")
        return value


class PhaseTimingInterval(TimingModel):
    started_at_unix_s: float = Field(ge=0)
    duration_s: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> PhaseTimingInterval:
        if not all(
            math.isfinite(value) for value in (self.started_at_unix_s, self.duration_s)
        ):
            raise ValueError("timing interval values must be finite")
        return self


class TrainingAttemptStarted(TimingModel):
    schema_version: Literal[1] = SUBSTEP_TIMING_SCHEMA_VERSION
    training_run_id: str = Field(min_length=1)
    training_attempt: int = Field(gt=0)
    started_at_unix_s: float = Field(ge=0)


class TrainingAttemptBoundary(TimingModel):
    schema_version: Literal[1] = SUBSTEP_TIMING_SCHEMA_VERSION
    training_run_id: str = Field(min_length=1)
    training_attempt: int = Field(gt=0)
    first_rollout_id: int = Field(ge=0)
    rollout_id_stop_exclusive: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_rollout_range(self) -> TrainingAttemptBoundary:
        if self.first_rollout_id > self.rollout_id_stop_exclusive:
            raise ValueError("first_rollout_id exceeds rollout_id_stop_exclusive")
        return self


class PhaseTiming(TimingModel):
    phase: TimingPhase
    started_at_unix_s: float = Field(ge=0)
    duration_s: float = Field(ge=0)
    count: int = Field(gt=0)
    intervals: tuple[PhaseTimingInterval, ...] = ()

    @model_validator(mode="after")
    def validate_durations(self) -> PhaseTiming:
        if not all(
            math.isfinite(value) for value in (self.started_at_unix_s, self.duration_s)
        ):
            raise ValueError("timing values must be finite")
        return self


class RoleTiming(TimingModel):
    role: TimingRole
    status: TimingCaptureStatus
    phases: tuple[PhaseTiming, ...] = ()

    @model_validator(mode="after")
    def validate_capture(self) -> RoleTiming:
        if self.status is TimingCaptureStatus.CAPTURED and not self.phases:
            raise ValueError("captured roles require phase timings")
        if self.status is not TimingCaptureStatus.CAPTURED and self.phases:
            raise ValueError("uncaptured roles cannot contain phase timings")
        phases = [timing.phase for timing in self.phases]
        if len(phases) != len(set(phases)):
            raise ValueError("role phase timings must be unique")
        return self


class SubstepTiming(TimingModel):
    schema_version: Literal[1] = SUBSTEP_TIMING_SCHEMA_VERSION
    event_type: Literal["substep_timing"] = "substep_timing"
    training_run_id: str = Field(min_length=1)
    training_attempt: int = Field(gt=0)
    first_rollout_id: int = Field(ge=0)
    rollout_id_stop_exclusive: int = Field(ge=0)
    rollout_id: int = Field(ge=0)
    started_at_unix_s: float = Field(ge=0)
    duration_s: float = Field(ge=0)
    roles: tuple[RoleTiming, ...]

    @model_validator(mode="after")
    def validate_step(self) -> SubstepTiming:
        if not math.isfinite(self.started_at_unix_s) or not math.isfinite(
            self.duration_s
        ):
            raise ValueError("step timing values must be finite")
        if not (
            self.first_rollout_id <= self.rollout_id < self.rollout_id_stop_exclusive
        ):
            raise ValueError("rollout_id is outside the attempt boundary")
        roles = [capture.role for capture in self.roles]
        if len(roles) != len(set(roles)):
            raise ValueError("step role timings must be unique")
        if TimingRole.DRIVER not in roles or TimingRole.ROLLOUT not in roles:
            raise ValueError("step timing requires driver and rollout roles")
        return self


class SubstepTimingKey(TimingModel):
    training_attempt: int = Field(gt=0)
    rollout_id: int = Field(ge=0)


class SubstepTimingQuery(TimingModel):
    keys: tuple[SubstepTimingKey, ...] = Field(max_length=512)


@dataclass(frozen=True)
class TimingInterval:
    phase: TimingPhase
    started_at_unix_s: float
    started_at_monotonic_s: float
    ended_at_monotonic_s: float


def aggregate_timing_intervals(
    intervals: list[TimingInterval],
) -> tuple[PhaseTiming, ...]:
    intervals_by_phase: dict[TimingPhase, list[TimingInterval]] = {}
    for interval in intervals:
        if interval.ended_at_monotonic_s < interval.started_at_monotonic_s:
            continue
        intervals_by_phase.setdefault(interval.phase, []).append(interval)

    aggregated: list[PhaseTiming] = []
    for phase, phase_intervals in intervals_by_phase.items():
        ordered = sorted(
            phase_intervals,
            key=lambda interval: interval.started_at_monotonic_s,
        )
        range_start = ordered[0].started_at_monotonic_s
        range_end = ordered[0].ended_at_monotonic_s
        range_start_unix_s = ordered[0].started_at_unix_s
        merged_intervals: list[PhaseTimingInterval] = []
        for interval in ordered[1:]:
            if interval.started_at_monotonic_s > range_end:
                merged_intervals.append(
                    PhaseTimingInterval(
                        started_at_unix_s=range_start_unix_s,
                        duration_s=range_end - range_start,
                    )
                )
                range_start = interval.started_at_monotonic_s
                range_end = interval.ended_at_monotonic_s
                range_start_unix_s = interval.started_at_unix_s
            else:
                range_end = max(range_end, interval.ended_at_monotonic_s)
        merged_intervals.append(
            PhaseTimingInterval(
                started_at_unix_s=range_start_unix_s,
                duration_s=range_end - range_start,
            )
        )
        aggregated.append(
            PhaseTiming(
                phase=phase,
                started_at_unix_s=min(
                    interval.started_at_unix_s for interval in merged_intervals
                ),
                duration_s=sum(interval.duration_s for interval in merged_intervals),
                count=len(ordered),
                intervals=tuple(merged_intervals),
            )
        )
    return tuple(sorted(aggregated, key=lambda timing: timing.started_at_unix_s))
