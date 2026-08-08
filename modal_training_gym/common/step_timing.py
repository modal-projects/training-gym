from __future__ import annotations

import time
from enum import Enum
from typing import Any, Awaitable

from pydantic import BaseModel, Field, field_validator

from modal_training_gym.common.status import SlimeStatus
from modal_training_gym.common.timing_recorder import (
    MAX_PHASE_INVOCATIONS,
    MAX_TIMING_PHASES,
)
from modal_training_gym.utils.metadata import MetadataStore, vol_list, vol_put


def is_safe_run_id(value: str) -> bool:
    return (
        bool(value)
        and value[0] != "."
        and all(
            character
            in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in value
        )
    )


class Role(str, Enum):
    DRIVER = "driver"
    ROLLOUT = "rollout"
    ACTOR = "actor"
    CRITIC = "critic"


class PhaseTiming(BaseModel):
    count: int
    total_duration_s: float
    longest_duration_s: float
    first_start_s: float
    last_end_s: float
    invocations: list[tuple[float, float]] = Field(
        default_factory=list, max_length=MAX_PHASE_INVOCATIONS
    )

    @property
    def average_duration_s(self) -> float:
        return self.total_duration_s / self.count if self.count else 0.0


class RoleTimingRecord(BaseModel):
    training_run_id: str
    rollout_id: int | None = Field(default=None, ge=0)
    role: Role
    created_at: int = 0
    lane_start_unix_s: float | None = None
    phases: dict[str, PhaseTiming] = Field(
        default_factory=dict, max_length=MAX_TIMING_PHASES
    )

    @field_validator("training_run_id")
    @classmethod
    def validate_training_run_id(cls, value: str) -> str:
        if not is_safe_run_id(value):
            raise ValueError("unsafe training run id")
        return value

    @property
    def storage_key(self) -> str:
        rollout = "pre-loop" if self.rollout_id is None else f"{self.rollout_id:08d}"
        return f"{rollout}__{self.role.value}"

    @staticmethod
    def store(training_run_id: str) -> str:
        return f"{MetadataStore.SUBSTEP_TIMING.value}/{training_run_id}"

    def _touch_created_at(self) -> None:
        if not self.created_at:
            self.created_at = int(time.time())

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch_created_at()
        return vol_put(
            self.store(self.training_run_id),
            self.storage_key,
            self.model_dump(mode="json"),
            is_async=is_async,
        )


class Substep(str, Enum):
    EVAL_BEFORE = SlimeStatus.EVAL_ROLLOUT_LOGGING.value
    GENERATE_ROLLOUTS = SlimeStatus.ROLLOUT_LOGGING.value
    OFFLOAD_ROLLOUT = SlimeStatus.OFFLOAD_ROLLOUT.value
    COMPUTE_LOG_PROBS = SlimeStatus.COMPUTE_LOG_PROBS.value
    OPTIMIZER_STEP = SlimeStatus.OPTIMIZER_STEP.value
    CHECKPOINT_SAVE = SlimeStatus.CHECKPOINT_SAVE.value
    OFFLOAD_TRAIN = SlimeStatus.OFFLOAD_TRAIN.value
    WEIGHT_SYNC = SlimeStatus.WEIGHT_SYNC.value
    EVAL_AFTER = f"{SlimeStatus.EVAL_ROLLOUT_LOGGING.value}_end"

    WAIT_FOR_ROLLOUT = "wait_for_rollout"
    WAIT_FOR_NEXT_ROLLOUT = "wait_for_next_rollout"
    TRAIN_MODELS = "train_models"
    GENERATE_SAMPLES = "generate_samples"
    SAMPLE_GENERATION = "sample_generation"
    REWARD = "reward"
    REWARD_BATCH = "reward_batch"
    REWARD_POST_PROCESS = "reward_post_process"
    FORWARD_BACKWARD = "forward_backward"


def rollout_lanes(records: list[dict[str, Any]]) -> dict[str, Any]:
    lanes = {
        record["role"]: {
            "role": record["role"],
            "lane_start_unix_s": record["lane_start_unix_s"],
            "phases": record["phases"],
        }
        for record in records
    }
    return {"roles": lanes}


def measured_run_times(
    training_run_id: str,
) -> tuple[
    dict[str, dict[str, int | None]], dict[str, dict[str, dict[str, float | None]]]
]:
    not_in_step = (
        Substep.CHECKPOINT_SAVE.value,
        Substep.EVAL_BEFORE.value,
        Substep.EVAL_AFTER.value,
    )
    step_times: dict[str, dict[str, int | None]] = {}
    substep_times: dict[str, dict[str, dict[str, float | None]]] = {}
    for rollout_id, records in sorted(
        load_run(training_run_id).items(),
        key=_rollout_sort_key,
    ):
        if rollout_id is None:
            continue
        substeps: dict[str, dict[str, float | None]] = {}
        step_duration = 0.0
        for record in records:
            lane_start = record["lane_start_unix_s"]
            if lane_start is None:
                continue
            role = record["role"]
            for name, phase in record["phases"].items():
                if role == Role.DRIVER.value and name not in not_in_step:
                    step_duration += phase["total_duration_s"]
                if role == Role.DRIVER.value:
                    key = name
                else:
                    key = f"{name} ({role})"
                substeps[key] = {
                    "start": lane_start + phase["first_start_s"],
                    "duration_s": phase["total_duration_s"],
                }
        if not substeps:
            continue
        step = str(rollout_id + 1)
        step_times[step] = {"duration_s": round(step_duration)}
        substep_times[step] = substeps
    return step_times, substep_times


def load_run(training_run_id: str) -> dict[int | None, list[dict[str, Any]]]:
    records = vol_list(RoleTimingRecord.store(training_run_id))
    grouped: dict[int | None, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["rollout_id"], []).append(record)
    return grouped


async def load_run_async(
    training_run_id: str,
) -> tuple[dict[int | None, list[dict[str, Any]]], bool]:
    records, had_failures = await vol_list(
        RoleTimingRecord.store(training_run_id),
        is_async=True,
        return_failures=True,
    )
    grouped: dict[int | None, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["rollout_id"], []).append(record)
    return grouped, had_failures


def _rollout_sort_key(item: tuple[int | None, Any]) -> tuple[bool, int]:
    rollout_id = item[0]
    return rollout_id is None, 0 if rollout_id is None else rollout_id


_LEGACY_RENAMES = {Substep.OPTIMIZER_STEP.value: Substep.TRAIN_MODELS.value}


def legacy_run_to_records(
    substep_times: dict[str, dict[str, dict[str, float | None]]] | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for step_key, subs in (substep_times or {}).items():
        if not step_key.isdigit():
            continue
        lane_start = min(
            (sub["start"] for sub in subs.values() if sub.get("start") is not None),
            default=None,
        )
        if lane_start is None:
            continue
        phases: dict[str, dict[str, float]] = {}
        for name, sub in subs.items():
            start, duration = sub.get("start"), sub.get("duration_s")
            if start is None or duration is None:
                continue
            rel = start - lane_start
            phases[_LEGACY_RENAMES.get(name, name)] = {
                "count": 1,
                "total_duration_s": round(duration, 6),
                "longest_duration_s": round(duration, 6),
                "first_start_s": round(rel, 6),
                "last_end_s": round(rel + duration, 6),
            }
        records.append(
            {
                "rollout_id": int(step_key) - 1,
                "role": Role.DRIVER.value,
                "lane_start_unix_s": lane_start,
                "phases": phases,
            }
        )
    return records
