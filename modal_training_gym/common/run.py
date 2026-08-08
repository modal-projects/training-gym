"""TrainingRun is a wrapper around a training run.

It is used to track the training run and its results.
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, PrivateAttr, computed_field, field_validator

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.status import FrameworkStatus, resolve_framework_status
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_put_with_summary,
)

if TYPE_CHECKING:
    from modal_training_gym.common.train_result import TrainResult
    from modal_training_gym.common.training_rollout import TrainingRolloutResult

TRAINING_RUNS_STORE_NAME = MetadataStore.TRAINING_RUNS.value


class FrameworkStatusUpdate(BaseModel):
    """Body of ``POST /api/framework-status``.

    Reporters (``common/status_reporter.py``, slime's ``phase_reporting``) post
    more keys than the dashboard tracks (``app_name``, ``metrics``, …); those
    extras are ignored. Progress values come from loosely-typed framework args,
    so anything non-numeric or negative reads as "not provided".
    """

    training_run_id: str
    phase: str
    is_active: bool | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    progress_unit: str | None = None
    rollout_id: int | None = None
    step_id: int | None = None
    # Client-side timestamp of the event (time.time() in the reporting
    # process); step timings use it so queue/network latency doesn't skew them.
    event_ts: float | None = None

    @field_validator(
        "progress_current", "progress_total", "rollout_id", "step_id", mode="before"
    )
    @classmethod
    def _non_negative_int_or_none(cls, value: object) -> int | None:
        if not isinstance(value, (int, float, str)):
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None


class TrainingRunStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingRun(BaseModel):
    """Handle to one launched training run — the record *and* the way to wait on it.

    ``TrainConfig.launch()`` returns a ``TrainingRun`` as soon as training is
    spawned, and persists it to the metadata volume (which is what the
    dashboard reads). Because the Modal app is started detached and the
    ``train`` function-call id is persisted on the record, a run outlives the
    process that launched it and can be picked back up by id from anywhere:

    ```python
    run = TrainConfig(...).launch()
    print(run.training_run_id, run.modal_app_url)

    # ...later, from any other process:
    run = TrainingRun.from_id("<training_run_id>")
    train_result = run.result()   # block for the TrainResult
    run.function_call.cancel(terminate_containers=True)   # or stop it early
    ```

    Never hand-roll ``_build_app()`` + ``app.train.spawn()`` to get this: that
    nested ``app.run()`` is ephemeral, so leaving the block (or Ctrl-C) stops
    the app and kills the run.
    """

    training_run_id: str
    modal_app_id: str = ""
    modal_app_url: str = ""
    framework: Framework
    config: Any
    dataset_id: str = ""
    deployment_id: str = ""
    status: TrainingRunStatus = TrainingRunStatus.RUNNING
    framework_status: FrameworkStatus | None = None
    created_at: int = 0
    started_at: int = 0
    ended_at: int | None = None
    completed_at: int | None = None
    updated_at: int = 0
    duration_seconds: int | None = None
    # Read-only for legacy substep runs
    step_times: dict[str, dict[str, int | None]] | None = None
    # Read-only for legacy substep runs
    substep_times: dict[str, dict[str, dict[str, float | None]]] | None = None
    # Terminal failure message (Ray driver error / exception) for a failed run,
    # so the cause is queryable from the record and shown on the dashboard even
    # after logs roll off. None while running / on success.
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    # Handle to the spawned ``app.train`` Modal FunctionCall so a launched run
    # can be waited on (see ``result()`` / ``__await__``). Empty until the run
    # is actually spawned by ``TrainConfig.launch()``.
    function_call_id: str = ""

    # Runtime-only handles attached by ``TrainConfig.launch()``; never persisted.
    _function_call: Any = PrivateAttr(default=None)
    _status_display: Any = PrivateAttr(default=None)

    @computed_field
    @property
    def group_id(self) -> str | None:
        """Group id, derived from ``metadata`` (its single source of truth).

        Exposed as a top-level attribute/serialized field so the dashboard and
        other callers can read ``run.group_id`` directly, but not stored
        separately — ``TrainConfig`` writes it into ``metadata`` (and
        ``metadata['group_tags']``), and this reads it back so the two can never
        drift out of sync.
        """
        meta = self.metadata or {}
        gid = meta.get("group_id")
        if gid:
            return str(gid)
        tags = meta.get("group_tags")
        if isinstance(tags, dict) and tags.get("group_id"):
            return str(tags["group_id"])
        return None

    # ── Launch-handle behavior (waiting on the spawned run) ──────────────────

    @property
    def function_call(self) -> Any:
        if self._function_call is not None:
            return self._function_call
        import modal

        return modal.FunctionCall.from_id(self.function_call_id)

    def result(
        self,
        *,
        timeout: float | None = None,
        stop_app_on_success: bool = True,
    ) -> TrainResult:
        """Block until the spawned training call finishes and return its TrainResult."""
        from modal_training_gym.common.modal_lifecycle import stop_app
        from modal_training_gym.common.status_reporter import (
            flush as flush_status_reporter,
        )
        from modal_training_gym.common.train_result import TrainResult

        if self._status_display is not None:
            self._status_display.start_polling(self.training_run_id)
        try:
            result_dict = self.function_call.get(timeout=timeout)
        finally:
            if self._status_display is not None:
                self._status_display.stop_polling()
            flush_status_reporter(timeout_seconds=2.0)

        if stop_app_on_success and self.modal_app_id:
            stop_app(self.modal_app_id)
        result = TrainResult(**TrainResult._parse_model_config(result_dict))
        print(f"Training complete: {result.training_run_id}")
        return result

    def __await__(self):
        import asyncio

        async def _wait() -> TrainResult:
            return await asyncio.to_thread(self.result)

        return _wait().__await__()

    def _summary_sort_key(self, item: dict[str, Any]) -> tuple[int, str]:
        return (
            int(item.get("created_at", 0) or 0),
            str(item.get("training_run_id", "")),
        )

    def apply_framework_status(
        self, update: FrameworkStatusUpdate
    ) -> FrameworkStatus | None:
        """Apply one framework-status report to this run (without saving).

        Sets ``framework_status``, merges the report into the
        ``framework_progress`` metadata blob, and records step start/finish
        times. Returns the resolved status, or ``None`` (run untouched) when
        ``update.phase`` isn't a valid status for this run's framework.
        """
        status = resolve_framework_status(update.phase, str(self.framework.value))
        if status is None:
            return None

        self.framework_status = status
        progress: dict[str, Any] = {
            "phase": status.value,
            "updated_at": int(time.time()),
        }
        # is_active: True = stage is actually running on hardware; False =
        # we've marked the stage but it's queuing for a GPU. Sent by the
        # orchestration code in common/train.py (queue=False) and by the
        # Modal function itself when its body starts executing (active=True).
        if update.is_active is not None:
            progress["is_active"] = update.is_active
        for key, value in (
            ("current", update.progress_current),
            ("total", update.progress_total),
            ("unit", update.progress_unit),
            ("rollout_id", update.rollout_id),
            ("step_id", update.step_id),
        ):
            if value is not None:
                progress[key] = value

        metadata = dict(self.metadata or {})
        existing_progress = metadata.get("framework_progress")
        if isinstance(existing_progress, dict):
            # Drop the existing is_active when we get a fresh transition into
            # a different phase — it shouldn't bleed across stage changes.
            if existing_progress.get("phase") != progress["phase"]:
                existing_progress = {
                    k: v for k, v in existing_progress.items() if k != "is_active"
                }
            progress = {**existing_progress, **progress}
        metadata["framework_progress"] = progress
        self.metadata = metadata
        return status

    def record_latest_rollout(self, rollout: TrainingRolloutResult) -> None:
        """Stamp a just-saved rollout's summary onto this run's metadata."""
        metadata = dict(self.metadata or {})
        metadata["latest_rollout"] = {
            "rollout_id": rollout.rollout_id,
            "mean": rollout.mean,
            "total": rollout.total,
            "created_at": rollout.created_at,
        }
        self.metadata = metadata

    def _touch(self) -> None:
        self.updated_at = int(time.time())

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch()
        return vol_put_with_summary(
            MetadataStore.TRAINING_RUNS,
            self.training_run_id,
            self.model_dump(mode="json"),
            summary_store=MetadataStore.TRAINING_RUNS_SUMMARY,
            item_id_key="training_run_id",
            sort_key=self._summary_sort_key,
            reverse=True,
            is_async=is_async,
        )

    @classmethod
    def from_id(
        cls, run_id: str, *, is_async: bool = False
    ) -> TrainingRun | Awaitable[TrainingRun]:
        data = vol_get(MetadataStore.TRAINING_RUNS, run_id, is_async=is_async)
        if is_async:

            async def _run() -> TrainingRun:
                return cls.model_validate(await data)

            return _run()
        return cls.model_validate(data)


def _resume_checkpoint(path: str, name: str, iteration: int | None) -> dict[str, Any]:
    return {
        "resume_checkpoint_path": path,
        "resume_checkpoint_name": name,
        "resume_from_iteration": iteration,
    }


def has_torch_dist_checkpoint(
    save_path: str,
    *,
    is_complete: Callable[[str], bool] | None = None,
) -> bool:
    return torch_dist_resume_checkpoint(save_path, is_complete=is_complete) is not None


def torch_dist_resume_checkpoint(
    save_path: str,
    *,
    is_complete: Callable[[str], bool] | None = None,
) -> dict[str, Any] | None:
    if not os.path.isdir(save_path):
        return None

    is_complete = is_complete or os.path.isdir
    tracker_path = os.path.join(save_path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tracker_path):
        try:
            with open(tracker_path) as f:
                marker = f.read().strip()
        except OSError:
            marker = ""
        if marker == "release":
            path = os.path.join(save_path, "release")
            return (
                _resume_checkpoint(path, "release", None) if is_complete(path) else None
            )
        if marker.isdigit():
            name = f"iter_{int(marker):07d}"
            path = os.path.join(save_path, name)
            return (
                _resume_checkpoint(path, name, int(marker))
                if is_complete(path)
                else None
            )

    try:
        candidates: list[tuple[int, str, str]] = []
        release_path = ""
        for entry in os.scandir(save_path):
            if not entry.is_dir() or not is_complete(entry.path):
                continue
            if entry.name == "release":
                release_path = entry.path
            elif entry.name.startswith("iter_"):
                try:
                    iteration = int(entry.name.removeprefix("iter_"))
                except ValueError:
                    continue
                candidates.append((iteration, entry.name, entry.path))
    except OSError:
        return None

    if candidates:
        iteration, name, path = max(candidates)
        return _resume_checkpoint(path, name, iteration)
    if release_path:
        return _resume_checkpoint(release_path, "release", None)
    return None


def run_scoped_save_root(save_root: str, training_run_id: str) -> str:
    save_root = str(save_root).rstrip("/") or "/"
    if os.path.basename(save_root) == training_run_id:
        return save_root
    return os.path.join(save_root, training_run_id)


def mark_training_attempt_started(
    run: TrainingRun,
    *,
    started_at: int,
) -> int:
    metadata = dict(run.metadata or {})
    try:
        attempt_count = int(metadata.get("attempt_count") or 0) + 1
    except (TypeError, ValueError):
        attempt_count = 1

    metadata["attempt_count"] = attempt_count
    metadata["last_attempt_started_at"] = started_at
    metadata["last_attempt_status"] = "running"
    metadata.pop("terminal_reason", None)

    run.status = TrainingRunStatus.RUNNING
    run.ended_at = None
    run.completed_at = None
    run.duration_seconds = None
    run.metadata = metadata
    return attempt_count


def mark_training_attempt_finished(
    run: TrainingRun,
    *,
    status: str,
    ended_at: int,
) -> None:
    metadata = dict(run.metadata or {})
    metadata["last_attempt_status"] = status
    metadata["last_attempt_ended_at"] = ended_at
    run.metadata = metadata


def record_resume_checkpoint(
    run: TrainingRun,
    resume_checkpoint: dict[str, Any] | None,
) -> None:
    metadata = dict(run.metadata or {})
    if resume_checkpoint is None:
        metadata["resumed_from_checkpoint"] = False
        for key in (
            "resume_checkpoint_path",
            "resume_checkpoint_name",
            "resume_from_iteration",
        ):
            metadata.pop(key, None)
    else:
        metadata["resumed_from_checkpoint"] = True
        metadata.update(resume_checkpoint)
    run.metadata = metadata


def wandb_run_id_for_attempt(training_run_id: str, attempt_count: int) -> str:
    base_run_id = training_run_id[:8]
    return base_run_id if attempt_count <= 1 else f"{base_run_id}-a{attempt_count}"


def record_wandb_attempt(
    run: TrainingRun,
    *,
    entity: str,
    project: str,
    group: str,
    run_id: str,
    attempt_count: int,
) -> None:
    if not project or not run_id:
        return

    metadata = dict(run.metadata or {})
    raw_attempts = metadata.get("wandb_attempts")
    attempts = raw_attempts if isinstance(raw_attempts, list) else []
    attempt = {
        "attempt": attempt_count,
        "entity": entity,
        "project": project,
        "group": group,
        "run_id": run_id,
    }
    attempts = [
        existing
        for existing in attempts
        if not (
            isinstance(existing, dict)
            and (
                existing.get("attempt") == attempt_count
                or existing.get("run_id") == run_id
            )
        )
    ]
    attempts.append(attempt)
    metadata["wandb_attempts"] = sorted(
        attempts,
        key=lambda item: int(item.get("attempt") or 0) if isinstance(item, dict) else 0,
    )
    metadata["wandb_latest_run_id"] = run_id
    run.metadata = metadata
