from __future__ import annotations

import io
import json
from enum import Enum
from typing import TypeVar

from pydantic import BaseModel

from modal_training_gym.common.substep_timing import (
    SubstepTiming,
    SubstepTimingKey,
    TrainingAttemptBoundary,
    TrainingAttemptStarted,
)
from modal_training_gym.utils.metadata import (
    MetadataStore,
    _metadata_volume,
    _safe_reload,
    vol_get,
)


StoredModel = TypeVar("StoredModel", bound=BaseModel)


class StoreResult(str, Enum):
    STORED = "stored"
    DUPLICATE = "duplicate"


class StoredValueConflictError(ValueError):
    pass


def _serialized(model: BaseModel) -> bytes:
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _key_path(store: MetadataStore, key: str) -> str:
    return f"{store.value}/{key}.json"


def _read_model(
    store: MetadataStore,
    key: str,
    model_type: type[StoredModel],
) -> StoredModel | None:
    try:
        value = vol_get(store, key)
    except KeyError:
        return None
    return model_type.model_validate(value)


def _store_once(
    store: MetadataStore,
    key: str,
    model: BaseModel,
) -> StoreResult:
    data = _serialized(model)
    volume = _metadata_volume()
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(io.BytesIO(data), _key_path(store, key))
        return StoreResult.STORED
    except FileExistsError:
        _safe_reload(volume)
        existing = _read_model(store, key, type(model))
        if existing is not None and _serialized(existing) == data:
            return StoreResult.DUPLICATE
        raise StoredValueConflictError(f"Conflicting value for {key}") from None


def _attempt_key(
    training_run_id: str,
    training_attempt: int,
    record_name: str,
) -> str:
    return f"{training_run_id}/{training_attempt:08d}/{record_name}"


def _timing_key(
    training_run_id: str,
    training_attempt: int,
    rollout_id: int,
) -> str:
    return f"{training_run_id}/{training_attempt:08d}/{rollout_id:08d}"


def store_attempt_started(
    attempt: TrainingAttemptStarted,
) -> StoreResult:
    return _store_once(
        MetadataStore.TRAINING_ATTEMPTS,
        _attempt_key(
            attempt.training_run_id,
            attempt.training_attempt,
            "started",
        ),
        attempt,
    )


def store_attempt_boundary(
    boundary: TrainingAttemptBoundary,
) -> StoreResult:
    return _store_once(
        MetadataStore.TRAINING_ATTEMPTS,
        _attempt_key(
            boundary.training_run_id,
            boundary.training_attempt,
            "boundary",
        ),
        boundary,
    )


def list_attempt_boundaries(
    training_run_id: str,
) -> list[TrainingAttemptBoundary]:
    from modal.exception import NotFoundError

    volume = _metadata_volume()
    _safe_reload(volume)
    directory = f"{MetadataStore.TRAINING_ATTEMPTS.value}/{training_run_id}"
    boundaries: list[TrainingAttemptBoundary] = []
    try:
        for entry in volume.iterdir(directory):
            if not entry.path.endswith("/boundary.json"):
                continue
            data = b"".join(volume.read_file(entry.path))
            boundaries.append(TrainingAttemptBoundary.model_validate_json(data))
    except (FileNotFoundError, NotFoundError):
        return []
    return sorted(boundaries, key=lambda boundary: boundary.training_attempt)


def latest_recorded_attempt(training_run_id: str) -> int:
    from modal.exception import NotFoundError

    volume = _metadata_volume()
    _safe_reload(volume)
    directory = f"{MetadataStore.TRAINING_ATTEMPTS.value}/{training_run_id}"
    attempts: list[int] = []
    try:
        for entry in volume.iterdir(directory):
            if not entry.path.endswith("/started.json"):
                continue
            data = b"".join(volume.read_file(entry.path))
            attempts.append(
                TrainingAttemptStarted.model_validate_json(data).training_attempt
            )
    except (FileNotFoundError, NotFoundError):
        return 0
    return max(attempts, default=0)


def store_substep_timing(
    timing: SubstepTiming,
) -> StoreResult:
    boundary = _read_model(
        MetadataStore.TRAINING_ATTEMPTS,
        _attempt_key(
            timing.training_run_id,
            timing.training_attempt,
            "boundary",
        ),
        TrainingAttemptBoundary,
    )
    event_boundary = TrainingAttemptBoundary(
        training_run_id=timing.training_run_id,
        training_attempt=timing.training_attempt,
        first_rollout_id=timing.first_rollout_id,
        rollout_id_stop_exclusive=timing.rollout_id_stop_exclusive,
    )
    if boundary is None:
        store_attempt_boundary(event_boundary)
    elif boundary != event_boundary:
        raise StoredValueConflictError("Timing event does not match attempt boundary")
    return _store_once(
        MetadataStore.SUBSTEP_TIMING,
        _timing_key(
            timing.training_run_id,
            timing.training_attempt,
            timing.rollout_id,
        ),
        timing,
    )


def read_substep_timings(
    training_run_id: str,
    keys: tuple[SubstepTimingKey, ...],
) -> list[SubstepTiming]:
    timings: list[SubstepTiming] = []
    for key in keys:
        timing = _read_model(
            MetadataStore.SUBSTEP_TIMING,
            _timing_key(
                training_run_id,
                key.training_attempt,
                key.rollout_id,
            ),
            SubstepTiming,
        )
        if timing is not None:
            timings.append(timing)
    return timings
