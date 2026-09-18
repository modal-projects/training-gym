"""Scalar metric histories mirrored from the framework's W&B-style logger.

The training container flattens every ``wandb.log`` payload to
``{"train/loss": 0.42, ...}`` and posts ``(step, ts, metrics)`` points to the
dashboard (see ``metric_mirror.py``, which stays pydantic-free for the
container). The dashboard keeps one directory per run on the metadata volume,
``metric-series/{run}/chunk-000001.json``, holding a fixed range of steps
each, so a 10k-step run is a few dozen files rather than one per point.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from modal_training_gym.common.metric_mirror import (
    MAX_STEP,
    as_finite_float as _as_finite_float,
)
from modal_training_gym.utils.metadata import MetadataStore

CHUNK_STEPS = 1000
MAX_POINTS_PER_BATCH = 5000
DEFAULT_MAX_POINTS_PER_KEY = 2000
MAX_POINTS_PER_KEY = 20_000


class MetricPoint(BaseModel):
    step: int = Field(ge=0, le=MAX_STEP)
    ts: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)


class MetricPointsBatch(BaseModel):
    training_run_id: str
    points: list[MetricPoint] = Field(
        default_factory=list, max_length=MAX_POINTS_PER_BATCH
    )
    # Identifies the logging process (``{hostname}:{pid}``) so a single
    # process's duplicate posts can be told apart from two ranks logging the
    # same step. Informational only.
    source: str = ""
    # Set on the process-exit flush; the dashboard persists immediately
    # rather than waiting for the next flush tick.
    final: bool = False


# In-memory shape: ``{step: {"ts": float, "m": {key: value}}}``.
StepRecord = dict[str, Any]
StepTable = dict[int, StepRecord]


def metric_series_store(training_run_id: str) -> str:
    return f"{MetadataStore.METRIC_SERIES.value}/{training_run_id}"


def chunk_index(step: int) -> int:
    return step // CHUNK_STEPS


def chunk_key(index: int) -> str:
    return f"chunk-{index:06d}"


def merge_points(table: StepTable, points: Iterable[MetricPoint]) -> set[int]:
    """Merge points into ``table`` (last write wins per ``(step, key)``).

    Returns the chunk indices that changed.
    """
    touched: set[int] = set()
    for point in points:
        if not point.metrics:
            continue
        record = table.get(point.step)
        if record is None:
            record = {"ts": point.ts, "m": {}}
            table[point.step] = record
        elif point.ts > float(record.get("ts", 0.0) or 0.0):
            record["ts"] = point.ts
        record["m"].update(point.metrics)
        touched.add(chunk_index(point.step))
    return touched


def chunk_payload(training_run_id: str, index: int, table: StepTable) -> dict[str, Any]:
    lo, hi = index * CHUNK_STEPS, (index + 1) * CHUNK_STEPS
    steps = {
        str(step): record
        for step, record in sorted(table.items())
        if lo <= step < hi and record.get("m")
    }
    return {"training_run_id": training_run_id, "chunk": index, "steps": steps}


def load_chunk(table: StepTable, payload: Mapping[str, Any]) -> None:
    """Merge a persisted chunk into ``table`` without clobbering newer points."""
    steps = payload.get("steps")
    if not isinstance(steps, Mapping):
        return
    for raw_step, raw_record in steps.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError):
            continue
        if not isinstance(raw_record, Mapping):
            continue
        metrics = raw_record.get("m")
        if not isinstance(metrics, Mapping):
            continue
        ts = _as_finite_float(raw_record.get("ts")) or 0.0
        record = table.get(step)
        if record is None:
            table[step] = {"ts": ts, "m": dict(metrics)}
            continue
        # Points already in memory arrived after the file was written.
        for key, value in metrics.items():
            record["m"].setdefault(key, value)
        if ts > float(record.get("ts", 0.0) or 0.0):
            record["ts"] = ts


def metric_keys(table: StepTable) -> list[str]:
    keys: set[str] = set()
    for record in table.values():
        keys.update(record.get("m", {}))
    return sorted(keys)


def series_for_key(
    table: StepTable,
    key: str,
    *,
    min_step: int | None = None,
    max_step: int | None = None,
) -> list[list[float]]:
    rows: list[list[float]] = []
    for step in sorted(table):
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        value = table[step].get("m", {}).get(key)
        if value is None:
            continue
        rows.append([step, value, float(table[step].get("ts", 0.0) or 0.0)])
    return rows


def downsample(rows: list[list[float]], max_points: int) -> list[list[float]]:
    """Reduce to at most ``max_points`` rows, keeping each bucket's min and max.

    Loss spikes and reward jumps survive; W&B's own history downsampler
    keeps extremes for the same reason. Rows are ``[step, value, ts]``.
    """
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    buckets = max(1, (max_points - 2) // 2)
    keep: set[int] = {0, len(rows) - 1}
    lo_step, hi_step = rows[0][0], rows[-1][0]
    span = (hi_step - lo_step) or 1
    bucket_of = lambda row: min(  # noqa: E731
        buckets - 1, int((row[0] - lo_step) / span * buckets)
    )
    start = 0
    while start < len(rows):
        bucket = bucket_of(rows[start])
        end = start
        while end < len(rows) and bucket_of(rows[end]) == bucket:
            end += 1
        lo_i = hi_i = start
        for i in range(start, end):
            if rows[i][1] < rows[lo_i][1]:
                lo_i = i
            if rows[i][1] > rows[hi_i][1]:
                hi_i = i
        keep.update((lo_i, hi_i))
        start = end
    return [rows[i] for i in sorted(keep)]


def series_response(
    training_run_id: str,
    table: StepTable,
    *,
    keys: list[str] | None = None,
    min_step: int | None = None,
    max_step: int | None = None,
    max_points: int = DEFAULT_MAX_POINTS_PER_KEY,
) -> dict[str, Any]:
    all_keys = metric_keys(table)
    wanted = [key for key in (keys or all_keys) if key in set(all_keys)]
    series: dict[str, list[list[float]]] = {}
    latest: dict[str, float] = {}
    for key in wanted:
        rows = series_for_key(table, key, min_step=min_step, max_step=max_step)
        if rows:
            latest[key] = rows[-1][1]
        series[key] = downsample(rows, max_points)
    steps = sorted(table)
    return {
        "training_run_id": training_run_id,
        "keys": all_keys,
        "step_range": [steps[0], steps[-1]] if steps else None,
        "point_count": len(steps),
        "series": series,
        "latest": latest,
    }
