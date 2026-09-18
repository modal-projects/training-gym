"""Scalar metric histories mirrored from the framework's ``wandb.log`` calls.

Stored on the metadata volume as
``metric-series/{run}/chunk-000001-{writer}.json``, each chunk covering
``CHUNK_STEPS`` steps (one file per dashboard container that ingested it, all
merged on read), so a 10k-step run is a handful of files rather than one per
point. In memory a run is ``{step: {key: value}}``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, Field

from modal_training_gym.utils.metadata import MetadataStore

CHUNK_STEPS = 1000
MAX_POINTS_PER_BATCH = 5000
DEFAULT_MAX_POINTS_PER_KEY = 2000
MAX_POINTS_PER_KEY = 20_000

StepTable = dict[int, dict[str, float]]


class MetricPoint(BaseModel):
    step: int = Field(ge=0, le=100_000_000)
    metrics: dict[str, float] = Field(default_factory=dict)


class MetricPointsBatch(BaseModel):
    training_run_id: str
    points: list[MetricPoint] = Field(
        default_factory=list, max_length=MAX_POINTS_PER_BATCH
    )
    final: bool = False  # process-exit flush: persist right away


def metric_series_store(training_run_id: str) -> str:
    return f"{MetadataStore.METRIC_SERIES.value}/{training_run_id}"


def chunk_key(step: int) -> str:
    return f"chunk-{step // CHUNK_STEPS:06d}"


def merge_points(table: StepTable, points: Iterable[MetricPoint]) -> set[str]:
    """Last write wins per ``(step, key)``. Returns the chunk keys touched."""
    touched: set[str] = set()
    for point in points:
        if point.metrics:
            table.setdefault(point.step, {}).update(point.metrics)
            touched.add(chunk_key(point.step))
    return touched


def chunk_payload(chunk: str, table: StepTable) -> dict[str, Any]:
    return {
        "steps": {
            str(step): metrics
            for step, metrics in sorted(table.items())
            if chunk_key(step) == chunk
        }
    }


def load_chunk(table: StepTable, payload: Mapping[str, Any]) -> None:
    """Merge a persisted chunk; points already in memory are newer and win."""
    steps = payload.get("steps")
    if not isinstance(steps, Mapping):
        return
    for raw_step, metrics in steps.items():
        if isinstance(metrics, Mapping) and str(raw_step).isdigit():
            merged = dict(metrics)
            merged.update(table.get(int(raw_step), {}))
            table[int(raw_step)] = merged


def downsample(rows: list[list[float]], max_points: int) -> list[list[float]]:
    """Keep at most ``max_points`` rows: both endpoints plus each bucket's
    min and max, so loss spikes survive (W&B's history sampler does the same)."""
    if len(rows) <= max_points:
        return rows
    if max_points < 4:
        return [rows[0], rows[-1]][:max_points]
    buckets = (max_points - 2) // 2
    lo_step, span = rows[0][0], (rows[-1][0] - rows[0][0]) or 1

    def bucket_of(i: int) -> int:
        return min(buckets - 1, int((rows[i][0] - lo_step) / span * buckets))

    keep = {0, len(rows) - 1}
    start = 0
    while start < len(rows):
        end = start
        while end < len(rows) and bucket_of(end) == bucket_of(start):
            end += 1
        keep.add(min(range(start, end), key=lambda i: rows[i][1]))
        keep.add(max(range(start, end), key=lambda i: rows[i][1]))
        start = end
    return [rows[i] for i in sorted(keep)]


def series_response(
    training_run_id: str,
    table: StepTable,
    *,
    keys: list[str] | None = None,
    max_points: int = DEFAULT_MAX_POINTS_PER_KEY,
) -> dict[str, Any]:
    all_keys = sorted({key for metrics in table.values() for key in metrics})
    series: dict[str, list[list[float]]] = {}
    latest: dict[str, float] = {}
    for key in keys or all_keys:
        rows = [
            [step, table[step][key]] for step in sorted(table) if key in table[step]
        ]
        if rows:
            series[key] = downsample(rows, max_points)
            latest[key] = rows[-1][1]
    steps = sorted(table)
    return {
        "training_run_id": training_run_id,
        "keys": all_keys,
        "series": series,
        "latest": latest,
        "step_range": [steps[0], steps[-1]] if steps else None,
        "point_count": len(steps),
    }
