"""Scalar metric histories mirrored from the framework's ``wandb.log`` calls.

Stored on the metadata volume as
``metric-series/{run}/chunk-000001-{writer}.json``, each chunk covering
``CHUNK_STEPS`` steps, so a 10k-step run is a handful of files rather than
one per point. Every dashboard container writes its own copy of a chunk and
readers merge all copies, newest ingest time per step winning, so autoscaled
replicas neither clobber nor reorder each other's points.
"""

from __future__ import annotations

import time
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


class RunMetrics:
    """One run's points, ``{step: {key: value}}``, plus when each step was
    last ingested so copies from different containers merge deterministically."""

    def __init__(self) -> None:
        self.table: StepTable = {}
        self.written: dict[int, float] = {}

    def merge_points(self, points: Iterable[MetricPoint]) -> set[str]:
        """Last write wins per ``(step, key)``. Returns the chunk keys touched."""
        now = time.time()
        touched: set[str] = set()
        for point in points:
            if point.metrics:
                self.table.setdefault(point.step, {}).update(point.metrics)
                self.written[point.step] = now
                touched.add(chunk_key(point.step))
        return touched

    def chunk_payload(self, chunk: str) -> dict[str, Any]:
        steps = sorted(s for s in self.table if chunk_key(s) == chunk)
        return {
            "steps": {str(s): self.table[s] for s in steps},
            "written": {str(s): self.written.get(s, 0.0) for s in steps},
        }

    def load_chunk(self, payload: Mapping[str, Any]) -> None:
        """Merge a persisted chunk: for each step the more recently ingested
        side (this table or the file) wins on conflicting keys."""
        steps = payload.get("steps")
        if not isinstance(steps, Mapping):
            return
        written = payload.get("written")
        written = written if isinstance(written, Mapping) else {}
        for raw_step, metrics in steps.items():
            if not isinstance(metrics, Mapping) or not str(raw_step).isdigit():
                continue
            step = int(raw_step)
            at = written.get(raw_step)
            at = float(at) if isinstance(at, (int, float)) else 0.0
            mine = self.table.get(step, {})
            if at > self.written.get(step, -1.0):
                self.table[step] = {**mine, **metrics}
                self.written[step] = at
            else:
                self.table[step] = {**metrics, **mine}


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
