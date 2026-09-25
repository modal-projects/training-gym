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

from modal_training_dojo.common.reporting import MAX_METRIC_POINTS_PER_BATCH
from modal_training_dojo.utils.metadata import MetadataStore

CHUNK_STEPS = 1000
MAX_POINTS_PER_BATCH = MAX_METRIC_POINTS_PER_BATCH
MAX_POINTS_PER_KEY = 1000  # per key per response; W&B samples similarly

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
    """One run's points, ``{step: {key: value}}``, plus when each value was
    ingested so copies from different containers merge deterministically."""

    def __init__(self) -> None:
        self.table: StepTable = {}
        self.written: dict[int, dict[str, float]] = {}

    def merge_points(self, points: Iterable[MetricPoint]) -> set[str]:
        """Last write wins per ``(step, key)``. Returns the chunk keys touched."""
        now = time.time()
        touched: set[str] = set()
        for point in points:
            if point.metrics:
                self.table.setdefault(point.step, {}).update(point.metrics)
                self.written.setdefault(point.step, {}).update(
                    dict.fromkeys(point.metrics, now)
                )
                touched.add(chunk_key(point.step))
        return touched

    def chunk_payload(self, chunk: str) -> dict[str, Any]:
        steps = sorted(s for s in self.table if chunk_key(s) == chunk)
        return {
            "steps": {str(s): self.table[s] for s in steps},
            "written": {str(s): self.written.get(s, {}) for s in steps},
        }

    def load_chunk(self, payload: Mapping[str, Any]) -> None:
        """Merge a persisted chunk; per ``(step, key)`` the more recently
        ingested value (in memory or in the file) wins."""
        steps = payload.get("steps")
        if not isinstance(steps, Mapping):
            return
        written = payload.get("written")
        written = written if isinstance(written, Mapping) else {}
        for raw_step, metrics in steps.items():
            if not (
                isinstance(metrics, Mapping) and metrics and str(raw_step).isdigit()
            ):
                continue
            step = int(raw_step)
            stamps = written.get(raw_step)
            stamps = stamps if isinstance(stamps, Mapping) else {}
            mine = self.table.setdefault(step, {})
            mine_at = self.written.setdefault(step, {})
            for key, value in metrics.items():
                at = stamps.get(key)
                at = float(at) if isinstance(at, (int, float)) else 0.0
                if key not in mine or at > mine_at.get(key, -1.0):
                    mine[key] = value
                    mine_at[key] = at


def downsample(rows: list[list[float]], max_points: int) -> list[list[float]]:
    """Keep both endpoints plus each bucket's min and max, so loss spikes
    survive (W&B's history sampler does the same)."""
    if len(rows) <= max_points:
        return rows
    buckets = max(1, (max_points - 2) // 2)
    keep = {0, len(rows) - 1}
    for b in range(buckets):
        span = range(b * len(rows) // buckets, (b + 1) * len(rows) // buckets)
        keep.add(min(span, key=lambda i: rows[i][1]))
        keep.add(max(span, key=lambda i: rows[i][1]))
    return [rows[i] for i in sorted(keep)]


def metric_series(table: StepTable, max_points: int) -> dict[str, list[list[float]]]:
    """``{key: [[step, value], ...]}``, each key downsampled to ``max_points``."""
    by_key: dict[str, list[list[float]]] = {}
    for step in sorted(table):
        for key, value in table[step].items():
            by_key.setdefault(key, []).append([step, value])
    return {key: downsample(rows, max_points) for key, rows in sorted(by_key.items())}
