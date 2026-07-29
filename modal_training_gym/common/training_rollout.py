"""Per-rollout training data — prompts, responses, rewards.

Mirrors the EvalResult shape but lives under its own MetadataStore so the
dashboard can list rollouts per training run without scanning the eval store.
Records are written by slime's `log_rollout_data` hook through the async
phase-reporter; reads happen via the dashboard's
``/api/runs/{id}/rollouts`` endpoint.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel, Field, model_validator

from modal_training_gym.common.coerce import safe_int
from modal_training_gym.common.sample import Sample
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items,
    vol_list_tree,
    vol_put_items,
    vol_put_with_summary,
)


# A rollout sample is just a Sample (shared with eval rows). Alias kept for any
# existing imports; new code should use Sample.
TrainingRolloutSample = Sample

# Metadata keys that are internal bookkeeping rather than reward-function
# tags — numeric, but not something a user wants charted as a custom metric.
_NON_TAG_METADATA_KEYS = frozenset(
    {
        "inference",
        "_metadata_type",
        "audio",
        "image",
        "response_length",
        "prompt_length",
        "rollout_id",
        "rollout_idx",
    }
)


def _tag_stats_for_samples(samples: list[Sample]) -> dict[str, dict[str, Any]]:
    """Per-tag numeric stats (count/mean/min/max) across a rollout's samples.

    Scans each sample's free-form ``metadata`` for numeric values under
    custom reward-function tag keys, so the dashboard can chart them over
    rollouts without a fixed schema. Pure function, no IO.
    """
    values_by_tag: dict[str, list[float]] = {}
    for sample in samples:
        for key, value in sample.metadata.items():
            if key in _NON_TAG_METADATA_KEYS:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values_by_tag.setdefault(key, []).append(float(value))

    return {
        tag: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        for tag, values in values_by_tag.items()
    }


class TrainingRolloutResult(BaseModel):
    """All samples from one rollout of one training run."""

    training_run_id: str
    rollout_id: int
    training_attempt: int | None = Field(default=None, gt=0)
    first_rollout_id: int | None = Field(default=None, ge=0)
    rollout_id_stop_exclusive: int | None = Field(default=None, ge=0)
    created_at: int = 0
    samples: list[Sample] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    rollout_time: float | None = None

    @model_validator(mode="after")
    def validate_attempt_boundary(self) -> TrainingRolloutResult:
        values = (
            self.training_attempt,
            self.first_rollout_id,
            self.rollout_id_stop_exclusive,
        )
        if all(value is None for value in values):
            return self
        if any(value is None for value in values):
            raise ValueError("attempt-scoped rollouts require a complete boundary")
        assert self.first_rollout_id is not None
        assert self.rollout_id_stop_exclusive is not None
        if not (
            self.first_rollout_id <= self.rollout_id < self.rollout_id_stop_exclusive
        ):
            raise ValueError("rollout_id is outside the attempt boundary")
        return self

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.score for s in self.samples) / len(self.samples)

    @property
    def storage_key(self) -> str:
        if self.training_attempt is not None:
            return (
                f"{self.training_run_id}/{self.training_attempt:08d}/"
                f"{self.rollout_id:08d}"
            )
        return f"{self.training_run_id}__{self.rollout_id:08d}"

    @property
    def error_summary(self) -> dict[str, Any] | None:
        """Extract error diagnostics from Harbor/agent rollout metrics.

        Returns None when there's nothing notable; otherwise a compact dict
        with the counts a human needs to diagnose an all-zero-reward rollout
        without digging into raw logs.
        """
        m = self.metrics
        if not m:
            return None

        total_samples = (
            safe_int(m.get("agent/valid_sample_count"))
            or safe_int(m.get("agent/raw_zero_reward_sample_count"))
            or self.total
            or 0
        )
        if not total_samples:
            return None

        out: dict[str, Any] = {}

        # Count samples that errored due to infra (sandbox creation, image build, etc.)
        for key, label in (
            ("agent/exit_status/remoteerror_sample_count", "remote_error"),
            ("agent/response_missing_sample_count", "response_missing"),
            ("agent/invalid_infra_sample_count", "infra_invalid"),
            ("agent/limits_exceeded_sample_count", "limits_exceeded"),
        ):
            v = safe_int(m.get(key))
            if v:
                out[label] = v

        if not out:
            return None

        out["total_samples"] = total_samples
        infra_errors = out.get("remote_error", 0) + out.get("infra_invalid", 0)
        if infra_errors >= total_samples:
            out["verdict"] = "all_infra_failure"
        elif infra_errors > 0:
            out["verdict"] = "partial_infra_failure"
        return out

    @property
    def tag_stats(self) -> dict[str, dict[str, Any]]:
        """Per-tag numeric stats across this rollout's samples, or {} if none.

        Populated from custom reward-function tags on ``Sample.metadata``
        (anything numeric that isn't internal bookkeeping) — lets the
        dashboard chart arbitrary reward-shaping signals over rollouts.
        """
        return _tag_stats_for_samples(self.samples)

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "training_run_id": self.training_run_id,
            "rollout_id": self.rollout_id,
            "created_at": self.created_at,
            "total": self.total,
            "mean": self.mean,
        }
        if self.training_attempt is not None:
            summary["training_attempt"] = self.training_attempt
        if self.first_rollout_id is not None:
            summary["first_rollout_id"] = self.first_rollout_id
        if self.rollout_id_stop_exclusive is not None:
            summary["rollout_id_stop_exclusive"] = self.rollout_id_stop_exclusive
        if self.rollout_time is not None:
            summary["rollout_time"] = self.rollout_time
        err = self.error_summary
        if err:
            summary["error_summary"] = err
        tags = self.tag_stats
        if tags:
            summary["tag_stats"] = tags
        return summary

    def _touch_created_at(self) -> None:
        if not self.created_at:
            self.created_at = int(time.time())

    @staticmethod
    def _summary_sort_key(item: dict[str, Any]) -> tuple[str, int]:
        return (
            str(item.get("training_run_id", "")),
            int(item.get("rollout_id", 0) or 0),
        )

    def _summary_item(self) -> dict[str, Any]:
        # summary_key keeps (run_id, rollout_id) uniqueness across runs.
        return {**self.to_summary(), "summary_key": self.storage_key}

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch_created_at()
        if self.training_attempt is not None:
            return vol_put_items(
                [
                    (
                        MetadataStore.TRAINING_ROLLOUTS,
                        self.storage_key,
                        self.model_dump(mode="json"),
                    ),
                    (
                        MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
                        self.storage_key,
                        self._summary_item(),
                    ),
                ],
                is_async=is_async,
            )
        return vol_put_with_summary(
            MetadataStore.TRAINING_ROLLOUTS,
            self.storage_key,
            self.model_dump(mode="json"),
            summary_store=MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
            summary_item=self._summary_item(),
            item_id_key="summary_key",
            sort_key=self._summary_sort_key,
            reverse=False,
            is_async=is_async,
        )

    @classmethod
    def list_summaries_for_run(cls, training_run_id: str) -> list[dict[str, Any]]:
        items = [
            *(vol_get_summary_items(MetadataStore.TRAINING_ROLLOUTS_SUMMARY) or []),
            *vol_list_tree(
                MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
                training_run_id,
            ),
        ]
        return sorted(
            (
                item
                for item in items
                if isinstance(item, dict)
                and item.get("training_run_id") == training_run_id
            ),
            key=lambda item: int(item.get("rollout_id", 0) or 0),
        )


def select_rollout_summaries(
    summaries: list[dict[str, object]],
    boundaries: list[tuple[int, int, int]],
) -> list[dict[str, object]]:
    if not boundaries:
        return [
            summary for summary in summaries if summary.get("training_attempt") is None
        ]

    selected: dict[int, dict[str, object]] = {}
    for summary in summaries:
        rollout_id = _summary_int(summary.get("rollout_id"))
        if rollout_id is None:
            continue
        attempt_value = summary.get("training_attempt")
        training_attempt = _summary_int(attempt_value)
        if attempt_value is not None and training_attempt is None:
            continue
        controlling_attempt = max(
            (
                attempt
                for attempt, first_rollout_id, rollout_id_stop_exclusive in boundaries
                if first_rollout_id <= rollout_id < rollout_id_stop_exclusive
            ),
            default=None,
        )
        if (
            training_attempt == controlling_attempt
            or training_attempt is None
            and controlling_attempt is None
        ):
            selected[rollout_id] = summary
    return [selected[rollout_id] for rollout_id in sorted(selected)]


def _summary_int(value: object) -> int | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (OverflowError, ValueError):
        return None
