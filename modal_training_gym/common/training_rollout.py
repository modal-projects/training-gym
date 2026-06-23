"""Per-rollout training data — prompts, responses, rewards.

Mirrors the EvalResult shape but lives under its own MetadataStore so the
dashboard can list rollouts per training run without scanning the eval store.
Records are written by slime's `log_rollout_data` hook through the async
phase-reporter; reads happen via the dashboard's
``/api/runs/{id}/rollouts`` endpoint.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from modal_training_gym.common.sample import Sample
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items,
    vol_put,
    vol_put_async,
    vol_upsert_summary_item,
    vol_upsert_summary_item_async,
)


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# A rollout sample is just a Sample (shared with eval rows). Alias kept for any
# existing imports; new code should use Sample.
TrainingRolloutSample = Sample


class TrainingRolloutResult(BaseModel):
    """All samples from one rollout of one training run."""

    training_run_id: str
    rollout_id: int
    created_at: int = 0
    samples: list[Sample] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    rollout_time: float | None = None

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
        # One canonical file per (run, rollout). Zero-padded rollout id so
        # listings return them in step order without needing a sort.
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
            _safe_int(m.get("agent/valid_sample_count"))
            or _safe_int(m.get("agent/raw_zero_reward_sample_count"))
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
            v = _safe_int(m.get(key))
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

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "training_run_id": self.training_run_id,
            "rollout_id": self.rollout_id,
            "created_at": self.created_at,
            "total": self.total,
            "mean": self.mean,
        }
        err = self.error_summary
        if err:
            summary["error_summary"] = err
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

    def save(self) -> None:
        self._touch_created_at()
        payload = self.model_dump(mode="json")
        vol_put(MetadataStore.TRAINING_ROLLOUTS, self.storage_key, payload)
        vol_upsert_summary_item(
            MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
            self._summary_item(),
            item_id_key="summary_key",
            sort_key=self._summary_sort_key,
            reverse=False,
        )

    async def save_async(self) -> None:
        self._touch_created_at()
        payload = self.model_dump(mode="json")
        await vol_put_async(MetadataStore.TRAINING_ROLLOUTS, self.storage_key, payload)
        await vol_upsert_summary_item_async(
            MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
            self._summary_item(),
            item_id_key="summary_key",
            sort_key=self._summary_sort_key,
            reverse=False,
        )

    @classmethod
    def list_summaries_for_run(cls, training_run_id: str) -> list[dict[str, Any]]:
        """Lightweight per-rollout summaries for one run, sorted by rollout_id."""
        items = vol_get_summary_items(MetadataStore.TRAINING_ROLLOUTS_SUMMARY) or []
        return sorted(
            (
                item
                for item in items
                if isinstance(item, dict)
                and item.get("training_run_id") == training_run_id
            ),
            key=lambda item: int(item.get("rollout_id", 0) or 0),
        )
