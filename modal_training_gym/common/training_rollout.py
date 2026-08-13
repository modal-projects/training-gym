"""Per-rollout training data — prompts, responses, rewards.

Mirrors the EvalResult shape but lives under its own MetadataStore so the
dashboard can list rollouts per training run without scanning the eval store.
Records are written by slime's `log_rollout_data` hook through the async
phase-reporter; reads happen via the dashboard's
``/api/runs/{id}/rollouts`` endpoint.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import time
from collections.abc import Awaitable
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from modal_training_gym.common.coerce import optional_int, safe_int
from modal_training_gym.common.sample import Sample
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get_summary_items,
    vol_put_with_summary,
)


class TrainingRolloutSample(Sample):
    """``rollout_index`` is slime's per-episode ``Sample.rollout_id``, not the
    step id on ``TrainingRolloutResult``."""

    rollout_index: int | None = None
    sample_index: int | None = None
    group_index: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _promote_sample(cls, value: Any) -> Any:
        if isinstance(value, Sample) and not isinstance(value, cls):
            return value.model_dump()
        return value


# Metadata keys that are internal bookkeeping rather than reward-function
# tags — numeric, but not something a user wants charted as a custom metric.
_NON_TAG_METADATA_KEYS = frozenset(
    {
        "inference",
        "_metadata_type",
        "audio",
        "image",
        "image_ref",
        "response_length",
        "prompt_length",
        "rollout_id",
        "rollout_idx",
    }
)


def _clean_prompt(text: str) -> str:
    """Make a chat-templated prompt readable for display.

    Dataset prompts often arrive as a chat template wrapping a Python repr
    of the messages list (e.g. ``<|im_start|>user\\n[{'content': '...',
    'role': 'user'}]<|im_end|>...``) plus a leaked reference/assistant turn.
    Pull the message content out and drop the template scaffolding; fall
    back to stripping special tokens when there's no messages repr.
    """
    start, end = text.find("[{"), text.rfind("}]")
    if start != -1 and end > start:
        try:
            data = ast.literal_eval(text[start : end + 2])
            if isinstance(data, list):
                parts = [
                    str(m["content"])
                    for m in data
                    if isinstance(m, dict) and m.get("content")
                ]
                if parts:
                    return "\n\n".join(parts).strip()
        except (ValueError, SyntaxError):
            pass
    cleaned = re.sub(r"<\|[^|]*\|>", "", text)
    cleaned = re.sub(r"</?think>", "", cleaned)
    # Drop standalone role-header lines left behind by the template.
    cleaned = re.sub(r"(?m)^(system|user|assistant)\s*$\n?", "", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _apply_parsed(rows: object) -> None:
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed = row.get("parsed_response")
        if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
            raw = row.get("response")
            if isinstance(raw, str):
                row["raw_response"] = raw
            row["response"] = parsed.get("content") or ""
            if parsed.get("thinking"):
                row["thinking"] = parsed["thinking"]
            if parsed.get("tool_calls"):
                row["tool_calls"] = parsed["tool_calls"]
        # Clean the (chat-templated) prompt for display, keeping the raw.
        raw_prompt = row.get("prompt")
        if isinstance(raw_prompt, str) and raw_prompt:
            cleaned_prompt = _clean_prompt(raw_prompt)
            if cleaned_prompt and cleaned_prompt != raw_prompt:
                row["raw_prompt"] = raw_prompt
                row["prompt"] = cleaned_prompt


def _numeric_tags(samples: list[TrainingRolloutSample]) -> dict[str, list[float]]:
    values_by_tag: dict[str, list[float]] = {}
    for sample in samples:
        for key, value in sample.metadata.items():
            if key in _NON_TAG_METADATA_KEYS:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values_by_tag.setdefault(key, []).append(float(value))
    return values_by_tag


class TrainingRolloutSummary(BaseModel):
    """Lightweight rollout data returned by the run-rollouts list endpoint."""

    training_run_id: str
    rollout_id: int
    created_at: int
    total: int
    episode_count: int | None = None
    mean: float
    export_size_bytes: int | None = None
    rollout_time: float | None = None
    error_summary: dict[str, Any] | None = None
    tag_stats: dict[str, dict[str, Any]] | None = None


class TrainingRolloutResult(BaseModel):
    """All samples from one rollout of one training run."""

    training_run_id: str
    rollout_id: int
    created_at: int = 0
    n_samples_per_prompt: int | None = None
    samples: list[TrainingRolloutSample] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    rollout_time: float | None = None

    def _rollout_groups(self) -> list[list[TrainingRolloutSample]]:
        groups: dict[tuple[str, int], list[TrainingRolloutSample]] = {}
        for position, sample in enumerate(self.samples):
            index = sample.rollout_index
            if index is None:
                index = optional_int(sample.metadata.get("rollout_id"))
            key = ("rollout", index) if index is not None else ("sample", position)
            groups.setdefault(key, []).append(sample)
        return list(groups.values())

    @property
    def total(self) -> int:
        return len(self.samples)

    @property
    def episode_count(self) -> int:
        return len(self._rollout_groups())

    @property
    def mean(self) -> float:
        groups = self._rollout_groups()
        if not groups:
            return 0.0
        return sum(sum(s.score for s in group) / len(group) for group in groups) / len(
            groups
        )

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
        """Per-tag count/mean/min/max, weighted per rollout like ``mean``."""
        values_by_tag: dict[str, list[float]] = {}
        for group in self._rollout_groups():
            for tag, values in _numeric_tags(group).items():
                values_by_tag.setdefault(tag, []).append(sum(values) / len(values))

        return {
            tag: {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for tag, values in values_by_tag.items()
        }

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "training_run_id": self.training_run_id,
            "rollout_id": self.rollout_id,
            "created_at": self.created_at,
            "total": self.total,
            "episode_count": self.episode_count,
            "mean": self.mean,
        }
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

    def _summary_item(self, *, export_size_bytes: int) -> dict[str, Any]:
        # summary_key keeps (run_id, rollout_id) uniqueness across runs.
        return {
            **self.to_summary(),
            "export_size_bytes": export_size_bytes,
            "summary_key": self.storage_key,
        }

    def save(self, *, is_async: bool = False) -> None | Awaitable[None]:
        self._touch_created_at()
        payload = self.model_dump(mode="json")
        export_payload = copy.deepcopy(payload)
        _apply_parsed(export_payload.get("samples"))
        export_size_bytes = len(
            (json.dumps(export_payload, ensure_ascii=False, indent=2) + "\n").encode()
        )
        return vol_put_with_summary(
            MetadataStore.TRAINING_ROLLOUTS,
            self.storage_key,
            payload,
            summary_store=MetadataStore.TRAINING_ROLLOUTS_SUMMARY,
            summary_item=self._summary_item(export_size_bytes=export_size_bytes),
            item_id_key="summary_key",
            sort_key=self._summary_sort_key,
            reverse=False,
            is_async=is_async,
        )

    @classmethod
    def list_summaries_for_run(
        cls, training_run_id: str
    ) -> list[TrainingRolloutSummary]:
        """Lightweight per-rollout summaries for one run, sorted by rollout_id."""
        items = vol_get_summary_items(MetadataStore.TRAINING_ROLLOUTS_SUMMARY) or []
        summaries: list[TrainingRolloutSummary] = []
        for item in items:
            if (
                not isinstance(item, dict)
                or item.get("training_run_id") != training_run_id
            ):
                continue
            try:
                summaries.append(TrainingRolloutSummary.model_validate(item))
            except ValidationError:
                continue
        return sorted(summaries, key=lambda summary: summary.rollout_id)
