from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from modal_training_gym.common.models.base import ParsedResponse


class TraceSpan(BaseModel):
    """One span or instant event in a sample's execution trace.

    A duration span has both ``start`` and ``end`` (seconds, rebased so the
    sample's first span starts at 0); an instant event has ``end is None``.
    ``attributes`` carries timings/counts only — never response or tool
    payloads, which already live on the Sample — so traces stay small (see the
    recorder's normalizer). ``parent`` is the enclosing span's name, if any.
    """

    name: str = ""
    start: float = 0.0
    end: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    parent: str | None = None


class RewardEvent(BaseModel):
    """One incremental or checkpoint-window reward emitted during a sample.

    ``reward`` is the credit assigned to this event/window, not the final
    sample score. Token offsets are relative to the sample response/loss-mask
    sequence, when the environment can provide them. Keeping this separate
    from ``Sample.score`` lets dashboards inspect shaped rewards without
    changing the scalar reward contract used by current trainers.
    """

    turn: int | None = None
    reward: float = 0.0
    cumulative_reward: float | None = None
    components: dict[str, float] = Field(default_factory=dict)
    token_start: int | None = None
    token_end: int | None = None
    label: str | None = None


class Sample(BaseModel):
    """One model interaction: the prompt, the raw response, its parsed
    structure (thinking / answer / tool calls), a score, and free-form
    metadata.

    Shared by eval rows (``EvalResult.rows``) and training rollout samples
    (``TrainingRolloutResult.samples``) — they were the same shape, so this is
    the single canonical type for both.
    """

    score: float = 0.0
    prompt: str = ""
    response: str = ""
    parsed_response: ParsedResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Optional transition-level observability. This is deliberately distinct
    # from ``score``: current trainers still consume one scalar per sample.
    reward_granularity: str | None = None
    reward_events: list[RewardEvent] | None = None
    # captured only when trace recording is enabled and only for a sampled
    # subset of each rollout's samples. ``None`` for untraced samples.
    trace: list[TraceSpan] | None = None
