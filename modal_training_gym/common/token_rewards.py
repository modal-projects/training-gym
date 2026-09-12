"""Build response-token reward vectors from transition reward events.

The rollout frameworks still expose one scalar reward for compatibility.  This
module is the small adapter used by the Slime image patch: when a sample has
transition events, it preserves the scalar total while also placing each event
on the action token that produced it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _event_values(sample: Any) -> Sequence[Any] | None:
    events = getattr(sample, "reward_events", None)
    if events is None:
        metadata = getattr(sample, "metadata", None) or {}
        events = metadata.get("reward_events")
        if events is None:
            balatro = metadata.get("balatro")
            if isinstance(balatro, Mapping):
                events = balatro.get("partial_reward_trace")
    return (
        events
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes))
        else None
    )


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _active_positions(loss_mask: Any, length: int) -> list[int]:
    if loss_mask is None:
        return list(range(length))
    try:
        return [
            index
            for index, value in enumerate(loss_mask[:length])
            if float(value) != 0.0
        ]
    except (TypeError, ValueError):
        return list(range(length))


def _event_position(
    event: Mapping[str, Any], active: list[int], length: int
) -> int | None:
    # Offsets are response-relative and token_end is exclusive.  The reward is
    # attached to the final token of the action span because that is the token
    # whose sampled action caused the transition.
    end = event.get("token_end")
    start = event.get("token_start")
    try:
        position = int(end) - 1 if end is not None else int(start)
    except (TypeError, ValueError):
        position = None
    if position is None or position < 0 or position >= length:
        return active[-1] if active else None
    if position in active:
        return position
    earlier = [candidate for candidate in active if candidate <= position]
    return earlier[-1] if earlier else (active[0] if active else None)


def build_token_reward_vectors(
    samples: Sequence[Any], target_rewards: Sequence[Any]
) -> list[list[float] | None] | None:
    """Return one response-length reward vector per sample when events exist.

    ``target_rewards`` are the framework's post-processed scalar rewards.  The
    event deltas remain in their original units; a residual is added to the
    final active token so every vector sums to that post-processed scalar.  A
    sample without events returns ``None`` and can therefore retain the
    framework's normal scalar-reward behavior in a mixed batch.
    """
    vectors: list[list[float] | None] = []
    found_events = False
    for index, sample in enumerate(samples):
        events = _event_values(sample)
        if not events:
            vectors.append(None)
            continue
        found_events = True
        loss_mask = getattr(sample, "loss_mask", None)
        length = (
            len(loss_mask)
            if loss_mask is not None
            else int(getattr(sample, "response_length", 0) or 0)
        )
        vector = [0.0] * length
        active = _active_positions(loss_mask, length)
        event_total = 0.0
        for raw_event in events:
            event = _as_mapping(raw_event)
            if event is None:
                continue
            reward = _finite_float(event.get("reward", event.get("total", 0.0)))
            if reward is None:
                continue
            position = _event_position(event, active, length)
            if position is not None:
                vector[position] += reward
                event_total += reward

        target = (
            _finite_float(target_rewards[index])
            if index < len(target_rewards)
            else None
        )
        if target is None:
            target = _finite_float(getattr(sample, "reward", 0.0)) or 0.0
        if active:
            vector[active[-1]] += target - event_total
        vectors.append(vector)
    return vectors if found_events else None
