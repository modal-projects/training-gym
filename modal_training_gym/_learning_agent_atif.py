"""Convert an observatory run record into a Harbor ATIF trajectory.

ATIF (Agent Trajectory Interchange Format, ATIF-v1.7) is the Harbor
framework's JSON spec for full agent interaction histories:
https://www.harborframework.com/docs/agents/trajectory-format

Mapping from the observatory's normalized events (observatory/schema.py):

    assistant event          -> one step, source "agent": text blocks join
                                into `message`, thinking blocks into
                                `reasoning_content`, tool_use blocks into
                                `tool_calls`, usage into `metrics`
    user event (tool_result) -> ObservationResult appended to the step that
                                issued the matching tool_call (ATIF folds
                                observations onto the calling step)
    user event (text)        -> one step, source "user"
    system event             -> one step, source "system"
    result event             -> one step, source "system" with the final
                                result text; cost feeds final_metrics

Step ids are re-numbered sequentially from 1 as the validator requires, so
they do NOT match the observatory event indices (an event index maps to at
most one step, and tool-result events collapse into earlier steps). Events
carry no timestamps in some LAB trace formats; the last known timestamp
(falling back to the run's launch time) is used so every step still has an
ISO 8601 value.
"""

from __future__ import annotations

import json
from typing import Any

ATIF_SCHEMA_VERSION = "ATIF-v1.7"

_EPOCH_ISO = "1970-01-01T00:00:00Z"

# Observatory usage keys -> ATIF per-step Metrics keys (both spellings the
# normalizer may keep, Anthropic- and OpenAI-style).
_USAGE_TO_METRICS = {
    "input_tokens": "prompt_tokens",
    "prompt_tokens": "prompt_tokens",
    "output_tokens": "completion_tokens",
    "completion_tokens": "completion_tokens",
    "cache_read_input_tokens": "cached_tokens",
    "cached_tokens": "cached_tokens",
}


def _text_of(value: Any) -> str:
    """Flatten a block content payload (str or content-part list) to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    if value is None:
        return ""
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _metrics_from_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    metrics: dict[str, Any] = {}
    for src_key, dst_key in _USAGE_TO_METRICS.items():
        value = usage.get(src_key)
        if isinstance(value, int) and dst_key not in metrics:
            metrics[dst_key] = value
    # Per-step Metrics requires prompt/completion tokens; skip partial data.
    if "prompt_tokens" in metrics and "completion_tokens" in metrics:
        return metrics
    return None


def events_to_atif(record: dict[str, Any], events: list[Any]) -> dict[str, Any]:
    """Build an ATIF trajectory dict from a run record and its events."""
    meta = record.get("meta") or {}
    index_row = record.get("index_row") or {}
    run_id = str(index_row.get("run_id") or meta.get("run_id") or "unknown")

    steps: list[dict[str, Any]] = []
    # Tool-call registrations. Trace formats that resume sessions (codex)
    # reuse call ids like "item_15" across sessions, so results resolve by
    # (session_idx, id) first and fall back to the latest bare-id match.
    call_step_by_session: dict[tuple[Any, str], int] = {}
    call_step_by_id: dict[str, int] = {}
    last_ts = str(meta.get("launched_at") or _EPOCH_ISO)
    total_cost: float | None = None

    for event in events:
        if not isinstance(event, dict):
            continue
        ts = event.get("ts") or last_ts
        last_ts = ts
        etype = event.get("type")
        blocks = [b for b in (event.get("blocks") or []) if isinstance(b, dict)]

        if etype == "assistant":
            texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
            thinking = [
                b.get("thinking", "") for b in blocks if b.get("type") == "thinking"
            ]
            tool_calls = []
            for block in blocks:
                if block.get("type") != "tool_use":
                    continue
                call_id = str(block.get("id") or f"call_{len(call_step_by_id)}")
                arguments = block.get("input")
                tool_calls.append(
                    {
                        "tool_call_id": call_id,
                        "function_name": str(block.get("name") or "tool"),
                        "arguments": arguments if isinstance(arguments, dict) else {},
                    }
                )
                call_step_by_session[(event.get("session_idx"), call_id)] = len(steps)
                call_step_by_id[call_id] = len(steps)
            step: dict[str, Any] = {
                "timestamp": ts,
                "source": "agent",
                "message": "\n\n".join(t for t in texts if t),
            }
            if thinking and any(thinking):
                step["reasoning_content"] = "\n\n".join(t for t in thinking if t)
            if event.get("model"):
                step["model_name"] = str(event["model"])
            if tool_calls:
                step["tool_calls"] = tool_calls
            metrics = _metrics_from_usage(event.get("usage"))
            if metrics:
                step["metrics"] = metrics
            steps.append(step)

        elif etype == "user":
            orphan_texts = [
                b.get("text", "") for b in blocks if b.get("type") == "text"
            ]
            for block in blocks:
                if block.get("type") != "tool_result":
                    continue
                content = _text_of(block.get("content"))
                call_id = str(block.get("tool_use_id") or "")
                target = call_step_by_session.get((event.get("session_idx"), call_id))
                if target is None:
                    target = call_step_by_id.get(call_id)
                if target is None:
                    # No matching call in the trace (e.g. window truncation
                    # upstream): keep the output as plain user input.
                    orphan_texts.append(content)
                    continue
                result: dict[str, Any] = {
                    "source_call_id": str(block.get("tool_use_id")),
                    "content": content,
                }
                if block.get("is_error"):
                    result["extra"] = {"is_error": True}
                observation = steps[target].setdefault("observation", {"results": []})
                observation["results"].append(result)
            if any(orphan_texts):
                steps.append(
                    {
                        "timestamp": ts,
                        "source": "user",
                        "message": "\n\n".join(t for t in orphan_texts if t),
                    }
                )

        elif etype == "result":
            if isinstance(event.get("total_cost_usd"), (int, float)):
                total_cost = float(event["total_cost_usd"])
            steps.append(
                {
                    "timestamp": ts,
                    "source": "system",
                    "message": _text_of(event.get("result"))
                    or f"session ended ({event.get('subtype') or 'result'})",
                    "extra": {"event_kind": "result"},
                }
            )

        else:  # system / unknown
            raw = event.get("raw")
            message = event.get("subtype") or etype or "system"
            steps.append(
                {
                    "timestamp": ts,
                    "source": "system",
                    "message": str(message),
                    **({"extra": {"raw": raw}} if isinstance(raw, dict) else {}),
                }
            )

    for i, step in enumerate(steps):
        step["step_id"] = i + 1

    prompt_total = sum(
        s.get("metrics", {}).get("prompt_tokens", 0) or 0 for s in steps
    )
    completion_total = sum(
        s.get("metrics", {}).get("completion_tokens", 0) or 0 for s in steps
    )
    final_metrics: dict[str, Any] = {
        "total_prompt_tokens": prompt_total,
        "total_completion_tokens": completion_total,
        "total_steps": len(steps),
    }
    if total_cost is not None:
        final_metrics["total_cost_usd"] = total_cost

    session_ids = (record.get("summary") or {}).get("session_ids") or []

    return {
        "schema_version": ATIF_SCHEMA_VERSION,
        "session_id": run_id,
        "agent": {
            "name": str(meta.get("scaffold") or "learning-agent"),
            "version": str(meta.get("trace_format") or "unknown"),
            "model_name": str(
                index_row.get("agent_model")
                or meta.get("base_model")
                or "unknown"
            ),
        },
        "steps": steps,
        "final_metrics": final_metrics,
        "extra": {
            "run_id": run_id,
            "task": index_row.get("task"),
            "track": index_row.get("track"),
            "trace_format": meta.get("trace_format"),
            "session_ids": session_ids,
            "source": "lab-observatory record.json events",
        },
    }
