"""Normalize OpenCode --format json traces into schema.py events.

Event shapes (see agents/opencode/human_readable_trace.py): {"type": "text"|
"tool_use"|"step_start"|"step_finish"|"error", "part": {...}, "timestamp": ms}.
A completed tool_use carries both input and output, so it expands into a
tool_use assistant event plus a tool_result user event. step_finish token
counts attach to the previous assistant event (like codex token_count).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from .claude_stream import (TurnCounter, base_event, finalize, iter_lines,
                            pick_ts, trim_raw, unknown_event)


def _ms_iso(ms) -> Optional[str]:
    if not isinstance(ms, (int, float)):
        return None
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_trace(lines: Iterable[str], line_ts: Optional[dict] = None) -> dict:
    events: list[dict] = []
    sessions: list[dict] = []
    turns = TurnCounter()
    last_assistant: Optional[dict] = None

    for no, obj, wall, text in iter_lines(lines):
        i = len(events)
        if obj is None:
            events.append(unknown_event(i, pick_ts(line_ts, no, wall),
                                        text, None, None))
            continue
        ts = pick_ts(line_ts, no, wall, _ms_iso(obj.get("timestamp")))
        t = obj.get("type")
        part = obj.get("part") or {}

        if t == "text":
            e = base_event(i, ts, "assistant", None, None, None)
            e["blocks"] = [{"type": "text", "text": part.get("text") or ""}]
            e["turn"] = turns.number("assistant")
            events.append(e)
            last_assistant = e
        elif t == "reasoning":
            e = base_event(i, ts, "assistant", None, None, None)
            e["blocks"] = [{"type": "thinking", "thinking": part.get("text") or ""}]
            e["turn"] = turns.number("assistant")
            events.append(e)
            last_assistant = e
        elif t == "tool_use":
            state = part.get("state") or {}
            call_id = part.get("id") or part.get("callID") or f"call_{i}"
            e = base_event(i, ts, "assistant", None, None, None)
            e["blocks"] = [{"type": "tool_use", "id": call_id,
                            "name": part.get("tool") or "tool",
                            "input": state.get("input") or {}}]
            e["turn"] = turns.number("assistant")
            events.append(e)
            last_assistant = e
            status = state.get("status")
            if status in ("completed", "error"):
                r = base_event(len(events), ts, "user", None, None, None)
                content = state.get("error") if status == "error" else state.get("output")
                r["blocks"] = [{"type": "tool_result", "tool_use_id": call_id,
                                "content": content or "",
                                "is_error": status == "error"}]
                turns.number("user")
                events.append(r)
        elif t == "step_finish":
            # tokens/cost ride on the previous assistant event, no event of its own
            tokens = part.get("tokens") or {}
            usage: dict[str, Any] = {}
            for src, dst in (("input", "input_tokens"), ("output", "output_tokens"),
                             ("reasoning", "reasoning_tokens")):
                if isinstance(tokens.get(src), int):
                    usage[dst] = tokens[src]
            cache = tokens.get("cache") or {}
            if isinstance(cache.get("read"), int):
                usage["cache_read_input_tokens"] = cache["read"]
            if isinstance(cache.get("write"), int):
                usage["cache_creation_input_tokens"] = cache["write"]
            if usage and last_assistant is not None:
                last_assistant["usage"] = usage
        elif t == "step_start":
            continue
        elif t == "error":
            e = base_event(i, ts, "system", "error", None, None)
            e["raw"] = trim_raw(obj)
            events.append(e)
        else:
            events.append(unknown_event(i, ts, obj, None, None))

    return finalize(events, sessions)
