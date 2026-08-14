"""Normalize Gemini CLI stream-json traces into schema.py events.

Two shapes exist: stream events ({"type": "init"|"message"|"tool_use"|
"tool_result"|"error"|"result", ...}) and legacy method lines ({"method": ...,
"response": [chunks with candidates/parts]}). Consecutive delta messages of the
same role are consolidated into one event; unknown lines pass through as
{type:"system", subtype:"unknown"} with trimmed raw.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .claude_stream import (TurnCounter, base_event, finalize, iter_lines,
                            pick_ts, trim_raw, unknown_event)


def _parts_to_blocks(parts: list, i: int) -> list[dict]:
    blocks: list[dict] = []
    for j, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        if part.get("text"):
            if part.get("thought"):
                blocks.append({"type": "thinking", "thinking": part["text"]})
            else:
                blocks.append({"type": "text", "text": part["text"]})
        if fn := part.get("functionCall"):
            blocks.append({"type": "tool_use",
                           "id": fn.get("id") or f"call_{i}_{j}",
                           "name": fn.get("name") or "tool",
                           "input": fn.get("args") or {}})
        if fr := part.get("functionResponse"):
            resp = fr.get("response")
            content = resp.get("output") if isinstance(resp, dict) and "output" in resp else resp
            blocks.append({"type": "tool_result",
                           "tool_use_id": fr.get("id") or fr.get("name") or "",
                           "content": content, "is_error": False})
    return blocks


def parse_trace(lines: Iterable[str], line_ts: Optional[dict] = None) -> dict:
    events: list[dict] = []
    sessions: list[dict] = []
    session_idx: Optional[int] = None
    cur_sid: Optional[str] = None
    turns = TurnCounter()
    pending: Optional[dict] = None  # accumulating delta-message run

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        role = "assistant" if pending["role"] == "assistant" else "user"
        e = base_event(len(events), pending["ts"], role, None, cur_sid, session_idx)
        e["blocks"] = [{"type": "text", "text": pending["content"]}]
        turn = turns.number(role)
        if turn is not None:
            e["turn"] = turn
        events.append(e)
        pending = None

    for no, obj, wall, text in iter_lines(lines):
        if obj is None:
            flush()
            events.append(unknown_event(len(events), pick_ts(line_ts, no, wall),
                                        text, cur_sid, session_idx))
            continue
        ts = pick_ts(line_ts, no, wall, obj.get("timestamp"))
        t = obj.get("type")

        if t == "message" and obj.get("delta"):
            role = obj.get("role", "assistant")
            if pending and pending["role"] != role:
                flush()
            if pending:
                pending["content"] += obj.get("content") or ""
            else:
                pending = {"role": role, "content": obj.get("content") or "", "ts": ts}
            continue
        flush()
        i = len(events)

        if t == "init":
            session_idx = 0 if session_idx is None else session_idx + 1
            cur_sid = obj.get("session_id")
            sessions.append({"session_idx": session_idx, "session_id": cur_sid,
                             "ts_start": ts, "model": obj.get("model"),
                             "cwd": obj.get("cwd"), "permission_mode": None,
                             "tools": []})
            e = base_event(i, ts, "system", "init", cur_sid, session_idx)
            e["raw"] = {"cwd": obj.get("cwd"), "model": obj.get("model")}
            events.append(e)
        elif t == "message":
            role = "assistant" if obj.get("role", "assistant") == "assistant" else "user"
            e = base_event(i, ts, role, None, cur_sid, session_idx)
            e["blocks"] = [{"type": "text", "text": obj.get("content") or ""}]
            turn = turns.number(role)
            if turn is not None:
                e["turn"] = turn
            events.append(e)
        elif t == "tool_use":
            e = base_event(i, ts, "assistant", None, cur_sid, session_idx)
            e["blocks"] = [{"type": "tool_use",
                            "id": obj.get("tool_id") or f"call_{i}",
                            "name": obj.get("tool_name") or "tool",
                            "input": obj.get("parameters") or {}}]
            e["turn"] = turns.number("assistant")
            events.append(e)
        elif t == "tool_result":
            content = obj.get("output")
            if content is None and obj.get("error") is not None:
                content = str(obj["error"])
            e = base_event(i, ts, "user", None, cur_sid, session_idx)
            e["blocks"] = [{"type": "tool_result",
                            "tool_use_id": obj.get("tool_id") or "",
                            "content": content,
                            "is_error": obj.get("status") == "error"
                                        or obj.get("error") is not None}]
            turns.number("user")
            events.append(e)
        elif t == "result":
            e = base_event(i, ts, "result", obj.get("status") or "result",
                           cur_sid, session_idx)
            stats = obj.get("stats") or {}
            if isinstance(stats.get("duration_ms"), int):
                e["duration_ms"] = stats["duration_ms"]
            usage = {k: v for k, v in stats.items()
                     if isinstance(v, int) and not isinstance(v, bool)
                     and k != "duration_ms"}
            if usage:
                e["usage"] = usage
            events.append(e)
        elif t == "error":
            e = base_event(i, ts, "system", "error", cur_sid, session_idx)
            e["raw"] = trim_raw(obj)
            events.append(e)
        elif "method" in obj:
            # legacy JSON-RPC-ish line: hoist candidate parts + usage
            blocks: list[dict] = []
            usage: Optional[dict] = None
            chunks = obj.get("response")
            if isinstance(chunks, dict):
                chunks = [chunks]
            for chunk in chunks or []:
                if not isinstance(chunk, dict):
                    continue
                for cand in chunk.get("candidates") or []:
                    parts = (cand.get("content") or {}).get("parts") or []
                    blocks.extend(_parts_to_blocks(parts, i))
                um = chunk.get("usageMetadata")
                if isinstance(um, dict):
                    usage = {k: v for k, v in um.items()
                             if isinstance(v, int) and not isinstance(v, bool)}
            if blocks:
                e = base_event(i, ts, "assistant", None, cur_sid, session_idx)
                e["blocks"] = blocks
                if usage:
                    e["usage"] = usage
                e["turn"] = turns.number("assistant")
                events.append(e)
            else:
                events.append(unknown_event(i, ts, obj, cur_sid, session_idx))
        else:
            events.append(unknown_event(i, ts, obj, cur_sid, session_idx))

    flush()
    return finalize(events, sessions)
