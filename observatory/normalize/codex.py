"""Normalize Codex CLI JSONL traces into schema.py events.

Handles both codex vocabularies seen in the wild:
  - legacy event stream: {"id": ..., "msg": {"type": "agent_message", ...}}
  - `codex exec --json`: {"type": "thread.started"|"item.completed"|..., "item": {...}}
Mapping: agent_reasoning -> thinking, agent_message -> text,
exec_command_begin -> tool_use{name:"command"}, exec_command_end -> tool_result,
token_count -> usage attached to the previous assistant event; everything else
passes through as {type:"system", subtype:"unknown"} with trimmed raw.
"""

from __future__ import annotations

import shlex
from typing import Any, Iterable, Optional

from .claude_stream import (TurnCounter, base_event, finalize, iter_lines,
                            pick_ts, trim_raw, unknown_event)

_DELTA_TYPES = {"agent_message_delta", "agent_reasoning_delta",
                "agent_reasoning_raw_content_delta", "exec_command_output_delta"}

# codex usage keys -> the canonical (claude-shaped) names the dashboard reads.
_USAGE_KEY_MAP = {"cached_input_tokens": "cache_read_input_tokens",
                  "cache_write_input_tokens": "cache_creation_input_tokens"}


def _canon_usage(usage: dict) -> dict:
    return {_USAGE_KEY_MAP.get(k, k): v for k, v in usage.items()}


def _cmd_str(command) -> str:
    if isinstance(command, list):
        return " ".join(shlex.quote(str(tok)) for tok in command)
    return str(command)


def _new_vocab(obj: dict) -> Optional[dict]:
    """Lift a `codex exec --json` line into the legacy msg vocabulary."""
    t = obj.get("type")
    item = obj.get("item") or {}
    it = item.get("item_type") or item.get("type")
    if t == "thread.started":
        return {"type": "session_configured",
                "session_id": obj.get("thread_id"), "model": obj.get("model"),
                "cwd": obj.get("cwd")}
    if t == "turn.completed":
        return {"type": "token_count", "usage": obj.get("usage")}
    if t == "item.started" and it == "command_execution":
        return {"type": "exec_command_begin", "call_id": item.get("id"),
                "command": item.get("command"), "cwd": item.get("cwd")}
    if t == "item.completed":
        if it == "command_execution":
            return {"type": "exec_command_end", "call_id": item.get("id"),
                    "command": item.get("command"),
                    "aggregated_output": item.get("aggregated_output"),
                    "exit_code": item.get("exit_code")}
        if it == "reasoning":
            return {"type": "agent_reasoning", "text": item.get("text")}
        if it in ("agent_message", "assistant_message"):
            return {"type": "agent_message", "message": item.get("text")}
    if t in ("item.updated", "turn.started"):
        return {"type": "_skip"}
    return None


def parse_trace(lines: Iterable[str], line_ts: Optional[dict] = None) -> dict:
    events: list[dict] = []
    sessions: list[dict] = []
    session_idx: Optional[int] = None
    cur_sid: Optional[str] = None
    turns = TurnCounter()
    last_assistant: Optional[dict] = None

    def assistant(i, ts, blocks) -> dict:
        e = base_event(i, ts, "assistant", None, cur_sid, session_idx)
        e["blocks"] = blocks
        e["turn"] = turns.number("assistant")
        return e

    for no, obj, wall, text in iter_lines(lines):
        i = len(events)
        if obj is None:
            events.append(unknown_event(i, pick_ts(line_ts, no, wall),
                                        text, cur_sid, session_idx))
            continue
        msg = obj.get("msg") if isinstance(obj.get("msg"), dict) else None
        if msg is None:
            msg = _new_vocab(obj) or obj
        t = msg.get("type") or ""
        ts = pick_ts(line_ts, no, wall)

        if t == "_skip" or t in _DELTA_TYPES:
            continue  # streaming fragments — the final event carries full content
        if t in ("session_configured", "thread.started"):
            session_idx = 0 if session_idx is None else session_idx + 1
            cur_sid = msg.get("session_id") or msg.get("thread_id")
            sessions.append({
                "session_idx": session_idx, "session_id": cur_sid, "ts_start": ts,
                "model": msg.get("model"), "cwd": msg.get("cwd"),
                "permission_mode": msg.get("approval_policy"), "tools": [],
            })
            e = base_event(i, ts, "system", "init", cur_sid, session_idx)
            e["raw"] = {"cwd": msg.get("cwd"), "model": msg.get("model"),
                        "approval_policy": msg.get("approval_policy")}
            events.append(e)
        elif t in ("agent_reasoning", "agent_reasoning_raw_content"):
            e = assistant(i, ts, [{"type": "thinking",
                                   "thinking": msg.get("text") or ""}])
            events.append(e)
            last_assistant = e
        elif t == "agent_message":
            e = assistant(i, ts, [{"type": "text",
                                   "text": msg.get("message") or msg.get("text") or ""}])
            events.append(e)
            last_assistant = e
        elif t == "user_message":
            e = base_event(i, ts, "user", None, cur_sid, session_idx)
            e["blocks"] = [{"type": "text", "text": msg.get("message") or ""}]
            turns.number("user")
            events.append(e)
        elif t == "exec_command_begin":
            call_id = msg.get("call_id") or f"call_{i}"
            inp: dict[str, Any] = {"command": _cmd_str(msg.get("command") or "")}
            if msg.get("cwd"):
                inp["cwd"] = msg["cwd"]
            e = assistant(i, ts, [{"type": "tool_use", "id": call_id,
                                   "name": "command", "input": inp}])
            events.append(e)
            last_assistant = e
        elif t == "exec_command_end":
            call_id = msg.get("call_id") or ""
            out = msg.get("aggregated_output")
            if out is None:
                out = msg.get("stdout") or ""
                if msg.get("stderr"):
                    out += ("\n" if out else "") + msg["stderr"]
            exit_code = msg.get("exit_code")
            e = base_event(i, ts, "user", None, cur_sid, session_idx)
            e["blocks"] = [{"type": "tool_result", "tool_use_id": call_id,
                            "content": out,
                            "is_error": exit_code not in (0, None)}]
            turns.number("user")
            events.append(e)
        elif t == "mcp_tool_call_begin":
            call_id = msg.get("call_id") or f"call_{i}"
            name = ".".join(x for x in (msg.get("server_name"),
                                        msg.get("tool_name")) if x) or "mcp"
            e = assistant(i, ts, [{"type": "tool_use", "id": call_id, "name": name,
                                   "input": msg.get("arguments") or {}}])
            events.append(e)
            last_assistant = e
        elif t == "mcp_tool_call_end":
            e = base_event(i, ts, "user", None, cur_sid, session_idx)
            e["blocks"] = [{"type": "tool_result",
                            "tool_use_id": msg.get("call_id") or "",
                            "content": msg.get("result"), "is_error": False}]
            turns.number("user")
            events.append(e)
        elif t == "token_count":
            usage = (msg.get("usage") or msg.get("turn")
                     or (msg.get("info") or {}).get("last_token_usage")
                     or msg.get("session")
                     or (msg.get("info") or {}).get("total_token_usage"))
            if isinstance(usage, dict) and last_assistant is not None:
                last_assistant["usage"] = _canon_usage(usage)
        elif t in ("task_complete", "turn_complete"):
            e = base_event(i, ts, "result", t, cur_sid, session_idx)
            if msg.get("last_agent_message"):
                e["result"] = msg["last_agent_message"]
            events.append(e)
        elif t == "error":
            e = base_event(i, ts, "system", "error", cur_sid, session_idx)
            e["raw"] = trim_raw(msg)
            events.append(e)
        else:
            events.append(unknown_event(i, ts, obj, cur_sid, session_idx))

    return finalize(events, sessions)
