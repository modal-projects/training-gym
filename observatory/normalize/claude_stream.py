"""Normalize Claude Code stream-json traces into schema.py events.

Also home to the small helpers shared by the codex/gemini/opencode
normalizers (line iteration, raw trimming, summary aggregation) so every
format returns the identical {events, sessions, summary, meta_bits} contract.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Iterator, Optional

TS_PREFIX_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\] ")
RAW_TRIM_BYTES = 2048
FINAL_RESULT_CAP = 4000

CLAUDE_TYPES = ("system", "assistant", "user", "result")


# ---- shared plumbing (used by all four normalizers) ----

def iter_lines(lines: Iterable[str]) -> Iterator[tuple[int, Optional[dict], Optional[str], str]]:
    """Yield (line_no, parsed_dict_or_None, wall_ts, text) per non-empty line.

    line_no is 1-based over ALL input lines (blank ones included) so it joins
    against the watcher's line_ts sidecar. An optional "[ISO] " prefix (from
    older timestamped captures) is stripped and surfaced as wall_ts.
    """
    for no, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s:
            continue
        wall = None
        m = TS_PREFIX_RE.match(s)
        if m:
            wall = m.group(1)
            s = s[m.end():]
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            obj = None
        if not isinstance(obj, dict):
            obj = None
        yield no, obj, wall, s


def pick_ts(line_ts: Optional[dict], line_no: int, wall: Optional[str],
            own: Optional[str] = None) -> Optional[str]:
    """Sidecar arrival time wins; then a [ts] prefix; then the event's own."""
    if line_ts:
        ts = line_ts.get(line_no)
        if ts:
            return ts
    return wall or own


def trim_raw(obj: dict, limit: int = RAW_TRIM_BYTES) -> dict:
    s = json.dumps(obj, ensure_ascii=False)
    if len(s) <= limit:
        return obj
    return {"type": obj.get("type"), "_trimmed": s[:limit]}


def base_event(i: int, ts: Optional[str], etype: str, subtype: Optional[str],
               session_id: Optional[str], session_idx: Optional[int],
               parent: Optional[str] = None) -> dict:
    return {"i": i, "ts": ts, "type": etype, "subtype": subtype,
            "session_id": session_id, "session_idx": session_idx,
            "parent_tool_use_id": parent}


def unknown_event(i: int, ts: Optional[str], obj_or_text,
                  session_id: Optional[str], session_idx: Optional[int]) -> dict:
    e = base_event(i, ts, "system", "unknown", session_id, session_idx)
    if isinstance(obj_or_text, dict):
        e["raw"] = trim_raw(obj_or_text)
    else:
        e["raw"] = {"unparsable": str(obj_or_text)[:RAW_TRIM_BYTES]}
    return e


def _sum_usage(usages: list[dict]) -> dict:
    total: dict[str, int] = {}
    for u in usages:
        for k, v in u.items():
            if isinstance(v, int) and not isinstance(v, bool):
                total[k] = total.get(k, 0) + v
    return total


def _ts_span_ms(events: list[dict]) -> Optional[int]:
    tss = sorted(e["ts"] for e in events if e.get("ts"))
    if len(tss) < 2:
        return None
    try:
        from datetime import datetime
        a = datetime.fromisoformat(tss[0].replace("Z", "+00:00"))
        b = datetime.fromisoformat(tss[-1].replace("Z", "+00:00"))
        return max(0, int((b - a).total_seconds() * 1000))
    except ValueError:
        return None


def summarize(events: list[dict], sessions: list[dict]) -> dict:
    models: list[str] = []
    stop_reasons: list[str] = []
    for e in events:
        m = e.get("model")
        if m and m not in models:
            models.append(m)
        sr = e.get("stop_reason")
        if sr and sr not in stop_reasons:
            stop_reasons.append(sr)
    for s in sessions:
        m = s.get("model")
        if m and m not in models:
            models.append(m)
    first = sessions[0] if sessions else {}
    summary = {
        "agent_models": models,
        "tools_offered": list(first.get("tools") or []),
        "permission_mode": first.get("permission_mode"),
        "cwd": first.get("cwd"),
        "num_turns": None,
        "duration_ms": None,
        "total_cost_usd": None,
        "usage_total": {},
        "stop_reasons": stop_reasons,
        "final_result_text": None,
        "session_count": len(sessions),
        "session_ids": [s.get("session_id") for s in sessions],
    }
    # A trace holds one result event per CLI session (claude) — or none at all
    # (codex: task_complete carries no usage). Aggregate across ALL sessions,
    # falling back to per-turn usage attached to assistant events.
    results = [e for e in events if e["type"] == "result"]
    if results:
        txt = results[-1].get("result")
        if isinstance(txt, str):
            summary["final_result_text"] = txt[:FINAL_RESULT_CAP]
    usage_total = _sum_usage([e["usage"] for e in results
                              if isinstance(e.get("usage"), dict)])
    if not usage_total:
        usage_total = _sum_usage([e["usage"] for e in events
                                  if e["type"] == "assistant"
                                  and isinstance(e.get("usage"), dict)])
    summary["usage_total"] = usage_total
    turns = [e["num_turns"] for e in results if isinstance(e.get("num_turns"), int)]
    if turns:
        summary["num_turns"] = sum(turns)
    else:
        max_turn = max((e.get("turn") or 0 for e in events), default=0)
        summary["num_turns"] = max_turn or None
    durs = [e["duration_ms"] for e in results
            if isinstance(e.get("duration_ms"), (int, float))]
    summary["duration_ms"] = sum(durs) if durs else _ts_span_ms(events)
    costs = [e["total_cost_usd"] for e in results
             if isinstance(e.get("total_cost_usd"), (int, float))]
    if costs:
        summary["total_cost_usd"] = round(sum(costs), 6)
    return summary


def meta_bits_from(events: list[dict]) -> dict:
    tss = [e["ts"] for e in events if e.get("ts")]
    return {"launched_at": tss[0] if tss else None,
            "finished_at": tss[-1] if tss else None}


def finalize(events: list[dict], sessions: list[dict]) -> dict:
    return {"events": events, "sessions": sessions,
            "summary": summarize(events, sessions),
            "meta_bits": meta_bits_from(events)}


class TurnCounter:
    """1-based turn numbers on assistant events; a run of assistant events is
    one turn, broken only by an intervening user event (system/result lines,
    e.g. rate_limit_event, are transparent — mirrors fixtures/make_demo_run)."""

    def __init__(self) -> None:
        self.turn = 0
        self.prev_role: Optional[str] = None

    def number(self, etype: str) -> Optional[int]:
        out = None
        if etype == "assistant":
            if self.prev_role != "assistant":
                self.turn += 1
            out = self.turn
        if etype in ("assistant", "user"):
            self.prev_role = etype
        return out


# ---- claude stream-json ----

def parse_trace(lines: Iterable[str], line_ts: Optional[dict] = None) -> dict:
    events: list[dict] = []
    sessions: list[dict] = []
    session_idx: Optional[int] = None
    cur_sid: Optional[str] = None
    turns = TurnCounter()

    for no, obj, wall, text in iter_lines(lines):
        i = len(events)
        if obj is None:
            ts = pick_ts(line_ts, no, wall)
            events.append(unknown_event(i, ts, text, cur_sid, session_idx))
            continue
        ts = pick_ts(line_ts, no, wall)
        t = obj.get("type")
        sid = obj.get("session_id") or cur_sid

        if t == "system" and obj.get("subtype") == "init":
            session_idx = 0 if session_idx is None else session_idx + 1
            cur_sid = sid
            tools = obj.get("tools") or []
            sessions.append({
                "session_idx": session_idx, "session_id": sid, "ts_start": ts,
                "model": obj.get("model"), "cwd": obj.get("cwd"),
                "permission_mode": obj.get("permissionMode"),
                "tools": [x.get("name") if isinstance(x, dict) else str(x) for x in tools],
            })
            e = base_event(i, ts, "system", "init", sid, session_idx)
            e["raw"] = {"cwd": obj.get("cwd"), "model": obj.get("model"),
                        "permissionMode": obj.get("permissionMode"),
                        "tools_count": len(tools)}
            events.append(e)
        elif t in ("assistant", "user"):
            cur_sid = sid
            msg = obj.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                blocks: list[Any] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                blocks = content
            else:
                blocks = []
            e = base_event(i, ts, t, obj.get("subtype"), sid, session_idx,
                           obj.get("parent_tool_use_id"))
            e["blocks"] = blocks
            if msg.get("usage") is not None:
                e["usage"] = msg["usage"]
            if msg.get("model"):
                e["model"] = msg["model"]
            if obj.get("uuid"):
                e["uuid"] = obj["uuid"]
            turn = turns.number(t)
            if turn is not None:
                e["turn"] = turn
            events.append(e)
        elif t == "result":
            cur_sid = sid
            e = base_event(i, ts, "result", obj.get("subtype"), sid, session_idx)
            for k in ("duration_ms", "num_turns", "total_cost_usd",
                      "stop_reason", "result", "usage"):
                if obj.get(k) is not None:
                    e[k] = obj[k]
            events.append(e)
        elif t == "system":
            e = base_event(i, ts, "system", obj.get("subtype") or "info",
                           sid, session_idx)
            e["raw"] = trim_raw(obj)
            events.append(e)
        else:
            events.append(unknown_event(i, ts, obj, sid, session_idx))

    return finalize(events, sessions)
