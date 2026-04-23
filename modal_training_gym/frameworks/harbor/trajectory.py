"""Uniform trajectory interface for Harbor agents.

Every Harbor agent records its interaction as a list of ``TrajectoryTurn``
and stores them via ``context.metadata["trajectory_messages"]``.  The
observability layer consumes these for structured turn-by-turn views.

Single-turn agents produce one user turn and one assistant turn.
Multi-turn agents append a turn for each message in the conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)


def parse_thinking(text: str) -> tuple[str | None, str]:
    """Split a leading ``<think>`` block from *text*.

    Returns ``(reasoning, remaining_content)``.  If no block is found,
    returns ``(None, text)`` unchanged.
    """
    if not text:
        return None, text
    m = _THINK_RE.match(text)
    if m:
        return m.group(1).strip() or None, text[m.end() :]
    return None, text


@dataclass
class TrajectoryTurn:
    """A single turn in an agent trajectory.

    Attributes:
        role: One of ``"user"``, ``"assistant"``, ``"tool"``, ``"system"``.
        content: The text content of this turn (thinking stripped out).
        reasoning: Extracted ``<think>`` content, if any.
        name: Tool name for tool-result turns.
        tool_calls: List of ``{"name": ..., "arguments": ...}`` dicts
            for assistant turns that invoke tools.
        tool_call_id: Identifier linking a tool-result turn to its call.
        duration_sec: Wall-clock time for this turn (typically LLM latency).
        token_usage: ``{"reasoning": N, "answer": N}`` token counts.
        t_start: Wall-clock timestamp when the turn began.
        t_end: Wall-clock timestamp when the turn ended.
    """

    role: str
    content: str = ""
    reasoning: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    duration_sec: float | None = None
    token_usage: dict[str, int] | None = None
    t_start: float | None = None
    t_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the dict format consumed by the observability layer."""
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "reasoning": self.reasoning,
            "name": self.name,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "duration_sec": self.duration_sec,
            "token_usage": self.token_usage,
        }
        if self.t_start is not None:
            d["_t_start"] = self.t_start
        if self.t_end is not None:
            d["_t_end"] = self.t_end
        return d
