from __future__ import annotations

import re
from difflib import SequenceMatcher


_COORD = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)"
_COORD_PAIR = re.compile(rf"(?<![\d.+-])({_COORD})\s*,\s*({_COORD})(?![\d.eE])")


def _parse_coordinates(text: str) -> tuple[float, float] | None:
    match = _COORD_PAIR.search(text)
    if match is None:
        return None
    try:
        x, y = float(match.group(1)), float(match.group(2))
    except ValueError:
        return None
    if 0 <= x <= 1 and 0 <= y <= 1:
        return (x, y)
    return None


def _parse_bbox(label: str) -> tuple[float, float, float, float]:
    left, top, right, bottom = (float(v) for v in label.split(","))
    return left, top, right, bottom


def _distance_outside_box(
    x: float, y: float, box: tuple[float, float, float, float]
) -> float:
    left, top, right, bottom = box
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return (dx * dx + dy * dy) ** 0.5


async def grounding_reward(args, sample, **kwargs) -> float:
    response = getattr(sample, "response", "") or ""
    label = getattr(sample, "label", "") or ""
    pred = _parse_coordinates(response)
    if pred is None:
        return -1.0
    box = _parse_bbox(label)
    left, top, right, bottom = box
    outside = _distance_outside_box(pred[0], pred[1], box)
    if outside == 0.0:
        return 1.0
    diag = ((right - left) ** 2 + (bottom - top) ** 2) ** 0.5
    margin = max(diag, 0.05)
    if outside >= margin:
        return -1.0
    return 1.0 - 2.0 * outside / margin


async def transcript_reward(args, sample, **kwargs) -> float:
    response = (getattr(sample, "response", "") or "").lower().strip()
    label = (getattr(sample, "label", "") or "").lower().strip()
    if not label:
        return 0.0
    return float(SequenceMatcher(None, label, response).ratio())
