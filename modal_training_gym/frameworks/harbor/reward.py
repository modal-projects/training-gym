"""Reward helpers for Harbor-backed Miles runs.

Loaded on the remote Miles image via ``--custom-rm-path``. Not importable
locally — requires ``miles`` (installed on the remote image).
"""

from __future__ import annotations

from miles.utils.types import Sample


async def reward_func(
    args, samples: Sample | list[Sample], **kwargs
) -> float | list[float]:
    """Extract the reward set by the Harbor agent function."""
    if isinstance(samples, list):
        return [float(s.metadata.get("reward", 0.0)) for s in samples]
    return float(samples.metadata.get("reward", 0.0))
