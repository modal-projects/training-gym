"""Custom generate hook for Harbor-backed Miles runs.

Loaded on the remote Miles image via ``--custom-generate-function-path``.
Not importable locally — requires ``miles`` and ``harbor`` on the remote image.
"""

from __future__ import annotations

import os
from typing import Any

from .agent_function import run as run_agent_trial


def _coerce_reward(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_response_text(result: dict[str, Any]) -> str:
    text = str(result.get("observability_response_text") or "").strip()
    if text:
        return text

    trajectory = result.get("trajectory_messages") or []
    for turn in reversed(trajectory):
        if not isinstance(turn, dict) or turn.get("role") != "assistant":
            continue
        reasoning = str(turn.get("reasoning") or "").strip()
        content = str(turn.get("content") or "").strip()
        if reasoning and content:
            return f"<think>{reasoning}</think>\n{content}"
        if reasoning:
            return f"<think>{reasoning}</think>"
        if content:
            return content

    return str(result.get("observability_written_content") or "").strip()


async def generate(args, sample, sampling_params, evaluation: bool = False):
    from miles.rollout.sglang_rollout import GenerateState
    from miles.utils.types import Sample

    state = GenerateState(args)
    prompt_ids = sample.tokens or state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    internal_base_url = os.getenv("HARBOR_INTERNAL_BASE_URL", "").strip()
    if not internal_base_url:
        raise RuntimeError("HARBOR_INTERNAL_BASE_URL is not set")

    request_kwargs = {
        "temperature": sampling_params.get("temperature"),
        "max_tokens": sampling_params.get("max_new_tokens"),
    }

    result = await run_agent_trial(
        base_url=internal_base_url,
        prompt=sample.prompt,
        request_kwargs=request_kwargs,
        metadata=sample.metadata,
        evaluation=evaluation,
    )
    result = result or {}

    response_text = _extract_response_text(result)
    response_token_ids = (
        state.tokenizer.encode(response_text, add_special_tokens=False)
        if response_text
        else []
    )

    sample.metadata = {**getattr(sample, "metadata", {}), **result}
    sample.reward = _coerce_reward(result.get("reward", 0.0))
    sample.response = response_text
    sample.tokens = prompt_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.loss_mask = [1] * sample.response_length

    exit_status = str(result.get("exit_status") or "").strip().lower()
    if exit_status and exit_status != "ok":
        sample.status = Sample.Status.FAILED
    elif (
        sampling_params.get("max_new_tokens") is not None
        and sample.response_length >= int(sampling_params["max_new_tokens"])
        and sample.response_length > 0
    ):
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.COMPLETED

    return sample
