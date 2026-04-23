"""Reward and rollout wrappers for Harbor-backed Miles runs.

Loaded on the remote Miles image via ``--custom-rm-path`` and
``--rollout-function-path``. Not importable locally — requires ``miles``
(installed on the remote image).
"""

from __future__ import annotations

import logging

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Extract the reward set by the Harbor agent function."""
    if isinstance(samples, list):
        return [float(s.metadata.get("reward", 0.0)) for s in samples]
    return float(samples.metadata.get("reward", 0.0))


def _aggregate_agent_metrics(samples: list[Sample]) -> dict:
    all_metrics = [
        s.metadata.get("agent_metrics", {})
        for s in samples
        if getattr(s, "metadata", None) and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return {}

    metrics = {}
    for key in ["total_time", "model_latency", "verifier_latency"]:
        values = [m.get(key, 0.0) for m in all_metrics if key in m]
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    rewards = [float(s.metadata.get("reward", 0.0)) for s in samples]
    if rewards:
        metrics["agent/reward_mean"] = sum(rewards) / len(rewards)
        metrics["agent/reward_max"] = max(rewards)

    return metrics


class RolloutFn(InferenceRolloutFn):
    """Rollout function that aggregates Harbor agent metrics per batch."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        flat_samples = []
        for group in output.samples:
            if isinstance(group, list):
                flat_samples.extend(group)
            else:
                flat_samples.append(group)

        metrics = _aggregate_agent_metrics(flat_samples)
        if metrics:
            output.metrics = output.metrics or {}
            output.metrics.update(metrics)
            logger.info("Harbor rollout metrics: %s", metrics)

        return output
