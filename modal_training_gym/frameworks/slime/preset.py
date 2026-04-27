"""Framework-specific model presets for slime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlimePreset:
    """Model-specific defaults for slime training.

    Every field is required — models must explicitly declare their
    slime configuration. Users can override any field by passing it
    to ``SlimeConfig(...)`` directly.
    """

    gpu_type: str
    actor_num_nodes: int
    actor_num_gpus_per_node: int
    colocate: bool
    tensor_model_parallel_size: int
    sequence_parallel: bool
