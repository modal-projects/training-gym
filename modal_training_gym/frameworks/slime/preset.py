"""Framework-specific model presets for slime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlimePreset:
    """Model-specific cluster and parallelism defaults for slime training.

    The first six fields are required — models must explicitly declare
    their slime configuration. Rollout and critic fields have sensible
    defaults for the common colocated single-node case.
    """

    gpu_type: str
    actor_num_nodes: int
    actor_num_gpus_per_node: int
    colocate: bool
    tensor_model_parallel_size: int
    sequence_parallel: bool
    rollout_num_gpus: int | None = None
    rollout_num_gpus_per_engine: int = 1
    use_critic: bool = False
    critic_num_nodes: int | None = None
    critic_num_gpus_per_node: int | None = None
