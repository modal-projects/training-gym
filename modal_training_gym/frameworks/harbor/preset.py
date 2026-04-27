"""Framework-specific model presets for Harbor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarborPreset:
    """Model-specific defaults for Harbor RL training.

    Every field is required — models must explicitly declare their
    Harbor configuration. The Harbor launcher reads these values
    to set parallelism and cluster topology.
    """

    gpu_type: str
    n_nodes: int
    tensor_model_parallel_size: int
    sequence_parallel: bool
    colocate: bool
