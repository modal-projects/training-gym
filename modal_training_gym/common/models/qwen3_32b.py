"""Qwen3-32B model spec as a concrete HFModelConfiguration subclass.

Set `architecture = ModelArchitecture(...)` on a further subclass if
you need to use this model with a framework that consumes
`--num-layers`/`--hidden-size` as CLI flags (e.g. slime).
"""

from .base import HFModelConfiguration, ModelTrainingConfig


class Qwen3_32B(HFModelConfiguration):
    """Qwen3-32B (32 billion parameters) from Alibaba.

    No ``architecture`` is set — frameworks that need it (e.g. slime
    with Megatron) should subclass and populate the field.
    """

    model_name = "Qwen/Qwen3-32B"
    training = ModelTrainingConfig(gpu_type="H100", n_nodes=4)
