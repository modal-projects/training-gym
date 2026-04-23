"""Qwen3-32B model spec as a concrete HFModelConfiguration subclass.

verl reads the architecture from the HF config (via its `converter_hf_to_mcore`
script) at checkpoint-conversion time, so `architecture` is left as the default
`None`. Set `architecture = ModelArchitecture(...)` on a further subclass if
you need to use this model with a framework that consumes
`--num-layers`/`--hidden-size` as CLI flags (e.g. SLIME).
"""

from .base import HFModelConfiguration, ModelTrainingConfig


class Qwen3_32B(HFModelConfiguration):
    """Qwen3-32B (32 billion parameters) from Alibaba.

    No ``architecture`` is set — frameworks that need it (e.g. SLIME
    with Megatron) should subclass and populate the field. Frameworks
    that read architecture from the HF config (e.g. ms-swift) work
    as-is.
    """

    model_name = "Qwen/Qwen3-32B"
    training = ModelTrainingConfig(gpu_type="H100", n_nodes=4)
