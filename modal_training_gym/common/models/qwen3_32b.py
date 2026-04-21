"""Qwen3-32B model spec as a concrete HFModelConfiguration subclass.

verl reads the architecture from the HF config (via its `converter_hf_to_mcore`
script) at checkpoint-conversion time, so `architecture` is left as the default
`None`. Set `architecture = ModelArchitecture(...)` on a further subclass if
you need to use this model with a framework that consumes
`--num-layers`/`--hidden-size` as CLI flags (e.g. SLIME).
"""

from .base import HFModelConfiguration


class Qwen3_32B(HFModelConfiguration):
    model_name = "Qwen/Qwen3-32B"
