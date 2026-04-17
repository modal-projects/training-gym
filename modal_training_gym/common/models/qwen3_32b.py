"""Qwen3-32B model spec — registered on import.

verl reads the architecture from the HF config (via its `converter_hf_to_mcore`
script) at checkpoint-conversion time, so the architecture fields here are a
stub. Populate with real values if you need to use this model with a framework
that consumes `--num-layers`/`--hidden-size` as CLI flags (e.g. SLIME).
"""

from .base import BaseModelType, ModelArchitecture, register_model

QWEN3_32B_ARCHITECTURE = ModelArchitecture()

register_model(
    BaseModelType.Qwen3_32B,
    hf_checkpoint="Qwen/Qwen3-32B",
    architecture=QWEN3_32B_ARCHITECTURE,
)
