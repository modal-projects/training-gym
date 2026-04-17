"""Llama-2-7B model spec — registered on import.

The torchrun / accelerate launchers pass `hf_checkpoint` to a user-supplied
training script (via `transformers.AutoModelForCausalLM.from_pretrained`), so
the script reads architecture from the HF config at runtime. The architecture
fields here are a stub.
"""

from .base import BaseModelType, ModelArchitecture, register_model

LLAMA2_7B_ARCHITECTURE = ModelArchitecture()

register_model(
    BaseModelType.Llama2_7B,
    hf_checkpoint="meta-llama/Llama-2-7b-hf",
    architecture=LLAMA2_7B_ARCHITECTURE,
)
