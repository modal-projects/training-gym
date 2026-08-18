"""Qwen3-8B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, parse_qwen3_response


class Qwen3_8B(HFModelConfiguration):
    """Qwen3-8B (8 billion parameters) from Alibaba.

    Architecture is resolved from ``config.json`` for Megatron-based frameworks.
    """

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-8B"
    architecture_overrides = {
        "qk_layernorm": True,
        "untie_embeddings_and_output_weights": True,
    }
