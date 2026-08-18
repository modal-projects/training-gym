"""Qwen3-1.7B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, parse_qwen3_response


class Qwen3_1_7B(HFModelConfiguration):
    """Qwen3-1.7B (1.7 billion parameters) from Alibaba.

    Architecture is resolved from ``config.json`` for Megatron-based frameworks.
    """

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-1.7B"
    architecture_overrides = {"qk_layernorm": True}
