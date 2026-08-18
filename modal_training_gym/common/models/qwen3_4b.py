"""Qwen3-4B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, parse_qwen3_response


class Qwen3_4B(HFModelConfiguration):
    """Qwen3-4B (4 billion parameters) from Alibaba.

    Architecture is resolved from ``config.json`` for Megatron-based frameworks.
    """

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-4B"
    architecture_overrides = {"qk_layernorm": True}
