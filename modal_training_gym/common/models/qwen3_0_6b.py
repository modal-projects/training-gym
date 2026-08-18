"""Qwen3-0.6B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, parse_qwen3_response


class Qwen3_0_6B(HFModelConfiguration):
    """Qwen3-0.6B (0.6 billion parameters) from Alibaba.

    Architecture is resolved from ``config.json`` for Megatron-based frameworks.
    """

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-0.6B"
    architecture_overrides = {"qk_layernorm": True}
