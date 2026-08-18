"""Qwen3.5-4B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, parse_qwen3_6_response


class Qwen3_5_4B(HFModelConfiguration):
    """Qwen3.5-4B (4 billion parameters) from Alibaba.

    Architecture is resolved from ``config.json`` for Megatron-based frameworks.
    """

    response_parser = staticmethod(parse_qwen3_6_response)

    model_name = "Qwen/Qwen3.5-4B"
    architecture_overrides = {
        "qk_layernorm": True,
        "megatron_spec": ["slime_plugins.models.qwen3_5", "get_qwen3_5_spec"],
        "apply_layernorm_1p": True,
        "use_gated_attention": True,
        "attention_output_gate": True,
        "rotary_base": 10000000,
        "rotary_percent": 0.25,
    }
