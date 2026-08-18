"""Qwen3-30B-A3B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, parse_qwen3_response


class Qwen3_30B(HFModelConfiguration):
    """Qwen3-30B-A3B (30B total, ~3B active) MoE model from Alibaba.

    Mixture-of-Experts with 128 experts, 8 active per token.
    Architecture is resolved from ``config.json`` for Megatron-based frameworks.
    """

    response_parser = staticmethod(parse_qwen3_response)

    model_name = "Qwen/Qwen3-30B-A3B"
    architecture_overrides = {
        "qk_layernorm": True,
        "num_experts": 0,
        "moe_ffn_hidden_size": 0,
        "moe_router_topk": 0,
        "untie_embeddings_and_output_weights": True,
    }
