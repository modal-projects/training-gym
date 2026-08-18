"""Qwen3.6-35B-A3B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, parse_qwen3_6_response


class Qwen3_6_35B(HFModelConfiguration):
    """Qwen3.6-35B-A3B (35B total, ~3B active) MoE model from Alibaba.

    Mixture-of-Experts with 256 experts, 8 active per token.
    Architecture is resolved from ``config.json`` for Megatron-based frameworks.
    """

    response_parser = staticmethod(parse_qwen3_6_response)

    model_name = "Qwen/Qwen3.6-35B-A3B"
    architecture_overrides = {
        "qk_layernorm": True,
        "ffn_hidden_size": 512,
        "moe_grouped_gemm": True,
        "moe_shared_expert_gate": True,
        "moe_router_score_function": "softmax",
        "moe_token_drop_policy": "probs",
        "moe_router_dtype": "fp32",
        "moe_permute_fusion": True,
        "moe_aux_loss_coeff": 0,
        "megatron_spec": ["slime_plugins.models.qwen3_5", "get_qwen3_5_spec"],
        "megatron_model_type": "qwen3.5-35B-A3B",
        "apply_layernorm_1p": True,
        "use_gated_attention": True,
        "attention_output_gate": True,
        "rotary_base": 10000000,
    }
