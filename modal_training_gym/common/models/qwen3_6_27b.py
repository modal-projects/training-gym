"""Qwen3.6-27B model configuration."""

from .base import HFModelConfiguration, parse_qwen3_6_response


class Qwen3_6_27B(HFModelConfiguration):
    """Qwen3.6-27B dense hybrid Gated DeltaNet/attention model.

    The Slime recipe validates text-only causal-language-model training. The
    checkpoint's vision encoder is not represented by ``ModelArchitecture``;
    multimodal training remains outside this preset's validated scope.
    """

    response_parser = staticmethod(parse_qwen3_6_response)

    model_name = "Qwen/Qwen3.6-27B"
    architecture_overrides = {
        "qk_layernorm": True,
        "megatron_spec": ["slime_plugins.models.qwen3_5", "get_qwen3_5_spec"],
        "megatron_model_type": "qwen3.5-27B",
        "apply_layernorm_1p": True,
        "use_gated_attention": True,
        "attention_output_gate": True,
        "rotary_base": 10000000,
    }
