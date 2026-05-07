"""Qwen3-30B-A3B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture


class Qwen3_30B(HFModelConfiguration):
    """Qwen3-30B-A3B (30B total, ~3B active) MoE model from Alibaba.

    Mixture-of-Experts with 128 experts, 8 active per token.
    Pre-configured with base ``ModelArchitecture`` for Megatron-based
    frameworks (slime). Downloads from ``Qwen/Qwen3-30B-A3B`` on HuggingFace.
    """

    model_name = "Qwen/Qwen3-30B-A3B"
    architecture = ModelArchitecture(
        num_layers=48,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=32,
        group_query_attention=True,
        num_query_groups=4,
        kv_channels=128,
        vocab_size=151936,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        use_rotary_position_embeddings=True,
        rotary_base=1000000,
    )
