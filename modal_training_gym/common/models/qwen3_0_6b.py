"""Qwen3-0.6B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture


class Qwen3_0_6B(HFModelConfiguration):
    """Qwen3-0.6B (0.6 billion parameters) from Alibaba.

    Pre-configured with full ``ModelArchitecture`` for Megatron-based
    frameworks (slime). Downloads from ``Qwen/Qwen3-0.6B`` on HuggingFace.
    """

    model_name = "Qwen/Qwen3-0.6B"
    architecture = ModelArchitecture(
        num_layers=28,
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=16,
        group_query_attention=True,
        num_query_groups=8,
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
