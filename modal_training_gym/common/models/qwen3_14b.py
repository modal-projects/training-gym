"""Qwen3-14B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture


class Qwen3_14B(HFModelConfiguration):
    """Qwen3-14B (14 billion parameters) from Alibaba.

    Pre-configured with full ``ModelArchitecture`` for Megatron-based
    frameworks (slime). Downloads from ``Qwen/Qwen3-14B`` on HuggingFace.
    """

    model_name = "Qwen/Qwen3-14B"
    architecture = ModelArchitecture(
        num_layers=40,
        hidden_size=5120,
        ffn_hidden_size=17408,
        num_attention_heads=40,
        group_query_attention=True,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=151936,
        normalization="RMSNorm",
        norm_epsilon=1e-6,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        use_rotary_position_embeddings=True,
        rotary_base=1000000,
    )
