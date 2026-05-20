"""GLM-4.7 (355B-A32B MoE) model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture


class GLM_4_7(HFModelConfiguration):
    """GLM-4.7 (355B total, ~32B active) MoE model from Zhipu AI.

    Mixture-of-Experts with 160 routed experts + 1 shared expert,
    8 active per token. First 3 layers are dense.
    Pre-configured with base ``ModelArchitecture`` for Megatron-based
    frameworks (slime). Downloads from ``zai-org/GLM-4.7`` on HuggingFace.
    """

    model_name = "zai-org/GLM-4.7"
    architecture = ModelArchitecture(
        num_layers=92,
        hidden_size=5120,
        ffn_hidden_size=12288,
        num_attention_heads=96,
        group_query_attention=True,
        num_query_groups=8,
        kv_channels=128,
        vocab_size=151552,
        normalization="RMSNorm",
        norm_epsilon=1e-5,
        swiglu=True,
        disable_bias_linear=False,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        use_rotary_position_embeddings=True,
        rotary_base=1000000,
    )
