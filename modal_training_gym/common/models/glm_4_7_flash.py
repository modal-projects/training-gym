"""GLM-4.7-Flash (30B-A3B MoE) model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture


class GLM_4_7_Flash(HFModelConfiguration):
    """GLM-4.7-Flash (30B total, ~3B active) MoE model from Zhipu AI.

    Mixture-of-Experts with 64 routed experts + 1 shared expert,
    4 active per token. Uses Multi-head Latent Attention (MLA).
    Pre-configured with base ``ModelArchitecture`` for Megatron-based
    frameworks (slime). Downloads from ``zai-org/GLM-4.7-Flash`` on HuggingFace.
    """

    model_name = "zai-org/GLM-4.7-Flash"
    architecture = ModelArchitecture(
        num_layers=47,
        hidden_size=2048,
        ffn_hidden_size=10240,
        num_attention_heads=20,
        group_query_attention=False,
        num_query_groups=0,
        kv_channels=0,
        vocab_size=154880,
        normalization="RMSNorm",
        norm_epsilon=1e-5,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        untie_embeddings_and_output_weights=True,
        num_experts=64,
        moe_router_topk=4,
        use_rotary_position_embeddings=True,
        rotary_base=1000000,
    )
