"""GLM-4.7-Flash (30B-A3B MoE) model spec."""

from .base import HFModelConfiguration, ModelArchitecture


class GLM_4_7_Flash(HFModelConfiguration):
    """GLM-4.7-Flash (30B total, 3B active) MoE from Zhipu AI.

    64 routed experts with top-4 routing plus 1 shared expert.
    Uses Multi-head Latent Attention (MLA) and multi-token prediction.
    Downloads from ``zai-org/GLM-4.7-Flash`` on HuggingFace.
    """

    model_name = "zai-org/GLM-4.7-Flash"
    architecture = ModelArchitecture(
        num_layers=47,
        hidden_size=2048,
        ffn_hidden_size=10240,
        num_attention_heads=20,
        group_query_attention=False,
        num_query_groups=20,
        kv_channels=128,
        vocab_size=154880,
        normalization="RMSNorm",
        norm_epsilon=1e-5,
        swiglu=True,
        disable_bias_linear=True,
        qk_layernorm=True,
        use_rotary_position_embeddings=True,
        rotary_base=1000000,
        num_experts=64,
        moe_router_topk=4,
        moe_ffn_hidden_size=1536,
        num_shared_experts=1,
        first_k_dense_replace=1,
    )
