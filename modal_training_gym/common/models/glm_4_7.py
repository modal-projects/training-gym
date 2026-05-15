"""GLM-4.7 (355B-A32B MoE) model spec."""

from .base import HFModelConfiguration, ModelArchitecture


class GLM_4_7(HFModelConfiguration):
    """GLM-4.7 (355B total, 32B active) MoE from Zhipu AI.

    160 routed experts with top-8 routing plus 1 shared expert.
    First 3 layers are dense; remaining 89 are MoE.
    Uses GQA (96 Q heads, 8 KV heads) with partial RoPE.
    Downloads from ``zai-org/GLM-4.7`` on HuggingFace.
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
        num_experts=160,
        moe_router_topk=8,
        moe_ffn_hidden_size=1536,
        num_shared_experts=1,
        first_k_dense_replace=3,
    )
