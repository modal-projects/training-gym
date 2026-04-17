"""Qwen3-4B model spec — registered on import."""

from .base import BaseModelType, ModelArchitecture, register_model

QWEN3_4B_ARCHITECTURE = ModelArchitecture(
    num_layers=36,
    hidden_size=2560,
    ffn_hidden_size=9728,
    num_attention_heads=32,
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

register_model(
    BaseModelType.Qwen3_4B,
    hf_checkpoint="Qwen/Qwen3-4B",
    architecture=QWEN3_4B_ARCHITECTURE,
)
