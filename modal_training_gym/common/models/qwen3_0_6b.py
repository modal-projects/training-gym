"""Qwen3-0.6B model spec as a concrete HFModelConfiguration subclass."""

from modal_training_gym.frameworks.slime.preset import SlimePreset

from .base import HFModelConfiguration, ModelArchitecture, ModelTrainingConfig


class Qwen3_0_6B(HFModelConfiguration):
    """Qwen3-0.6B (0.6 billion parameters) from Alibaba.

    Ships lightweight single-node presets so tutorials and smoke tests can
    run without overriding the launcher shape.
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
    training = ModelTrainingConfig(
        gpu_type="H100",
        n_nodes=1,
        gpus_per_node=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        lora_rank=8,
        lora_alpha=16,
    )
    slime = SlimePreset(
        gpu_type="H100",
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )
