"""Qwen3-4B model spec as a concrete HFModelConfiguration subclass."""

from .base import HFModelConfiguration, ModelArchitecture
from modal_training_gym.frameworks.harbor.preset import HarborPreset
from modal_training_gym.frameworks.slime.preset import SlimePreset


class Qwen3_4B(HFModelConfiguration):
    """Qwen3-4B (4 billion parameters) from Alibaba.

    Pre-configured with full ``ModelArchitecture`` for Megatron-based
    frameworks (slime, ms-swift Megatron mode). Downloads from
    ``Qwen/Qwen3-4B`` on HuggingFace.
    """

    model_name = "Qwen/Qwen3-4B"
    architecture = ModelArchitecture(
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
    slime = SlimePreset(
        gpu_type="H100",
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        colocate=True,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )
    harbor = HarborPreset(
        gpu_type="H100",
        n_nodes=2,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        colocate=True,
    )
