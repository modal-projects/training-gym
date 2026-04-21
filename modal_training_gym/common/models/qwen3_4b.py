"""Qwen3-4B model spec as a concrete ModelConfiguration subclass."""

from .base import ModelArchitecture, ModelConfiguration


class Qwen3_4B(ModelConfiguration):
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

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.model_name)
