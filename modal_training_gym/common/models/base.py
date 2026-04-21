"""Model configuration + `download_model()` hook, shared across training frameworks.

A `ModelConfiguration` bundles identity (`model_name`), an optional local
`model_path`, and an optional transformer `architecture`. Pure data —
each framework config reads what it needs from an attached subclass.

Subclass and set `model_name` / `model_path` / `architecture` as class
attributes (or pass them as constructor kwargs), then override
`download_model()` to materialize weights into the shared model volume.

The four built-in models (`Qwen3_4B`, `Qwen3_32B`, `GLM_4_7`, `Llama2_7B`)
are concrete subclasses in their own per-model modules. For a custom
HuggingFace model, write your own subclass — no registry, no global
state.

Example (mirrors the `DatasetConfig` pattern in
`modal_training_gym/common/dataset.py`):

    from huggingface_hub import snapshot_download
    from modal_training_gym.common.models import (
        ModelArchitecture, ModelConfiguration,
    )

    class SmolLM2_135M(ModelConfiguration):
        model_name = "HuggingFaceTB/SmolLM2-135M"

        def download_model(self) -> None:
            snapshot_download(repo_id=self.model_name)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelArchitecture:
    """Transformer architecture parameters for a specific model."""

    num_layers: int = 0
    hidden_size: int = 0
    ffn_hidden_size: int = 0
    num_attention_heads: int = 0
    group_query_attention: bool = True
    num_query_groups: int = 0
    kv_channels: int = 0
    vocab_size: int = 0
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-6
    swiglu: bool = True
    disable_bias_linear: bool = True
    qk_layernorm: bool = True
    use_rotary_position_embeddings: bool = True
    rotary_base: int = 10000


class ModelConfiguration:
    """Known model families. Extend as new models are added.

    Subclass and set `model_name`, optionally `model_path` and
    `architecture`, and override `download_model()` to materialize weights
    into the model volume.
    """

    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def download_model(self) -> None:
        """Download or materialize weights into the model volume."""
        raise NotImplementedError(
            f"{type(self).__name__} has no download_model()"
        )
