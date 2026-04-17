"""Model identity + architecture, shared across training frameworks.

A `Model` bundles:

  - `BaseModelType` — enum tag naming the supported model family.
  - `hf_checkpoint`  — HuggingFace repo id (e.g. `"Qwen/Qwen3-4B"`).
  - `ModelArchitecture` — transformer-architecture fields (layers, heads, …).

Pure data — each framework config writes its own converter that turns a
`Model` into its specific CLI flags. The registry (`register_model`) lets
`Model(BaseModelType.Qwen3_4B)` auto-fill `hf_checkpoint` + `architecture`
from per-model files; pass either field explicitly to override.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BaseModelType(str, Enum):
    """Known model families. Extend as new models are added."""

    Qwen3_4B = "Qwen3_4B"
    Qwen3_32B = "Qwen3_32B"
    GLM_4_7 = "GLM_4_7"
    Llama2_7B = "Llama2_7B"


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


# Registry populated at import time by per-model modules (qwen3_4b.py etc.).
# Maps `BaseModelType` → (hf_checkpoint, architecture). Look up via the public
# `Model(model_type)` constructor, which infers missing fields from here.
_MODEL_REGISTRY: dict[BaseModelType, tuple[str, ModelArchitecture]] = {}


def register_model(
    model_type: BaseModelType,
    *,
    hf_checkpoint: str,
    architecture: ModelArchitecture,
) -> None:
    """Register the canonical hf_checkpoint + architecture for a model type."""
    _MODEL_REGISTRY[model_type] = (hf_checkpoint, architecture)


@dataclass
class Model:
    """A training model = type tag + weights location + architecture.

    Only `model_type` is required — `hf_checkpoint` and `architecture` are
    looked up from the registry populated by per-model modules. Pass either
    field explicitly to override the default for a registered type.
    """

    model_type: BaseModelType
    hf_checkpoint: str | None = None
    architecture: ModelArchitecture | None = None

    def __post_init__(self) -> None:
        spec = _MODEL_REGISTRY.get(self.model_type)
        if spec is None:
            if self.hf_checkpoint is None or self.architecture is None:
                raise ValueError(
                    f"No registered spec for {self.model_type!r}; either call "
                    f"`register_model(...)` in a model module, or pass "
                    f"hf_checkpoint and architecture explicitly."
                )
            return
        default_hf, default_arch = spec
        if self.hf_checkpoint is None:
            self.hf_checkpoint = default_hf
        if self.architecture is None:
            self.architecture = default_arch
