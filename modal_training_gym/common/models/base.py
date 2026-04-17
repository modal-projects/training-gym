"""Model-identity + architecture configs shared across training frameworks.

A `Model` bundles:

  - `BaseModelType` — enum tag for a supported model family.
  - `hf_checkpoint`  — HuggingFace repo id (e.g. "Qwen/Qwen3-4B").
  - `ModelArchitecture` — transformer-architecture fields (layers, heads, …).

Framework configs (e.g. `SlimeConfig`) accept a `Model` as a field; the
model's fields are merged into the framework's flat field dict at
`cli_args()` time, so the user no longer has to hand-copy architecture
values into every experiment config.

Conversion:
  - `Model.to_fields()`        → flat dict of hf_checkpoint + architecture
  - `Model.from_fields(flat)`  → `Model` from a framework's flat field dict
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from enum import Enum
from typing import Any


class BaseModelType(str, Enum):
    """Known model families. Extend as new models are added."""

    Qwen3_4B = "Qwen3_4B"


# Architecture fields a Model contributes to a framework's flat field dict.
# Extending this exposes a new architecture knob to frameworks.
ARCHITECTURE_FIELDS: tuple[str, ...] = (
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "group_query_attention",
    "num_query_groups",
    "kv_channels",
    "vocab_size",
    "normalization",
    "norm_epsilon",
    "swiglu",
    "disable_bias_linear",
    "qk_layernorm",
    "use_rotary_position_embeddings",
    "rotary_base",
)


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

    def to_fields(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


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

    def to_fields(self) -> dict[str, Any]:
        """Flat {field_name: value} dict this model contributes to a framework."""
        assert self.architecture is not None  # guaranteed by __post_init__
        return {
            "hf_checkpoint": self.hf_checkpoint,
            **self.architecture.to_fields(),
        }

    @classmethod
    def from_fields(
        cls,
        fields: dict[str, Any],
        *,
        model_type: BaseModelType | None = None,
    ) -> "Model":
        """Build a `Model` from a flat field dict.

        Reverse of attaching a `Model` — pull model-related fields out of an
        existing framework config that has them set directly. `model_type`
        defaults to the first enum member if not supplied.
        """
        arch = ModelArchitecture(
            **{k: fields[k] for k in ARCHITECTURE_FIELDS if k in fields}
        )
        return cls(
            model_type=model_type or next(iter(BaseModelType)),
            hf_checkpoint=fields.get("hf_checkpoint", ""),
            architecture=arch,
        )
