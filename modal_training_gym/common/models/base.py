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
    """Transformer architecture parameters for a specific model.

    These fields map directly to Megatron-LM model-parallel configuration
    flags. Framework launchers read them to generate the correct CLI
    arguments for distributed training.

    ## Model Dimensions

    num_layers : int
        Number of transformer layers. Default ``0``.
    hidden_size : int
        Hidden dimension size. Default ``0``.
    ffn_hidden_size : int
        Feed-forward network intermediate size. Default ``0``.
    vocab_size : int
        Vocabulary size. Default ``0``.

    ## Attention

    num_attention_heads : int
        Number of attention heads. Default ``0``.
    group_query_attention : bool
        Enable grouped-query attention (GQA). Default ``True``.
    num_query_groups : int
        Number of KV head groups for GQA. Default ``0``.
    kv_channels : int
        Per-head key/value channel dimension. Default ``0``.

    ## Normalization and Activation

    normalization : str
        Layer normalization type. Default ``"RMSNorm"``.
    norm_epsilon : float
        Normalization epsilon. Default ``1e-6``.
    swiglu : bool
        Use SwiGLU activation in FFN. Default ``True``.
    disable_bias_linear : bool
        Disable bias in linear layers. Default ``True``.
    qk_layernorm : bool
        Apply layer norm to query and key projections. Default ``True``.

    ## Position Encoding

    use_rotary_position_embeddings : bool
        Use RoPE positional encoding. Default ``True``.
    rotary_base : int
        Base frequency for RoPE. Default ``10000``.
    """

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

    def to_megatron_args(self) -> list[str]:
        """Generate Megatron-LM CLI flags from this architecture spec."""
        args: list[str] = []
        if self.num_layers:
            args += ["--num-layers", str(self.num_layers)]
        if self.hidden_size:
            args += ["--hidden-size", str(self.hidden_size)]
        if self.ffn_hidden_size:
            args += ["--ffn-hidden-size", str(self.ffn_hidden_size)]
        if self.num_attention_heads:
            args += ["--num-attention-heads", str(self.num_attention_heads)]
        if self.group_query_attention:
            args.append("--group-query-attention")
        if self.num_query_groups:
            args += ["--num-query-groups", str(self.num_query_groups)]
        if self.vocab_size:
            args += ["--make-vocab-size-divisible-by", "1"]
        if self.normalization:
            args += ["--normalization", self.normalization]
        if self.swiglu:
            args.append("--swiglu")
        if self.disable_bias_linear:
            args.append("--disable-bias-linear")
        if self.use_rotary_position_embeddings:
            args += ["--position-embedding-type", "rope"]
            if self.rotary_base != 10000:
                args += ["--rotary-base", str(self.rotary_base)]
        return args


class ModelConfiguration:
    """Base class for model identity and weight-download logic.

    Subclass and set ``model_name`` (and optionally ``model_path`` and
    ``architecture``) as class attributes, then override
    ``download_model()`` to materialize weights into the shared model
    volume.

    Parameters (constructor kwargs or class attributes)
    ---------------------------------------------------
    model_name : str
        HuggingFace repo ID or other model identifier. Default ``""``.
    model_path : str | None
        Override local path for model weights. When ``None``, frameworks
        derive the path from ``model_name``. Default ``None``.
    architecture : ModelArchitecture | None
        Transformer architecture spec for Megatron-based frameworks.
        Not required for HuggingFace-only workflows. Default ``None``.

    Methods
    -------
    download_model()
        Download or materialize weights into the model volume. Must be
        overridden by subclasses.
    """

    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def download_model(self) -> None:
        """Download or materialize weights into the model volume."""
        raise NotImplementedError(f"{type(self).__name__} has no download_model()")


class HFModelConfiguration(ModelConfiguration):
    """ModelConfiguration for models hosted on HuggingFace.

    Implements ``download_model()`` via ``huggingface_hub.snapshot_download``
    using ``self.model_name`` as the repo ID. Subclass this for any
    HF-hosted model — only set ``model_name`` (plus optionally
    ``architecture`` and ``model_path``); the download step needs no
    override.

    All built-in model configs (``Qwen3_4B``, ``Qwen3_32B``, ``GLM_4_7``,
    ``Llama2_7B``) extend this class.

    Methods
    -------
    download_model()
        Downloads the full model snapshot from HuggingFace Hub into the
        local cache.
    """

    def download_model(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=self.model_name)
