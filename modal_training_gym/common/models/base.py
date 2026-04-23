"""Model configuration + `download_model()` hook, shared across training frameworks.

A `ModelConfiguration` bundles identity (`model_name`), an optional local
`model_path`, an optional transformer `architecture`, and an optional
`training` profile. Pure data — each framework config reads what it
needs from an attached subclass.

Subclass and set `model_name` / `model_path` / `architecture` / `training`
as class attributes (or pass them as constructor kwargs), then override
`download_model()` to materialize weights into the shared model volume.

The built-in models (`Qwen3_4B`, `Qwen3_32B`, `GLM_4_7`, `Llama2_7B`,
`Kimi_K2_5`) are concrete subclasses in their own per-model modules. For
a custom HuggingFace model, write your own subclass — no registry, no
global state.

Example (mirrors the `DatasetConfig` pattern in
`modal_training_gym/common/dataset.py`):

    from huggingface_hub import snapshot_download
    from modal_training_gym.common.models import (
        ModelArchitecture, ModelConfiguration, ModelTrainingConfig,
    )

    class SmolLM2_135M(ModelConfiguration):
        model_name = "HuggingFaceTB/SmolLM2-135M"
        training = ModelTrainingConfig(
            gpu_type="H100",
            tensor_model_parallel_size=1,
        )

        def download_model(self) -> None:
            snapshot_download(repo_id=self.model_name)
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modal_training_gym.common import GPUType


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
        if self.kv_channels:
            args += ["--kv-channels", str(self.kv_channels)]
        if self.vocab_size:
            args += ["--make-vocab-size-divisible-by", "1"]
        if self.normalization:
            args += ["--normalization", self.normalization]
        if self.norm_epsilon != 1e-6:
            args += ["--norm-epsilon", str(self.norm_epsilon)]
        if self.swiglu:
            args.append("--swiglu")
        if self.disable_bias_linear:
            args.append("--disable-bias-linear")
        if self.qk_layernorm:
            args.append("--qk-layernorm")
        if self.use_rotary_position_embeddings:
            args += ["--position-embedding-type", "rope"]
            if self.rotary_base != 10000:
                args += ["--rotary-base", str(self.rotary_base)]
        return args


@dataclass
class ModelTrainingConfig:
    """Optimal training parameters for a model + GPU combination.

    Encapsulates parallelism, MoE, and LoRA settings that are tuned
    per model and GPU type. Frameworks call ``to_framework_overrides()``
    to get a dict they can merge into their own config fields.

    ## GPU

    gpu_type : GPUType
        Target GPU type these settings are tuned for. Default ``"H100"``.
    n_nodes : int
        Recommended cluster size for this model. Default ``1``.

    ## Parallelism

    tensor_model_parallel_size : int
        Tensor parallelism degree. Default ``1``.
    pipeline_model_parallel_size : int
        Pipeline parallelism degree. Default ``1``.
    context_parallel_size : int
        Context parallelism degree. Default ``1``.
    sequence_parallel : bool
        Enable sequence parallelism. Default ``False``.

    ## Expert Parallelism (MoE)

    expert_model_parallel_size : int
        Expert parallelism degree. Default ``1``.
    moe_permute_fusion : bool
        Enable MoE permute fusion. Default ``False``.
    moe_grouped_gemm : bool
        Enable grouped GEMM for MoE. Default ``False``.
    moe_shared_expert_overlap : bool
        Enable shared expert overlap. Default ``False``.
    moe_aux_loss_coeff : float
        Auxiliary loss coefficient for MoE load balancing. Default ``0.0``.

    ## LoRA

    lora_rank : int
        LoRA rank. Default ``8``.
    lora_alpha : int
        LoRA alpha scaling. Default ``16``.
    target_modules : str
        LoRA target modules. Default ``"all-linear"``.
    merge_lora : bool
        Merge LoRA weights after training. Default ``False``.
    """

    gpu_type: GPUType = "H100"
    n_nodes: int = 1

    # Parallelism
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False

    # Expert parallelism (MoE)
    expert_model_parallel_size: int = 1
    moe_permute_fusion: bool = False
    moe_grouped_gemm: bool = False
    moe_shared_expert_overlap: bool = False
    moe_aux_loss_coeff: float = 0.0

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    target_modules: str = "all-linear"
    merge_lora: bool = False

    def to_framework_overrides(self) -> dict[str, Any]:
        """Return training params as a dict for framework config consumption.

        Keys match the field names used by framework configs
        (``tensor_model_parallel_size``, ``lora_rank``, etc.). Frameworks
        merge this dict into their config, with explicit framework-level
        values taking precedence.
        """
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


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
    training : ModelTrainingConfig | None
        Optimal training parameters (parallelism, MoE, LoRA) for this
        model on a specific GPU type. Frameworks pull defaults from here
        so users don't need to manually specify model-tuned flags.
        Default ``None``.

    Methods
    -------
    download_model()
        Download or materialize weights into the model volume. Must be
        overridden by subclasses.
    """

    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None
    training: ModelTrainingConfig | None = None

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

        kwargs: dict = {"repo_id": self.model_name}
        if self.model_path:
            kwargs["local_dir"] = str(self.model_path)
        snapshot_download(**kwargs)
