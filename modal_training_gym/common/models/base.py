"""Model configuration + `download()` hook, shared across training frameworks.

A `ModelConfig` bundles identity (`model_name`), an optional local
`model_path`, an optional transformer `architecture`, and an optional
`training` profile. Pure data — each framework config reads what it
needs from an attached subclass.

Subclass and set `model_name` / `model_path` / `architecture` / `training`
as class attributes (or pass them as constructor kwargs), then override
`download()` to materialize weights into the shared model volume.

The built-in models (`Qwen3_0_6B`, `Qwen3_4B`, `Qwen3_32B`, `GLM_4_7`,
`Llama2_7B`, `Kimi_K2_5`) are concrete subclasses in their own per-model
modules. For a custom HuggingFace model, write your own subclass — no
registry, no global state.

Example (mirrors the `DatasetConfig` pattern in
`modal_training_gym/common/dataset.py`):

    from huggingface_hub import snapshot_download
    from modal_training_gym.common.models import (
        ModelArchitecture, ModelConfig, ModelTrainingConfig,
    )

    class SmolLM2_135M(ModelConfig):
        model_name = "HuggingFaceTB/SmolLM2-135M"
        training = ModelTrainingConfig(
            gpu_type="H100",
            tensor_model_parallel_size=1,
        )

        def download(self) -> None:
            snapshot_download(repo_id=self.model_name)
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields
from typing import TYPE_CHECKING, Any

from modal_training_gym.common.deploy import DeployConfig, ModelDeployment

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
            args += ["--vocab-size", str(self.vocab_size)]
            args += ["--make-vocab-size-divisible-by", "1"]
        if self.normalization:
            args += ["--normalization", self.normalization]
        # Megatron defaults do not consistently match HF config defaults,
        # so emit the model's declared norm epsilon explicitly.
        if self.norm_epsilon:
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
    gpus_per_node : int
        Recommended GPU count per node for this model. Default ``8``.

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
    gpus_per_node: int = 8

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



class ModelConfig:
    """Base class for model identity and weight-download logic.

    Subclass and set ``model_name`` (and optionally ``model_path`` and
    ``architecture``) as class attributes, then override
    ``download()`` to materialize weights into the shared model
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
    deploy : DeployConfig | None
        vLLM serving configuration (GPU, extra args, deploy strategy).
        When ``None``, ``serve()`` infers settings from the model's
        training presets. Default ``None``.

    Methods
    -------
    download()
        Download or materialize weights into the model volume. Must be
        overridden by subclasses.
    serve()
        Build + deploy a vLLM app and return the endpoint URL.
    """

    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None
    training: ModelTrainingConfig | None = None
    deploy: DeployConfig | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def download(self) -> None:
        """Download or materialize weights into the model volume."""
        raise NotImplementedError(f"{type(self).__name__} has no download()")


    def serve(
        self,
        *,
        app_name: str | None = None,
        served_model_name: str | None = None,
        checkpoint_path: str | None = None,
        checkpoints_volume: str | None = None,
        checkpoints_mount_path: str | None = None,
    ) -> "ModelDeployment":
        """Build + deploy a vLLM serving app and return endpoint metadata.

        Uses ``model_path`` when set (or ``checkpoint_path`` override),
        otherwise falls back to ``model_name`` (HF repo id).
        """
        import modal

        from modal_training_gym.common.serve_vllm import build_vllm_serve_app

        model_path = checkpoint_path or self.model_path or self.model_name
        if not model_path:
            raise ValueError(
                f"{type(self).__name__} has no model path to serve. "
                "Set model_path, model_name, or pass checkpoint_path."
            )

        default_name_src = self.model_name or model_path
        default_slug = (
            default_name_src.rstrip("/").split("/")[-1].replace("_", "-").lower()
        )
        resolved_app_name = app_name or f"{default_slug}-serve"
        resolved_served_model_name = served_model_name or default_slug
        resolved_checkpoints_volume = (
            checkpoints_volume or None
        )
        resolved_mount_path = (
            checkpoints_mount_path
        )
        d = self.deploy
        build_kwargs: dict[str, Any] = dict(
            app_name=resolved_app_name,
            model_path=model_path,
            served_model_name=resolved_served_model_name,
            checkpoints_volume=resolved_checkpoints_volume,
        )
        if resolved_mount_path is not None:
            build_kwargs["checkpoints_mount_path"] = resolved_mount_path
        if d and d.gpu:
            build_kwargs["gpu"] = d.gpu
        if d and d.n_gpu is not None:
            build_kwargs["n_gpu"] = d.n_gpu
        if d and d.extra_vllm_args:
            build_kwargs["extra_vllm_args"] = d.extra_vllm_args

        app = build_vllm_serve_app(**build_kwargs)
        env_name = d.environment_name if d else None
        strategy = d.deploy_strategy if d else "rolling"
        app.deploy(
            environment_name=env_name,
            strategy=strategy,
        )
        serve_fn = modal.Function.from_name(
            resolved_app_name,
            "serve",
            environment_name=env_name,
        )
        url = serve_fn.get_web_url()
        if not url:
            raise RuntimeError(
                f"Deployed {resolved_app_name!r} but no web URL was returned for function 'serve'."
            )
        return ModelDeployment(
            app=app,
            app_name=resolved_app_name,
            served_model_name=resolved_served_model_name,
            url=url,
            deploy=d or DeployConfig(
                environment_name=env_name,
                deploy_strategy=strategy,
            ),
        )


class HFModelConfiguration(ModelConfig):
    """ModelConfig for models hosted on HuggingFace.

    Implements ``download()`` via ``huggingface_hub.snapshot_download``
    using ``self.model_name`` as the repo ID. Subclass this for any
    HF-hosted model — only set ``model_name`` (plus optionally
    ``architecture`` and ``model_path``); the download step needs no
    override.

    All built-in model configs (``Qwen3_0_6B``, ``Qwen3_4B``, ``Qwen3_32B``,
    ``GLM_4_7``, ``Llama2_7B``, ``Kimi_K2_5``) extend this class.

    Methods
    -------
    download()
        Downloads the full model snapshot from HuggingFace Hub into the
        local cache.
    """

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        kwargs: dict = {"repo_id": self.model_name}
        if self.model_path:
            kwargs["local_dir"] = str(self.model_path)
        snapshot_download(**kwargs)
