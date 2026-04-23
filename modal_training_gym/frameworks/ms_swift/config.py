"""Configuration classes for ms-swift Megatron training on Modal.

`MsSwiftFrameworkConfig` holds all the `megatron sft` CLI flags. `MsSwiftConfig`
adds app-level behavior (`build_app`) on top of it.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common import GPUType

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration, ModelTrainingConfig
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
CHECKPOINTS_PATH = Path("/checkpoints")

# ── Types ─────────────────────────────────────────────────────────────────────

# Fields that are NOT emitted as megatron CLI flags.
_SKIP_FIELDS = {
    "gpu",
    "image",
    "transformers_version",
    "environment",
    "n_nodes",
    "gpus_per_node",
    "app_tags",
}


# Fields from ModelTrainingConfig that map to MsSwiftFrameworkConfig fields.
_MODEL_TRAINING_FIELDS = {
    "gpu_type": "gpu",
    "n_nodes": "n_nodes",
    "tensor_model_parallel_size": "tensor_model_parallel_size",
    "pipeline_model_parallel_size": "pipeline_model_parallel_size",
    "context_parallel_size": "context_parallel_size",
    "sequence_parallel": "sequence_parallel",
    "expert_model_parallel_size": "expert_model_parallel_size",
    "moe_permute_fusion": "moe_permute_fusion",
    "moe_grouped_gemm": "moe_grouped_gemm",
    "moe_shared_expert_overlap": "moe_shared_expert_overlap",
    "moe_aux_loss_coeff": "moe_aux_loss_coeff",
    "lora_rank": "lora_rank",
    "lora_alpha": "lora_alpha",
    "target_modules": "target_modules",
    "merge_lora": "merge_lora",
}


def model_training_overrides(
    model: "ModelConfiguration",
) -> dict[str, Any]:
    """Extract model-level training defaults as ms-swift framework config fields.

    Returns a dict keyed by ``MsSwiftFrameworkConfig`` field names. Values
    come from the model's ``ModelTrainingConfig``. Returns an empty dict
    when the model has no training config.
    """
    if model.training is None:
        return {}
    overrides = model.training.to_framework_overrides()
    return {
        ms_swift_key: overrides[tc_key]
        for tc_key, ms_swift_key in _MODEL_TRAINING_FIELDS.items()
        if tc_key in overrides
    }


_SENTINEL = object()


def _get_dataclass_default(cls: type, field_name: str) -> Any:
    """Return the dataclass default for a field, or _SENTINEL if none."""
    import dataclasses

    for f in dataclasses.fields(cls):
        if f.name == field_name:
            if f.default is not dataclasses.MISSING:
                return f.default
            if f.default_factory is not dataclasses.MISSING:
                return f.default_factory()
            return _SENTINEL
    return _SENTINEL


@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class MsSwiftFrameworkConfig:
    """ms-swift Megatron SFT configuration, including Modal infrastructure.

    All non-skip attributes are forwarded to ``megatron sft`` as CLI flags
    (``--flag value``). ms-swift uses underscore-style names and explicit
    ``"true"``/``"false"`` string-valued booleans.

    ## Modal Infrastructure

    gpu : GPUType
        Modal GPU type. Default ``"H100"``.
    image : str
        Docker image for the training container.
    transformers_version : str
        Transformers version to reinstall for compatibility. Default ``"4.57.3"``.
    app_tags : dict
        Extra Modal app tags. Default ``{}``.
    environment : dict[str, str]
        Environment variables for the training container.
    n_nodes : int
        Number of Modal cluster nodes. Default ``4``.
    gpus_per_node : int
        GPUs per node. Default ``8``.

    ## Tuner and Data

    tuner_type : str
        Fine-tuning method (e.g. ``"lora"``). Default ``"lora"``.
    split_dataset_ratio : float
        Train/eval split ratio. Default ``0.01``.
    perform_initialization : bool
        Emit ``--perform_initialization`` flag. Default ``True``.
    use_distributed_optimizer : bool
        Enable distributed optimizer. Default ``True``.

    ## Batch

    global_batch_size : int
        Global batch size across all ranks. Default ``8``.
    packing : bool
        Enable sequence packing. Default ``True``.
    padding_free : bool
        Enable padding-free attention. Default ``True``.
    use_precision_aware_optimizer : bool
        Enable precision-aware optimizer. Default ``True``.

    ## Training

    train_iters : int | None
        Max training iterations. Overrides ``num_train_epochs``. Default ``None``.
    num_train_epochs : int
        Number of training epochs. Default ``4``.
    lr : float
        Learning rate. Default ``1e-4``.
    lr_warmup_fraction : float
        Fraction of training for LR warmup. Default ``0.05``.
    lr_decay_style : str
        LR decay schedule. Default ``"cosine"``.
    min_lr : float
        Minimum learning rate. Default ``1e-5``.
    weight_decay : float
        Weight decay coefficient. Default ``0.1``.
    clip_grad : float
        Gradient clipping norm. Default ``1.0``.
    adam_beta1 : float
        Adam beta1. Default ``0.9``.
    adam_beta2 : float
        Adam beta2. Default ``0.95``.
    seed : int
        Random seed. Default ``42``.

    ## Precision and Memory

    bf16 : bool
        Enable BF16 training. Default ``True``.
    recompute_granularity : str
        Activation recomputation granularity. Default ``"selective"``.
    recompute_modules : str
        Modules to recompute. Default ``"core_attn"``.
    attention_softmax_in_fp32 : bool
        Compute attention softmax in FP32. Default ``True``.

    ## Context and Attention

    max_length : int
        Maximum sequence length. Default ``2048``.
    attention_backend : str
        Attention implementation. Default ``"flash"``.

    ## IO and Checkpointing

    dataset_num_proc : int
        Dataset preprocessing workers. Default ``8``.
    dataloader_num_workers : int
        DataLoader workers. Default ``4``.
    save_interval : int
        Checkpoint save interval (iterations). Default ``50``.
    no_save_optim : bool
        Skip saving optimizer state. Default ``True``.
    no_save_rng : bool
        Skip saving RNG state. Default ``True``.
    save_safetensors : bool
        Save in safetensors format. Default ``True``.
    use_hf : int
        Use HuggingFace checkpoint format. Default ``1``.
    add_version : bool
        Add version sub-directory to save path. Default ``False``.

    ## Logging and Eval

    log_interval : int
        Logging interval (iterations). Default ``1``.
    eval_iters : int
        Evaluation iterations. Default ``10``.
    eval_interval : int
        Evaluation interval (iterations). Default ``50``.

    ## LoRA

    lora_dropout : float
        LoRA dropout. Default ``0.05``.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu: GPUType = "H100"
    image: str = (
        "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:"
        "ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.3"
    )
    # Baseten's shipped transformers may lack GLM-4 MoE support; reinstall this
    # version to ensure compatibility.
    transformers_version: str = "4.57.3"

    # ── Modal app tags ───────────────────────────────────────────────────────
    # Merged with the framework's default tags (training/source/framework) at
    # app-build time. Use for per-run tagging.
    app_tags: dict = field(default_factory=dict)

    # ── Launcher-only (NOT megatron CLI flags) ───────────────────────────────
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
    )
    n_nodes: int = 4
    gpus_per_node: int = 8

    # ── Tuner & data split ───────────────────────────────────────────────────
    tuner_type: str = "lora"
    split_dataset_ratio: float = 0.01
    # Bare flag — emitted as `--perform_initialization` (no value) when True.
    perform_initialization: bool = True
    use_distributed_optimizer: bool = True

    # ── Batch ────────────────────────────────────────────────────────────────
    global_batch_size: int = 8
    packing: bool = True
    padding_free: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Training ─────────────────────────────────────────────────────────────
    train_iters: int | None = None
    num_train_epochs: int = 4
    lr: float = 1e-4
    lr_warmup_fraction: float = 0.05
    lr_decay_style: str = "cosine"
    min_lr: float = 1e-5
    weight_decay: float = 0.1
    clip_grad: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    seed: int = 42

    # ── Precision / memory ───────────────────────────────────────────────────
    bf16: bool = True
    recompute_granularity: str = "selective"
    recompute_modules: str = "core_attn"
    attention_softmax_in_fp32: bool = True

    # ── Context / attention ──────────────────────────────────────────────────
    max_length: int = 2048
    attention_backend: str = "flash"

    # ── IO / checkpointing ───────────────────────────────────────────────────
    dataset_num_proc: int = 8
    dataloader_num_workers: int = 4
    save_interval: int = 50
    no_save_optim: bool = True
    no_save_rng: bool = True
    save_safetensors: bool = True
    use_hf: int = 1
    # Disable versioned sub-dir in save path. Required so cross-node rank 0
    # and last-rank see the same path (v0-timestamp suffix would race).
    add_version: bool = False

    # ── Logging / eval ───────────────────────────────────────────────────────
    log_interval: int = 1
    eval_iters: int = 10
    eval_interval: int = 50

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_dropout: float = 0.05

    @classmethod
    def from_toml(cls, path: str | Path) -> "MsSwiftFrameworkConfig":
        import msgspec

        payload = msgspec.toml.decode(Path(path).read_bytes())
        if not isinstance(payload, dict):
            raise ValueError(f"Expected TOML table at root in {path!r}")
        return cls(**payload)


class MsSwiftConfig:
    """Top-level wrapper that composes an ms-swift Megatron SFT run.

    Bundles a model, dataset, W&B logging config, and ms-swift-specific
    framework settings. Call ``build_app()`` to get a Modal ``App``
    ready to run.

    ## Fields

    dataset : DatasetConfig | None
        Dataset configuration with ``prepare()`` hook. Default ``None``.
    model : ModelConfiguration | None
        Model identity and download hook. Default ``None``.
    wandb : WandbConfig | None
        Weights & Biases logging config. Default ``None``.
    framework_config : MsSwiftFrameworkConfig
        ms-swift Megatron training and infrastructure settings.
        Default ``MsSwiftFrameworkConfig()``.

    Methods
    -------
    cli_args(output_dir)
        Generate the full ``megatron sft`` CLI argument list.
    build_app(name=None)
        Build and return a Modal ``App`` for this training run.
    """

    dataset: "DatasetConfig | None"
    model: "ModelConfiguration | None"
    wandb: "WandbConfig | None"
    framework_config: MsSwiftFrameworkConfig

    def __init__(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfiguration | None" = None,
        wandb: "WandbConfig | None" = None,
        framework_config: MsSwiftFrameworkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.wandb = wandb
        self.framework_config = framework_config or MsSwiftFrameworkConfig()

    def _fields(self) -> dict[str, Any]:
        fields = dict(vars(self.framework_config))
        if self.model is not None:
            overrides = model_training_overrides(self.model)
            for k, v in overrides.items():
                if k in fields:
                    dc_default = _get_dataclass_default(MsSwiftFrameworkConfig, k)
                    if dc_default is not _SENTINEL and fields[k] == dc_default:
                        fields[k] = v
                else:
                    fields[k] = v
        fields = {k: v for k, v in fields.items() if k not in _SKIP_FIELDS}
        if self.dataset is not None and hasattr(self.dataset, "prompt_data"):
            fields["dataset"] = self.dataset.prompt_data
        if self.model is not None:
            fields["model"] = self.model.model_name
        if self.wandb is not None:
            fields["report_to"] = "wandb"
            if self.wandb.project:
                fields["wandb_project"] = self.wandb.project
            exp = self.wandb.exp_name or self.wandb.group
            if exp:
                fields["wandb_exp_name"] = exp
        if fields.get("eval_iters", 0) <= 0:
            fields.pop("eval_interval", None)
        return fields

    def cli_args(self, *, output_dir: str) -> list[str]:
        out: list[str] = []
        fields = dict(self._fields())
        if fields.pop("perform_initialization", False):
            out.append("--perform_initialization")
        fields["output_dir"] = output_dir
        for key, val in fields.items():
            if val is None:
                continue
            flag = f"--{key}"
            if isinstance(val, bool):
                out += [flag, "true" if val else "false"]
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    def build_app(
        self,
        *,
        name: str | None = None,
    ) -> "App":
        from .launcher import build_ms_swift_app

        return build_ms_swift_app(swift=self, name=name)
