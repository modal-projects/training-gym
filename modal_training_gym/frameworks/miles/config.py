"""Configuration for Miles RL training on Modal.

Miles is an RLVR training framework (built on Megatron + Ray) distributed as
a Docker image (`radixark/miles:…`) containing both the trainer and its
patched Megatron-LM stack. Users customize a run by providing "recipe args"
— a block of Miles CLI flags describing the model architecture + training
hyperparameters. The launcher adds the cluster + Ray glue.

Training defaults are exposed as typed fields on `MilesFrameworkConfig`.
They are emitted as CLI flags *before* `recipe_args`, so recipe_args can
override any default.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

from modal_training_gym.common import GPUType
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration, ModelTrainingConfig
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# Fields that are NOT emitted as Miles CLI flags.
_SKIP_FIELDS = {
    "gpu",
    "miles_image",
    "image_run_commands",
    "n_nodes",
    "recipe_args",
    "extra_args",
    "custom_config_yaml",
    "colocate",
    "actor_nodes",
    "rollout_num_gpus",
    "app_tags",
}


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
}


def model_training_overrides(
    model: "ModelConfiguration",
) -> dict[str, Any]:
    """Extract model-level training defaults as Miles framework config fields.

    Returns a dict keyed by ``MilesFrameworkConfig`` field names. Values
    come from the model's ``ModelTrainingConfig``. Returns an empty dict
    when the model has no training config.
    """
    if model.training is None:
        return {}
    overrides = model.training.to_framework_overrides()
    return {
        miles_key: overrides[tc_key]
        for tc_key, miles_key in _MODEL_TRAINING_FIELDS.items()
        if tc_key in overrides
    }


def model_training_cli_args(model: "ModelConfiguration") -> list[str]:
    """Convert model training config to hyphenated Miles CLI flags.

    Emits parallelism and MoE flags only (no LoRA, no infrastructure).
    Returns an empty list when the model has no training config.
    """
    overrides = model_training_overrides(model)
    overrides.pop("gpu", None)
    overrides.pop("n_nodes", None)
    out: list[str] = []
    for key, val in overrides.items():
        if val is None or val is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        if val is True:
            out.append(flag)
        else:
            out += [flag, str(val)]
    return out


@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class MilesFrameworkConfig:
    """Miles RLVR configuration, including Modal infrastructure.

    Typed fields provide training defaults and are emitted as Miles CLI
    flags (``--hyphen-name value``). They appear *before* ``recipe_args``,
    so ``recipe_args`` values take precedence.

    ## Modal Infrastructure

    miles_image : str
        Docker image with patched Megatron-LM + Miles trainer.
    image_run_commands : list[str]
        Extra commands appended to the image build. Default ``[]``.
    n_nodes : int
        Number of cluster nodes. Default ``1``.
    app_tags : dict
        Extra Modal app tags. Default ``{}``.

    ## Recipe and Overrides

    recipe_args : str
        Raw Miles CLI flag block. Model architecture flags and per-run
        overrides go here. Values override typed defaults. Default ``""``.
    extra_args : str
        Extra flags appended after ``recipe_args``. Default ``""``.
    custom_config_yaml : str
        Inline YAML overrides passed via ``--custom-config-path``. Default ``""``.

    ## Actor and Rollout Topology

    colocate : bool
        Reuse a single compute pool for actor + rollout. Default ``True``.
    actor_nodes : int | None
        Actor node count. Defaults to ``n_nodes`` (colocated) or
        ``n_nodes - 1`` (non-colocated). Default ``None``.
    rollout_num_gpus : int | None
        Rollout GPU count for non-colocated mode. Default ``None``.

    ## RL Algorithm

    advantage_estimator : str
        Advantage estimation method. Default ``"grpo"``.
    eps_clip : float
        PPO clipping epsilon. Default ``0.2``.
    clip_grad : float
        Gradient clipping norm. Default ``1.0``.
    kl_coef : float
        KL divergence penalty coefficient. Default ``0.0``.
    normalize_advantages : bool
        Normalize advantages. Default ``False``.
    seed : int
        Random seed. Default ``1234``.

    ## Optimizer

    lr : float
        Learning rate. Default ``1e-6``.
    lr_decay_style : str
        LR decay schedule. Default ``"constant"``.
    weight_decay : float
        Weight decay. Default ``0.0``.
    adam_beta1 : float
        Adam beta1. Default ``0.9``.
    adam_beta2 : float
        Adam beta2. Default ``0.95``.

    ## Batch

    micro_batch_size : int
        Micro batch size per GPU. Default ``1``.
    n_samples_per_prompt : int
        Rollout samples per prompt. Default ``8``.

    ## Precision and Memory

    bf16 : bool
        Enable BF16 training. Default ``True``.
    attention_softmax_in_fp32 : bool
        Compute attention softmax in FP32. Default ``True``.

    ## Regularization

    attention_dropout : float
        Attention dropout rate. Default ``0.0``.
    hidden_dropout : float
        Hidden layer dropout rate. Default ``0.0``.

    ## Rollout

    rollout_temperature : float
        Sampling temperature for rollouts. Default ``1.0``.

    ## Checkpointing

    no_save_optim : bool
        Skip saving optimizer state. Default ``True``.
    megatron_to_hf_mode : str
        HF checkpoint loading mode. Default ``"bridge"``.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    # Miles ships a pre-built image with patched Megatron-LM + the trainer.
    miles_image: str = "radixark/miles:dev-202603231227"
    # Extra commands appended to the image build (e.g. apply a local patch,
    # pip-install a pin). Each entry is passed to `Image.run_commands(...)`.
    image_run_commands: list[str] = field(default_factory=list)

    # ── Infrastructure ───────────────────────────────────────────────────────
    n_nodes: int = 1

    @property
    def gpus_per_node(self) -> int:
        # Our multi-node training set-up requires 8 GPUs per node.
        return 8

    # ── Recipe + overrides ───────────────────────────────────────────────────
    # Raw Miles CLI flag block — comments + blank lines OK, shlex-parsed.
    # Model architecture flags (--num-layers, --hidden-size, etc.) and any
    # per-run overrides go here. Values override the typed defaults above.
    recipe_args: str = ""
    # Extra flags appended after recipe_args, before the framework-enforced flags.
    extra_args: str = ""
    # Inline YAML overrides. When non-empty, written to /tmp/<run_id>.yaml
    # on the head rank and passed via `--custom-config-path`.
    custom_config_yaml: str = ""

    # ── Actor / rollout topology ─────────────────────────────────────────────
    # `colocate=True` reuses a single compute pool for actor + rollout.
    # When False, the launcher carves `rollout_num_gpus` out of the cluster.
    colocate: bool = True
    # When unset, actor_nodes defaults to `n_nodes` (colocated) or
    # `n_nodes - 1` (non-colocated).
    actor_nodes: int | None = None
    rollout_num_gpus: int | None = None

    # ── Modal app tags ───────────────────────────────────────────────────────
    app_tags: dict = field(default_factory=dict)

    # ── RL algorithm ─────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    clip_grad: float = 1.0
    kl_coef: float = 0.0
    normalize_advantages: bool = False
    seed: int = 1234

    # ── Optimizer ────────────────────────────────────────────────────────────
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    # ── Batch ────────────────────────────────────────────────────────────────
    micro_batch_size: int = 1
    n_samples_per_prompt: int = 8

    # ── Precision / memory ───────────────────────────────────────────────────
    bf16: bool = True
    attention_softmax_in_fp32: bool = True

    # ── Regularization ───────────────────────────────────────────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # ── Rollout ──────────────────────────────────────────────────────────────
    rollout_temperature: float = 1.0

    # ── Checkpointing ────────────────────────────────────────────────────────
    no_save_optim: bool = True
    megatron_to_hf_mode: str = "bridge"

    # ── Helpers ──────────────────────────────────────────────────────────────

    def cli_args(self) -> list[str]:
        """Convert typed training defaults to Miles CLI flags.

        Miles uses hyphen-separated flag names (``--clip-grad``) and bare
        boolean flags (``--bf16`` when True, omitted when False).
        """
        out: list[str] = []
        for key, val in vars(self).items():
            if key.startswith("_") or key in _SKIP_FIELDS or val is None:
                continue
            flag = f"--{key.replace('_', '-')}"
            if isinstance(val, bool):
                if val:
                    out.append(flag)
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    @staticmethod
    def _clean_arg_text(arg_text: str) -> str:
        """Strip `#` comments + blank lines from a recipe args block."""
        lines: list[str] = []
        for raw in arg_text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    def parsed_recipe_args(self) -> list[str]:
        """Shlex-parse `recipe_args` + `extra_args` into a flat argv list."""
        import shlex

        cleaned = "\n".join(
            part
            for part in (
                self._clean_arg_text(self.recipe_args),
                self._clean_arg_text(self.extra_args),
            )
            if part
        )
        return shlex.split(cleaned) if cleaned else []

    def resolved_actor_nodes(self) -> int:
        if self.actor_nodes is not None:
            return self.actor_nodes
        if self.colocate:
            return self.n_nodes
        if self.n_nodes < 2:
            raise ValueError(
                "Non-colocated rollout needs spare cluster capacity. "
                "Either set colocate=True or bump n_nodes >= 2."
            )
        return self.n_nodes - 1

    def resolved_rollout_num_gpus(self) -> int | None:
        if self.colocate:
            return None
        if self.rollout_num_gpus is not None and self.rollout_num_gpus > 0:
            return self.rollout_num_gpus
        spare = (self.n_nodes - self.resolved_actor_nodes()) * self.gpus_per_node
        if spare < 1:
            raise ValueError(
                f"Non-colocated rollout has no spare GPUs — n_nodes={self.n_nodes}, "
                f"actor_nodes={self.resolved_actor_nodes()}, "
                f"gpus_per_node={self.gpus_per_node}."
            )
        return spare


class MilesConfig:
    """Top-level wrapper that composes a Miles RLVR training run.

    Bundles a model, dataset, W&B logging config, and Miles-specific
    framework settings. Call ``build_app()`` to get a Modal ``App``
    ready to run.

    ## Fields

    dataset : DatasetConfig | None
        Dataset configuration with ``prepare()`` hook. Default ``None``.
    model : ModelConfiguration | None
        Model identity and download hook. Default ``None``.
    wandb : WandbConfig | None
        Weights & Biases logging config. Default ``None``.
    framework_config : MilesFrameworkConfig
        Miles-specific training and infrastructure settings.
        Default ``MilesFrameworkConfig()``.

    Methods
    -------
    build_app(name=None)
        Build and return a Modal ``App`` for this training run.
    """

    dataset: "DatasetConfig | None"
    model: "ModelConfiguration | None"
    wandb: "WandbConfig | None"
    framework_config: MilesFrameworkConfig

    def __init__(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfiguration | None" = None,
        wandb: "WandbConfig | None" = None,
        framework_config: MilesFrameworkConfig | None = None,
        name: str = "",
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.wandb = wandb
        self.framework_config = framework_config or MilesFrameworkConfig()
        self.name = name

    def build_app(
        self,
        *,
        name: str | None = None,
    ) -> "App":
        from .launcher import build_miles_app
        from modal_training_gym.common.framework import resolve_app_name, resolve_gpu

        app_name = resolve_app_name("miles", name or self.name, self.model)
        return build_miles_app(miles=self, gpu=resolve_gpu(self.model), name=app_name)
