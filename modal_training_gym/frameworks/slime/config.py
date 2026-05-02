"""Base configuration classes and volume mount paths for slime.

Two separate concerns:

  ModalConfig  — Modal infrastructure (gpu model, async mode, dev overlay)
  SlimeConfig  — slime training arguments

Each experiment defines one instance of each. All non-private, non-callable
attributes on a SlimeConfig subclass become slime CLI args automatically via
cli_args(). The 'environment' field is the only exception — it is injected
into the Ray job runtime env, not passed to slime directly.
"""

import math
from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.checkpoint import CheckpointConfig
from modal_training_gym.common.models import (
    ModelArchitecture,
    ModelConfig,
)
from modal_training_gym.common.wandb import WandbConfig
from modal_training_gym.frameworks.slime.preset import SlimePreset

if TYPE_CHECKING:
    from modal import App

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# ── Types ─────────────────────────────────────────────────────────────────────

# Fields on SlimeConfig that are NOT slime CLI args.
# `dataset`, `model`, and `wandb` are container objects; their own fields are
# expanded into cli args inside `_fields()`, but the containers themselves are
# never emitted as flags.
_SLIME_SKIP = {
    "environment",
    "async_mode",
    "dataset",
    "model",
    "wandb",
    "modal",
    "name",
    "app_tags",
    "image_run_commands",
    "local_python_sources",
    "preset",
    "checkpoint",
    "custom_rm_function",
}

# SlimeConfig fields that slime reads as YAML files at runtime.
# Users may set these as inline dicts in Python configs; the launcher
# materializes them to temp YAML files before building the CLI command.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


class ModalConfig:
    """Modal infrastructure configuration for slime — image setup and dev overlays.

    ## Fields

    local_slime : str | None
        Path to a local slime repo checkout for dev overlay. When set, the
        launcher mounts it into the image. Default ``None``.
    image_run_commands : list[str]
        Shell commands run during image build (e.g. ``git apply /tmp/my.patch``).
        Default ``[]``.
    local_python_sources : list[str]
        Sibling Python modules (by import name) to ship into the training
        image via ``add_local_python_source``. Use for helper modules like
        custom reward functions referenced via slime's ``custom_rm_path``.
        Default ``[]``.
    """

    local_slime: str | None = None
    image_run_commands: list[str]
    local_python_sources: list[str]

    def __init__(self, **kwargs: Any) -> None:
        self.image_run_commands = list(kwargs.pop("image_run_commands", []))
        self.local_python_sources = list(kwargs.pop("local_python_sources", []))
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class SlimeConfig:
    """slime GRPO training configuration.

    All non-skip fields are forwarded to slime as CLI args
    (``--hyphen-name value``).

    ## Launcher Instructions

    environment : dict
        Injected into the Ray job runtime env.
    async_mode : bool
        When ``True``, uses ``train_async.py``. Default ``False``.

    ## Composed Configs

    dataset : DatasetConfig | None
        Dataset configuration. Default ``None``.
    model : ModelConfig | None
        Model identity. Requires ``ModelArchitecture``. Default ``None``.
    wandb : WandbConfig | None
        W&B logging config. Default ``None``.
    modal : ModalConfig | None
        Modal infrastructure config. Default ``None``.
    app_tags : dict
        Extra Modal app tags. Default ``{}``.

    ## Cluster and Parallelism

    preset : SlimePreset | None
        Cluster and parallelism settings. Defaults to the model's
        ``SlimePreset`` if not set. Raises if neither is provided.
        See ``SlimePreset`` for available fields.

    ## RL Algorithm

    advantage_estimator : str
        Advantage estimation method. Default ``"grpo"``.
    n_samples_per_prompt : int
        Rollout samples per prompt. Default ``2``.
    eps_clip : float
        PPO clipping epsilon. Default ``0.2``.
    eps_clip_high : float
        Asymmetric high-side PPO clip. Default ``0.28``.
    use_kl_loss : bool
        Enable KL divergence loss. Default ``False``.
    kl_loss_type : str
        KL loss variant. Default ``"low_var_kl"``.
    kl_loss_coef : float
        KL loss coefficient. Default ``0.0``.
    entropy_coef : float
        Entropy bonus coefficient. Default ``0.0``.
    ref_load : str
        Reference model checkpoint path. Default ``""``.

    ## Rollout

    num_rollout : int
        Number of rollout episodes per step. Default ``1``.
    rollout_batch_size : int
        Batch size for rollout generation. Default ``8``.
    rollout_max_response_len : int
        Maximum response length during rollout. Default ``8192``.
    rollout_temperature : float
        Sampling temperature for rollouts. Default ``1.0``.
    sglang_mem_fraction_static : float
        SGLang static memory fraction. Default ``0.7``.

    ## Training

    global_batch_size : int
        Global batch size. Default ``16``.
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
    optimizer : str
        Optimizer name. Default ``"adam"``.

    ## Memory and Precision

    attention_dropout : float
        Attention dropout. Default ``0.0``.
    hidden_dropout : float
        Hidden layer dropout. Default ``0.0``.
    attention_softmax_in_fp32 : bool
        Compute softmax in FP32. Default ``True``.
    accumulate_allreduce_grads_in_fp32 : bool
        Accumulate allreduce grads in FP32. Default ``True``.
    recompute_granularity : str
        Activation recomputation granularity. Default ``""``.
    recompute_method : str
        Activation recomputation method. Default ``""``.
    recompute_num_layers : int | None
        Number of layers to recompute. Default ``None``.

    ## Dynamic Batching

    use_dynamic_batch_size : bool
        Use dynamic batch sizing. Default ``False``.
    max_tokens_per_gpu : int | None
        Max tokens per GPU for dynamic batching. Default ``None``.

    ## Eval

    eval_interval : int
        Evaluation interval in training steps. Default ``20``.
    n_samples_per_eval_prompt : int
        Eval samples per prompt. Default ``4``.
    eval_max_response_len : int
        Max response length for eval. Default ``16384``.
    eval_top_p : float
        Top-p sampling for eval. Default ``1.0``.
    eval_config : dict | None
        YAML eval configuration. Default ``None``.

    ## Checkpointing

    save : bool
        Enable checkpoint saving. Default ``True``.
    save_interval : int
        Checkpoint save interval. Default ``1000``.
    megatron_to_hf_mode : str
        Checkpoint conversion mode (``"bridge"`` or ``"raw"``). Default ``"bridge"``.
    use_fault_tolerance : bool
        Enable fault tolerance. Default ``False``.

    ## Custom Reward

    custom_rm_function : Callable | None
        A reward function to use. When set, the launcher automatically
        writes the function's source file into the training image and
        derives ``custom_rm_path`` from it. Also sets ``rm_type`` to
        ``"async_rm"`` unless already overridden. Default ``None``.
    custom_rm_path : str
        Python import path for custom reward function. Normally
        auto-derived from ``custom_rm_function``; set manually only
        when shipping the reward module yourself via
        ``local_python_sources``. Default ``""``.

    ## SGLang

    sglang_config : dict | None
        YAML SGLang server configuration. Default ``None``.
    custom_config_path : dict | None
        YAML custom config overrides. Default ``None``.
    apply_chat_template_kwargs : dict | None
        Extra kwargs for chat template. Default ``None``.
    """

    # ── Composed configs (required) ─────────────────────────────────────────
    dataset: DatasetConfig
    model: ModelConfig

    # ── App identity ─────────────────────────────────────────────────────────
    name: str = ""
    app_tags: dict = field(default_factory=dict)

    # ── Launcher instructions (not slime CLI flags) ─────────────────────────
    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    )
    async_mode: bool = False
    wandb: WandbConfig | None = None
    modal: ModalConfig | None = None
    image_run_commands: list[str] = field(default_factory=list)
    local_python_sources: list[str] = field(default_factory=list)

    # ── Cluster and parallelism ────────────────────────────────────────────
    preset: SlimePreset | None = None

    # ── RL algorithm ────────────────────────────────────────────────────────
    advantage_estimator: str = "grpo"
    n_samples_per_prompt: int = 2
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28
    use_kl_loss: bool = True
    kl_loss_type: str = "low_var_kl"
    kl_loss_coef: float = 0.0
    entropy_coef: float = 0.0
    ref_load: str = ""

    # ── Rollout ─────────────────────────────────────────────────────────────
    num_rollout: int = 1
    rollout_batch_size: int = 8
    rollout_max_response_len: int = 8192
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.7

    # ── Training ────────────────────────────────────────────────────────────
    global_batch_size: int = 16
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer: str = "adam"

    # ── Memory and precision ────────────────────────────────────────────────
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    accumulate_allreduce_grads_in_fp32: bool = True
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 1

    # ── Dynamic batching ────────────────────────────────────────────────────
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 9216

    # ── Eval ────────────────────────────────────────────────────────────────
    eval_interval: int = 20
    n_samples_per_eval_prompt: int = 4
    eval_max_response_len: int = 16384
    eval_top_p: float = 1.0
    eval_config: dict | None = None

    # ── Checkpointing ───────────────────────────────────────────────────────
    save: str = "/checkpoints"
    save_interval: int = 1000
    megatron_to_hf_mode: str = "bridge"
    use_fault_tolerance: bool = True

    # ── Reward model ─────────────────────────────────────────────────────────
    rm_type: str | None = None
    custom_rm_path: str = ""
    custom_rm_function: Callable | None = None

    # ── SGLang / config overrides ───────────────────────────────────────────
    sglang_config: dict | None = None
    custom_config_path: dict | None = None
    apply_chat_template_kwargs: str = ""

    # ── Internal ──────────────────────────────────────────────────────────────

    # ── Container → slime flag converters ────────────────────────────────────
    #
    # Each converter maps a common config struct (`DatasetConfig`, `ModelConfig`,
    # `WandbConfig`) to the specific field names slime's CLI expects. Kept
    # explicit so the slime vocabulary lives in one place and the common
    # configs stay framework-agnostic.

    def __post_init__(self) -> None:
        if self.preset is None and self.model is not None:
            model_preset = getattr(self.model, "slime", None)
            if model_preset is not None:
                object.__setattr__(self, "preset", model_preset)

        if self.preset is None:
            raise ValueError(
                "SlimeConfig requires a SlimePreset. Pass preset=SlimePreset(...) "
                "or attach a SlimePreset to your model."
            )

        if self.custom_rm_function is not None and not self.custom_rm_path:
            fn = self.custom_rm_function
            mod = getattr(fn, "__module__", None) or ""
            name = getattr(fn, "__qualname__", None) or fn.__name__
            if mod == "__main__":
                import inspect

                src_file = inspect.getfile(fn)
                mod = Path(src_file).stem
            object.__setattr__(self, "custom_rm_path", f"{mod}.{name}")

    @staticmethod
    def _dataset_to_fields(ds: "DatasetConfig") -> dict[str, Any]:
        return {
            "prompt_data": ds.prompt_data,
            "eval_prompt_data": ds.eval_prompt_data,
            "input_key": ds.input_key,
            "label_key": ds.label_key,
            "apply_chat_template": ds.apply_chat_template,
            "rollout_shuffle": ds.rollout_shuffle,
        }

    @staticmethod
    def _validate_custom_model_architecture(
        m: "ModelConfig",
    ) -> "ModelArchitecture":
        """Require a `ModelArchitecture` on the attached model.

        slime consumes architecture fields as CLI flags, so a model without an
        explicit `ModelArchitecture` cannot produce a valid command line. This
        check fires before any remote Modal function is defined — invoked from
        `_model_to_fields()` (the converter that emits the flags) and from
        `build_slime_app()` preflight — so errors surface locally. Returns
        the non-None architecture for type-narrowing at the call site.
        """
        if m.architecture is None:
            raise ValueError(
                "SlimeConfig requires a ModelArchitecture on the attached "
                "ModelConfig. Set `architecture = ModelArchitecture(...)` "
                "on your subclass."
            )
        return m.architecture

    @staticmethod
    def _model_to_fields(m: "ModelConfig") -> dict[str, Any]:
        arch = SlimeConfig._validate_custom_model_architecture(m)
        return {
            "hf_checkpoint": m.model_name,
            "num_layers": arch.num_layers,
            "hidden_size": arch.hidden_size,
            "ffn_hidden_size": arch.ffn_hidden_size,
            "num_attention_heads": arch.num_attention_heads,
            "group_query_attention": arch.group_query_attention,
            "num_query_groups": arch.num_query_groups,
            "kv_channels": arch.kv_channels,
            "vocab_size": arch.vocab_size,
            "normalization": arch.normalization,
            "norm_epsilon": arch.norm_epsilon,
            "swiglu": arch.swiglu,
            "disable_bias_linear": arch.disable_bias_linear,
            "qk_layernorm": arch.qk_layernorm,
            "use_rotary_position_embeddings": arch.use_rotary_position_embeddings,
            "rotary_base": arch.rotary_base,
        }

    @staticmethod
    def _wandb_to_fields(w: "WandbConfig") -> dict[str, Any]:
        return {
            "use_wandb": True,
            "wandb_project": w.project,
            "wandb_group": w.group,
            "wandb_key": w.key,
            "disable_wandb_random_suffix": w.disable_random_suffix,
        }

    @staticmethod
    def _preset_to_fields(p: "SlimePreset") -> dict[str, Any]:
        return {
            "actor_num_nodes": p.actor_num_nodes,
            "actor_num_gpus_per_node": p.actor_num_gpus_per_node,
            "colocate": p.colocate,
            "tensor_model_parallel_size": p.tensor_model_parallel_size,
            "sequence_parallel": p.sequence_parallel,
            "rollout_num_gpus": p.rollout_num_gpus,
            "rollout_num_gpus_per_engine": p.rollout_num_gpus_per_engine,
            "use_critic": p.use_critic,
            "critic_num_nodes": p.critic_num_nodes,
            "critic_num_gpus_per_node": p.critic_num_gpus_per_node,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fields(self) -> dict[str, Any]:
        """Flat field dict for CLI generation.

        Attached containers (`dataset`, `model`, `wandb`, `preset`) are
        expanded last via the per-framework converters, so their values
        override direct fields with the same names.
        """
        import dataclasses as _dc

        fields: dict[str, Any] = {}
        for f in _dc.fields(self):
            fields[f.name] = getattr(self, f.name)
        if self.preset is not None:
            fields.update(self._preset_to_fields(self.preset))
        if self.dataset is not None:
            fields.update(self._dataset_to_fields(self.dataset))
        if self.model is not None:
            fields.update(self._model_to_fields(self.model))
        if self.wandb is not None:
            fields.update(self._wandb_to_fields(self.wandb))
        return {k: v for k, v in fields.items() if k not in _SLIME_SKIP}

    # ── Public API ────────────────────────────────────────────────────────────

    def cli_args(self) -> list[str]:
        """slime CLI arguments derived from this config.

        Conversion rules:
          field_name → --field-name  (underscore to hyphen)
          True       → --flag        (no value)
          False/None → omitted
          list       → --flag v1 v2 ...
          other      → --flag value
        """
        out: list[str] = []
        for key, val in self._fields().items():
            if val is None or val is False or val == "":
                continue
            flag = f"--{key.replace('_', '-')}"
            if val is True:
                out.append(flag)
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    def prepare_data(self) -> None:
        """Materialize the training data.

        If a `dataset` (`DatasetConfig`) is attached, delegate to it. Otherwise
        subclasses must override this method.
        """
        ds = getattr(self, "dataset", None)
        if ds is not None:
            ds.prepare()
            return
        raise NotImplementedError(f"{type(self).__name__} has no prepare_data()")

    @property
    def total_nodes(self) -> int:
        """Total Modal cluster nodes required by this config.

        Derived from actor/critic/rollout GPU counts. Modal provisions whole
        nodes, so we ceil-divide. Raises if total GPUs aren't a clean multiple
        of gpus_per_node.
        """
        f = self._fields()
        gpus_per_node = f.get("actor_num_gpus_per_node", 8)
        actor_nodes = f.get("actor_num_nodes", 1)
        colocate = f.get("colocate", False)
        use_critic = f.get("use_critic", False)
        critic_nodes = f.get("critic_num_nodes") or actor_nodes
        critic_gpus = f.get("critic_num_gpus_per_node") or gpus_per_node
        rollout_gpus = f.get("rollout_num_gpus")

        training_gpus = actor_nodes * gpus_per_node
        if use_critic:
            training_gpus += critic_nodes * critic_gpus

        if colocate:
            total_gpus = training_gpus
        else:
            rollout_gpus = rollout_gpus or (actor_nodes * gpus_per_node)
            total_gpus = training_gpus + rollout_gpus

        if total_gpus % gpus_per_node != 0:
            raise ValueError(
                f"total_gpus={total_gpus} is not a multiple of gpus_per_node={gpus_per_node}. "
                f"Adjust actor_num_nodes, rollout_num_gpus, or actor_num_gpus_per_node."
            )

        return math.ceil(total_gpus / gpus_per_node)

    checkpoint: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig(iteration_prefix="iter_")
    )

    def build_app(
        self,
        *,
        name: str | None = None,
        modal: ModalConfig | None = None,
    ) -> "App":
        from .launcher import build_slime_app
        from modal_training_gym.common.framework import resolve_app_name

        app_name = resolve_app_name("slime", name or self.name, self.model)
        assert self.preset is not None
        gpu = self.preset.gpu_type
        return build_slime_app(
            modal=modal or self.modal or ModalConfig(),
            gpu=gpu,
            slime=self,
            name=app_name,
        )


# Resolve forward references for pydantic validation
from modal_training_gym.common.dataset import DatasetConfig  # noqa: E402,F811
from modal_training_gym.common.models import ModelArchitecture, ModelConfig  # noqa: E402,F811
from modal_training_gym.common.wandb import WandbConfig  # noqa: E402,F811
from pydantic.dataclasses import rebuild_dataclass  # noqa: E402

rebuild_dataclass(SlimeConfig)
