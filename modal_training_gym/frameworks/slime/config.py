"""Base configuration classes and volume mount paths for slime.

Two separate concerns:

  ModalConfig  — Modal infrastructure (gpu model, async mode, dev overlay)
  SlimeConfig  — SLIME training arguments

Each experiment defines one instance of each. All non-private, non-callable
attributes on a SlimeConfig subclass become SLIME CLI args automatically via
cli_args(). The 'environment' field is the only exception — it is injected
into the Ray job runtime env, not passed to SLIME directly.
"""

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modal_training_gym.common import GPUType

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import (
        ModelArchitecture,
        ModelConfiguration,
    )
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# ── Types ─────────────────────────────────────────────────────────────────────

# Fields on SlimeConfig that are NOT SLIME CLI args.
# `dataset`, `model`, and `wandb` are container objects; their own fields are
# expanded into cli args inside `_fields()`, but the containers themselves are
# never emitted as flags.
_SLIME_SKIP = {
    "environment",
    "async_mode",
    "slime_model_script",
    "dataset",
    "model",
    "wandb",
    "modal",
    "app_tags",
}

# SlimeConfig fields that SLIME reads as YAML files at runtime.
# Users may set these as inline dicts in Python configs; the launcher
# materializes them to temp YAML files before building the CLI command.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


class ModalConfig:
    """Modal infrastructure configuration — GPU provisioning and image setup only."""

    gpu: GPUType = "H100"
    local_slime: str | None = None  # path to local slime repo for dev overlay
    patch_files: list[
        str
    ] = []  # local patch files; each injected into image at /tmp/<filename>
    image_run_commands: list[
        str
    ] = []  # commands to run during image build (e.g. git apply /tmp/my.patch)
    # Sibling Python modules (by import name) to ship into the training image
    # via `add_local_python_source`. Use when a tutorial has helper modules
    # alongside it — e.g. a custom reward function referenced via SLIME's
    # `custom_rm_path`. Each name must be importable on the local sys.path
    # at the time `build_app()` runs.
    local_python_sources: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class SlimeConfig:
    """Base SLIME training configuration.

    Subclass and set class attributes to configure an experiment.
    All attributes (except 'environment') are forwarded to SLIME as CLI args.
    Each experiment must be fully self-contained — no inherited defaults.

    Fields in _SLIME_SKIP are launcher instructions, not SLIME CLI args:
      environment       — injected into the Ray job runtime env
      async_mode        — selects train_async.py vs train.py
      slime_model_script — path relative to /root/slime to a shell script that
                           defines MODEL_ARGS for model architecture; sourced
                           before running the train command

    Example:

        class MyExperiment(SlimeConfig):
            async_mode = False
            slime_model_script = ""
            hf_checkpoint = "Qwen/Qwen3-8B"
            actor_num_nodes = 1
            actor_num_gpus_per_node = 8
            megatron_to_hf_mode = "bridge"
            ...

        slime = MyExperiment()
    """

    # Launcher instructions — not passed to SLIME CLI (see _SLIME_SKIP).
    environment: dict = {
        "PYTHONPATH": "/root/Megatron-LM/",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
    }
    async_mode: bool = False  # True → use train_async.py
    slime_model_script: str = ""  # shell script path relative to /root/slime
    # When set, the dataset's fields (`prompt_data`, `input_key`, …) are merged
    # into the flat field dict and `prepare_data()` delegates to it.
    dataset: "DatasetConfig | None" = None
    # When set, the model's `hf_checkpoint` and architecture fields
    # (`num_layers`, `hidden_size`, …) are merged into the flat field dict.
    model: "ModelConfiguration | None" = None
    # When set, wandb logging is enabled and the config's fields
    # (`wandb_project`, `wandb_group`, …) are merged into the flat field dict.
    wandb: "WandbConfig | None" = None
    # Merged with the framework's default Modal app tags
    # (`training` / `source` / `framework`) at app-build time.
    app_tags: dict = {}
    modal: ModalConfig | None = None

    def __init__(self, **kwargs: Any) -> None:
        # Fresh environment dict per instance — never mutate the class-level default.
        self.environment = dict(type(self).environment)
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ── Internal ──────────────────────────────────────────────────────────────

    # ── Container → SLIME flag converters ────────────────────────────────────
    #
    # Each converter maps a common config struct (`DatasetConfig`, `ModelConfiguration`,
    # `WandbConfig`) to the specific field names SLIME's CLI expects. Kept
    # explicit so the SLIME vocabulary lives in one place and the common
    # configs stay framework-agnostic.

    @staticmethod
    def _dataset_to_fields(ds: "DatasetConfig") -> dict[str, Any]:
        return {
            "prompt_data": ds.prompt_data,
            "eval_prompt_data": ds.eval_prompt_data,
            "input_key": ds.input_key,
            "label_key": ds.label_key,
            "apply_chat_template": ds.apply_chat_template,
            "rollout_shuffle": ds.rollout_shuffle,
            "rm_type": ds.rm_type,
        }

    @staticmethod
    def _validate_custom_model_architecture(
        m: "ModelConfiguration",
    ) -> "ModelArchitecture":
        """Require a `ModelArchitecture` on the attached model.

        SLIME consumes architecture fields as CLI flags, so a model without an
        explicit `ModelArchitecture` cannot produce a valid command line. This
        check fires before any remote Modal function is defined — invoked from
        `_model_to_fields()` (the converter that emits the flags) and from
        `build_slime_app()` preflight — so errors surface locally. Returns
        the non-None architecture for type-narrowing at the call site.
        """
        if m.architecture is None:
            raise ValueError(
                "SlimeConfig requires a ModelArchitecture on the attached "
                "ModelConfiguration. Set `architecture = ModelArchitecture(...)` "
                "on your subclass."
            )
        return m.architecture

    @staticmethod
    def _model_to_fields(m: "ModelConfiguration") -> dict[str, Any]:
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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fields(self) -> dict[str, Any]:
        """Merged field dict from the class hierarchy; instance attrs win.

        Attached containers (`dataset`, `model`, `wandb`) are expanded last via
        the per-framework converters above, so their values override any
        direct attrs with the same names.
        """
        fields: dict[str, Any] = {}
        for cls in reversed(type(self).__mro__):
            if cls is object:
                continue
            fields.update(
                {
                    k: v
                    for k, v in vars(cls).items()
                    if not k.startswith("_") and not callable(v)
                }
            )
        fields.update(vars(self))
        if self.dataset is not None:
            fields.update(self._dataset_to_fields(self.dataset))
        if self.model is not None:
            fields.update(self._model_to_fields(self.model))
        if self.wandb is not None:
            fields.update(self._wandb_to_fields(self.wandb))
        return {k: v for k, v in fields.items() if k not in _SLIME_SKIP}

    # ── Public API ────────────────────────────────────────────────────────────

    def cli_args(self) -> list[str]:
        """SLIME CLI arguments derived from this config.

        Conversion rules:
          field_name → --field-name  (underscore to hyphen)
          True       → --flag        (no value)
          False/None → omitted
          list       → --flag v1 v2 ...
          other      → --flag value
        """
        out: list[str] = []
        for key, val in self._fields().items():
            if val is None or val is False:
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

    def to_dataset_config(self) -> "DatasetConfig":
        """Extract dataset-related SLIME flags back into a `DatasetConfig`.

        Reverse of attaching a `DatasetConfig` as the `dataset` field — works
        whether the fields came from an attached container or were set
        directly. The returned instance has no `prepare()` method.
        """
        from modal_training_gym.common.dataset import DatasetConfig

        f = self._fields()
        return DatasetConfig(
            prompt_data=f.get("prompt_data", ""),
            eval_prompt_data=f.get("eval_prompt_data"),
            input_key=f.get("input_key", ""),
            label_key=f.get("label_key", ""),
            apply_chat_template=f.get("apply_chat_template", True),
            rollout_shuffle=f.get("rollout_shuffle", True),
            rm_type=f.get("rm_type", ""),
        )

    def to_model(self) -> "ModelConfiguration":
        """Extract model-related SLIME flags back into a `ModelConfiguration`.

        Returns the attached `ModelConfiguration` when one is set. Otherwise,
        constructs an ad-hoc instance from the raw SLIME flag fields, with
        `architecture=None` when every architecture flag is at its default
        and a populated `ModelArchitecture` when any field differs.
        """
        if self.model is not None:
            return self.model

        from modal_training_gym.common.models import (
            ModelArchitecture,
            ModelConfiguration,
        )

        fields: dict[str, Any] = {}
        for cls in reversed(type(self).__mro__):
            if cls is object:
                continue
            fields.update(
                {
                    k: v
                    for k, v in vars(cls).items()
                    if not k.startswith("_") and not callable(v)
                }
            )
        fields.update(vars(self))

        arch_candidate = ModelArchitecture(
            num_layers=fields.get("num_layers", 0),
            hidden_size=fields.get("hidden_size", 0),
            ffn_hidden_size=fields.get("ffn_hidden_size", 0),
            num_attention_heads=fields.get("num_attention_heads", 0),
            group_query_attention=fields.get("group_query_attention", True),
            num_query_groups=fields.get("num_query_groups", 0),
            kv_channels=fields.get("kv_channels", 0),
            vocab_size=fields.get("vocab_size", 0),
            normalization=fields.get("normalization", "RMSNorm"),
            norm_epsilon=fields.get("norm_epsilon", 1e-6),
            swiglu=fields.get("swiglu", True),
            disable_bias_linear=fields.get("disable_bias_linear", True),
            qk_layernorm=fields.get("qk_layernorm", True),
            use_rotary_position_embeddings=fields.get(
                "use_rotary_position_embeddings", True
            ),
            rotary_base=fields.get("rotary_base", 10000),
        )
        architecture = arch_candidate if arch_candidate != ModelArchitecture() else None
        return ModelConfiguration(
            model_name=fields.get("hf_checkpoint", "") or "",
            architecture=architecture,
        )

    def to_wandb_config(self) -> "WandbConfig":
        """Extract wandb-related SLIME flags back into a `WandbConfig`.

        Reverse of attaching a `WandbConfig` — works whether fields came from
        an attached container or were set directly.
        """
        from modal_training_gym.common.wandb import WandbConfig

        f = self._fields()
        return WandbConfig(
            project=f.get("wandb_project", ""),
            group=f.get("wandb_group", ""),
            key=f.get("wandb_key", ""),
            disable_random_suffix=f.get("disable_wandb_random_suffix", True),
        )

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

    def build_app(
        self,
        *,
        name: str | None = None,
        modal: ModalConfig | None = None,
    ) -> "App":
        from .launcher import build_slime_app

        return build_slime_app(
            modal=modal or self.modal or ModalConfig(),
            slime=self,
            name=name,
        )
