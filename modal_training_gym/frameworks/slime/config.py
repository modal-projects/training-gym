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
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Model
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# ── Types ─────────────────────────────────────────────────────────────────────

GPUType = Literal["H100", "H200", "B200", "B300", "A100"]

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
}

# SlimeConfig fields that SLIME reads as YAML files at runtime.
# Users may set these as inline dicts in Python configs; the launcher
# materializes them to temp YAML files before building the CLI command.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


class ModalConfig:
    """Modal infrastructure configuration — GPU provisioning and image setup only."""

    gpu: GPUType = "H200"
    local_slime: str | None = None  # path to local slime repo for dev overlay
    patch_files: list[
        str
    ] = []  # local patch files; each injected into image at /tmp/<filename>
    image_run_commands: list[
        str
    ] = []  # commands to run during image build (e.g. git apply /tmp/my.patch)

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
    model: "Model | None" = None
    # When set, wandb logging is enabled and the config's fields
    # (`wandb_project`, `wandb_group`, …) are merged into the flat field dict.
    wandb: "WandbConfig | None" = None

    def __init__(self, **kwargs: Any) -> None:
        # Fresh environment dict per instance — never mutate the class-level default.
        self.environment = dict(type(self).environment)
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fields(self) -> dict[str, Any]:
        """Merged field dict from the class hierarchy; instance attrs win.

        An attached `dataset` (`DatasetConfig`) has its fields merged last, so
        values set on the dataset override any direct attrs with the same names.
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
        for key in ("dataset", "model", "wandb"):
            container = fields.get(key)
            if container is not None and hasattr(container, "to_fields"):
                fields.update(container.to_fields())
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
        """Pull dataset-related fields out of this config into a `DatasetConfig`.

        Reverse of attaching a `DatasetConfig` as the `dataset` field — gives
        back a bare `DatasetConfig` carrying the current dataset values
        regardless of whether they came from a direct attr or an attached
        `DatasetConfig`. The returned instance has no `prepare()` method.
        """
        from modal_training_gym.common.dataset import DatasetConfig

        return DatasetConfig.from_fields(self._fields())

    def to_model(self) -> "Model":
        """Pull model-related fields out of this config into a `Model`.

        Reverse of attaching a `Model` as the `model` field — works whether
        the fields came from an attached `Model` or were set directly.
        """
        from modal_training_gym.common.models import Model

        return Model.from_fields(self._fields())

    def to_wandb_config(self) -> "WandbConfig":
        """Pull wandb-related fields out of this config into a `WandbConfig`.

        Reverse of attaching a `WandbConfig` as the `wandb` field — works
        whether the fields came from an attached `WandbConfig` or were set
        directly.
        """
        from modal_training_gym.common.wandb import WandbConfig

        return WandbConfig.from_fields(self._fields())

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
