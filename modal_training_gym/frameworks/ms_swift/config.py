"""Configuration classes for ms-swift Megatron training on Modal.

`MsSwiftConfig` holds all the `megatron sft` CLI flags; `cli_args()` emits
them using ms-swift's conventions (underscore-style flag names, string-valued
booleans). Containers (`dataset`, `model`, `wandb`) are interpreted
explicitly since ms-swift's flag vocabulary differs from the SLIME-style
SlimeConfig — containers aren't merged blindly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from modal_training_gym.common import GPUType

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Model
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
CHECKPOINTS_PATH = Path("/checkpoints")

# ── Types ─────────────────────────────────────────────────────────────────────

# Fields that are NOT emitted as megatron CLI flags (launcher-only or
# interpreted explicitly via container mapping in `_fields()`).
_SKIP_FIELDS = {
    "environment",
    "dataset",
    "model",
    "wandb",
    "n_nodes",
    "gpus_per_node",
    "app_tags",
}


class MsSwiftModalConfig:
    """Modal infrastructure for ms-swift — GPU family + image overrides."""

    gpu: GPUType = "B200"
    image: str = (
        "modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:"
        "ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.3"
    )
    # Baseten's shipped transformers may lack GLM-4 MoE support; reinstall this
    # version to ensure compatibility.
    transformers_version: str = "4.57.3"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class MsSwiftConfig:
    """ms-swift Megatron training configuration.

    Subclass and override class attributes to customize. All non-skip
    attributes are forwarded to `megatron sft` as `--flag value` CLI args
    (ms-swift uses underscore-style names and explicit `"true"`/`"false"`
    string-valued booleans).

    Attach `dataset` / `model` / `wandb` as containers; they're translated to
    ms-swift's specific flag names inside `_fields()`.
    """

    # ── Containers (interpreted in _fields) ──────────────────────────────────
    dataset: "DatasetConfig | None" = None
    model: "Model | None" = None
    wandb: "WandbConfig | None" = None

    # ── Modal app tags ───────────────────────────────────────────────────────
    # Merged with the framework's default tags (training/source/framework) at
    # app-build time. Use for per-run tagging.
    app_tags: dict = {}

    # ── Launcher-only (NOT megatron CLI flags) ───────────────────────────────
    environment: dict = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }
    n_nodes: int = 4
    gpus_per_node: int = 8

    # ── Tuner & data split ───────────────────────────────────────────────────
    tuner_type: str = "lora"
    split_dataset_ratio: float = 0.01
    # Bare flag — emitted as `--perform_initialization` (no value) when True.
    perform_initialization: bool = True

    # ── Parallelism ──────────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 2
    expert_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 4
    context_parallel_size: int = 1
    sequence_parallel: bool = True

    # ── MoE ──────────────────────────────────────────────────────────────────
    moe_permute_fusion: bool = True
    moe_grouped_gemm: bool = True
    moe_shared_expert_overlap: bool = True
    moe_aux_loss_coeff: float = 1e-3

    # ── Batch ────────────────────────────────────────────────────────────────
    global_batch_size: int = 8
    packing: bool = False
    use_precision_aware_optimizer: bool = True

    # ── Training ─────────────────────────────────────────────────────────────
    num_train_epochs: int = 4
    lr: float = 1e-4
    lr_warmup_fraction: float = 0.05
    lr_decay_iters: int = 100000
    min_lr: float = 1e-5

    # ── Context / attention ──────────────────────────────────────────────────
    max_length: int = 2048
    attention_backend: str = "flash"

    # ── IO / checkpointing ───────────────────────────────────────────────────
    dataset_num_proc: int = 8
    save_interval: int = 50
    no_save_optim: bool = True
    no_save_rng: bool = True
    use_hf: int = 1
    # Disable versioned sub-dir in save path. Required so cross-node rank 0
    # and last-rank see the same path (v0-timestamp suffix would race).
    add_version: bool = False

    # ── Logging / eval ───────────────────────────────────────────────────────
    log_interval: int = 1
    eval_iters: int = 10
    eval_interval: int = 50

    # ── LoRA ─────────────────────────────────────────────────────────────────
    target_modules: str = "all-linear"
    lora_rank: int = 128
    lora_alpha: int = 32
    merge_lora: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.environment = dict(type(self).environment)
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fields(self) -> dict[str, Any]:
        """Flat CLI field dict. Containers are interpreted explicitly."""
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
        fields = {k: v for k, v in fields.items() if k not in _SKIP_FIELDS}

        if self.dataset is not None and hasattr(self.dataset, "prompt_data"):
            fields["dataset"] = self.dataset.prompt_data
        if self.model is not None:
            fields["model"] = self.model.hf_checkpoint
        if self.wandb is not None:
            fields["report_to"] = "wandb"
            if self.wandb.project:
                fields["wandb_project"] = self.wandb.project
            # ms-swift uses `--wandb_exp_name` (not `--wandb_group`). Fall back
            # to `group` if `exp_name` isn't set so the user can pick either.
            exp = self.wandb.exp_name or self.wandb.group
            if exp:
                fields["wandb_exp_name"] = exp

        # Match the reference: only emit --eval_interval when eval_iters > 0.
        if fields.get("eval_iters", 0) <= 0:
            fields.pop("eval_interval", None)

        return fields

    # ── Public API ────────────────────────────────────────────────────────────

    def cli_args(self, *, output_dir: str) -> list[str]:
        """Build the list of `megatron sft ...` flags.

        `output_dir` is a per-run value provided by the launcher, not a field
        on the config, because it typically embeds the run_id.

        Conversion rules:
          field_name → --field_name   (ms-swift keeps underscores)
          True/False → --flag true|false   (string-valued, not bare)
          None       → omitted
          list       → --flag v1 v2 …
          other      → --flag value

        `perform_initialization` is emitted as a bare flag (no value) when
        True, to match the ms-swift reference; omitted when False.
        """
        out: list[str] = []
        fields = dict(self._fields())

        # Bare flag: emit without value, then remove from the general pass.
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
