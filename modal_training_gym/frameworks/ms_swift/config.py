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
    from modal_training_gym.common.models import ModelConfiguration
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


@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class MsSwiftFrameworkConfig:
    """ms-swift Megatron configuration, including Modal infrastructure.

    Subclass and override class attributes to customize. All non-skip
    attributes are forwarded to `megatron sft` as `--flag value` CLI args
    (ms-swift uses underscore-style names and explicit `"true"`/`"false"`
    string-valued booleans).

    Attach `dataset` / `model` / `wandb` as containers; they're translated to
    ms-swift's specific flag names inside `_fields()`.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu: GPUType = "B200"
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
    train_iters: int = 0
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

    @classmethod
    def from_toml(cls, path: str | Path) -> "MsSwiftFrameworkConfig":
        import msgspec

        payload = msgspec.toml.decode(Path(path).read_bytes())
        if not isinstance(payload, dict):
            raise ValueError(f"Expected TOML table at root in {path!r}")
        return cls(**payload)


class MsSwiftConfig:
    """ms-swift run config with a built-in Modal app constructor."""

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
