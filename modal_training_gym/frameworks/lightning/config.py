"""Configuration for PyTorch Lightning Fabric–launched multi-node training.

Mirrors the shape of the `torchrun` and `hf_accelerate` frameworks — runs a
user-provided script on a Modal cluster, but uses Lightning Fabric's
`fabric run` CLI as the process launcher. That unlocks Lightning-specific
strategies (`ddp`, `fsdp`, `deepspeed`, …) without writing distributed
launch code yourself.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from modal_training_gym.common import GPUType
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

DATASET_MOUNT_PATH = Path("/data")
MODEL_MOUNT_PATH = Path("/model")
SCRIPTS_MOUNT_PATH = Path("/scripts")

# ── Types ─────────────────────────────────────────────────────────────────────

Accelerator = Literal["cpu", "gpu", "tpu", "mps", "auto"]
Strategy = Literal["ddp", "ddp_spawn", "fsdp", "deepspeed", "auto"]

# Default training image. Add more Lightning-ecosystem deps here if needed.
_DEFAULT_TRAIN_PIP = (
    "click==8.1.8",  # required by Lightning's fabric CLI
    "torch==2.6.0",
    "lightning==2.4.0",
    "requests==2.32.3",
    "wandb",
)


@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class LightningFrameworkConfig:
    """Lightning Fabric settings, including Modal infrastructure."""

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu: GPUType = "H100"
    python_version: str = "3.12"
    pip_deps: tuple[str, ...] = _DEFAULT_TRAIN_PIP

    # ── Infrastructure ───────────────────────────────────────────────────────
    n_nodes: int = 2
    gpus_per_node: int = 8
    main_port: int = 29500

    # ── Training script ──────────────────────────────────────────────────────
    train_script_source: str = ""
    train_script_name: str = "train.py"

    # ── Script args ──────────────────────────────────────────────────────────
    script_args: list[str] = field(default_factory=list)

    # ── Lightning Fabric-specific ────────────────────────────────────────────
    accelerator: Accelerator = "gpu"
    strategy: Strategy = "ddp"
    precision: str = "bf16-mixed"

    # ── Modal app tags ───────────────────────────────────────────────────────
    # Merged with the framework's default tags (training/source/framework) at
    # app-build time. Use for per-run tagging (experiment id, user, env, …).
    app_tags: dict = field(default_factory=dict)


class LightningConfig:
    dataset: "DatasetConfig | None"
    model: "ModelConfiguration | None"
    wandb: "WandbConfig | None"
    framework_config: LightningFrameworkConfig

    def __init__(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfiguration | None" = None,
        wandb: "WandbConfig | None" = None,
        framework_config: LightningFrameworkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.wandb = wandb
        self.framework_config = framework_config or LightningFrameworkConfig()

    def build_app(
        self,
        *,
        name: str | None = None,
    ) -> "App":
        from .launcher import build_lightning_app

        return build_lightning_app(config=self, name=name)
