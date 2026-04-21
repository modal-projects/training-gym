"""Configuration for HuggingFace `accelerate launch`-based multi-node training.

Like the sibling `torchrun` framework, this runs a user-provided training
script on a Modal cluster, but uses `accelerate launch` as the process
launcher instead of `torchrun`. Accelerate adds amenities like
`--mixed_precision`, config-file-driven launches, etc.
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

MixedPrecision = Literal["no", "fp16", "bf16", "fp8"]

# Default training image (matches the reference starcoder example).
_DEFAULT_TRAIN_PIP = (
    "datasets>=2.19",
    "sympy",
    "transformers",
    "trl",
    "wandb",
    "huggingface_hub",
    "torch==2.6.0",
    "accelerate",
    # Required by modal_training_gym config modules + cloudpickle
    # deserialization on the remote side.
    "cloudpickle",
    "msgspec",
    "pydantic",
)


@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class AccelerateFrameworkConfig:
    """accelerate settings, including Modal infrastructure."""

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu: GPUType = "H100"
    python_version: str = "3.12"
    pip_deps: tuple[str, ...] = _DEFAULT_TRAIN_PIP

    # ── Infrastructure ───────────────────────────────────────────────────────
    n_nodes: int = 2
    gpus_per_node: int = 8
    master_port: int = 29500

    # ── Training script ──────────────────────────────────────────────────────
    train_script_source: str = ""
    train_script_name: str = "train.py"

    # ── Script args ──────────────────────────────────────────────────────────
    script_args: list[str] = field(default_factory=list)

    # ── Accelerate-specific ──────────────────────────────────────────────────
    mixed_precision: MixedPrecision = "bf16"

    # ── Modal app tags ───────────────────────────────────────────────────────
    # Merged with the framework's default tags (training/source/framework) at
    # app-build time. Use for per-run tagging (experiment id, user, env, …).
    app_tags: dict = field(default_factory=dict)


class AccelerateConfig:
    dataset: "DatasetConfig | None"
    model: "ModelConfiguration | None"
    wandb: "WandbConfig | None"
    framework_config: AccelerateFrameworkConfig

    def __init__(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfiguration | None" = None,
        wandb: "WandbConfig | None" = None,
        framework_config: AccelerateFrameworkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.wandb = wandb
        self.framework_config = framework_config or AccelerateFrameworkConfig()

    def build_app(
        self,
        *,
        name: str | None = None,
    ) -> "App":
        from .launcher import build_accelerate_app

        return build_accelerate_app(config=self, name=name)
