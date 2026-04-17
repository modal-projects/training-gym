"""Configuration for PyTorch Lightning Fabric–launched multi-node training.

Mirrors the shape of the `torchrun` and `hf_accelerate` frameworks — runs a
user-provided script on a Modal cluster, but uses Lightning Fabric's
`fabric run` CLI as the process launcher. That unlocks Lightning-specific
strategies (`ddp`, `fsdp`, `deepspeed`, …) without writing distributed
launch code yourself.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Model
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

DATASET_MOUNT_PATH = Path("/data")
MODEL_MOUNT_PATH = Path("/model")
SCRIPTS_MOUNT_PATH = Path("/scripts")

# ── Types ─────────────────────────────────────────────────────────────────────

GPUType = Literal["H100", "H200", "B200", "B300", "A100"]
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


class LightningModalConfig:
    """Modal infrastructure for Lightning — GPU, Python, pip deps."""

    gpu: GPUType = "H100"
    python_version: str = "3.12"
    pip_deps: tuple[str, ...] = _DEFAULT_TRAIN_PIP

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class LightningConfig:
    """Config for a `fabric run`-launched multi-node Lightning run.

    Mirrors `TorchrunConfig` / `AccelerateConfig` so the same
    `dataset` / `model` / `wandb` containers + `train_script_source` can be
    reused across launchers. Lightning-specific knobs (`accelerator`,
    `strategy`) live on this class only.
    """

    # ── Containers ───────────────────────────────────────────────────────────
    dataset: "DatasetConfig | None" = None
    model: "Model | None" = None
    wandb: "WandbConfig | None" = None

    # ── Infrastructure ───────────────────────────────────────────────────────
    n_nodes: int = 2
    gpus_per_node: int = 8
    main_port: int = 29500

    # ── Training script ──────────────────────────────────────────────────────
    train_script_source: str = ""
    train_script_name: str = "train.py"

    # ── Script args ──────────────────────────────────────────────────────────
    script_args: list[str] = []

    # ── Lightning Fabric-specific ────────────────────────────────────────────
    accelerator: Accelerator = "gpu"
    strategy: Strategy = "ddp"
    precision: str = "bf16-mixed"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
