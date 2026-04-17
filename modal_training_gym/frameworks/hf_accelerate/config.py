"""Configuration for HuggingFace `accelerate launch`-based multi-node training.

Like the sibling `torchrun` framework, this runs a user-provided training
script on a Modal cluster, but uses `accelerate launch` as the process
launcher instead of `torchrun`. Accelerate adds amenities like
`--mixed_precision`, config-file-driven launches, etc.
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
)


class AccelerateModalConfig:
    """Modal infrastructure for accelerate — GPU, Python, pip deps."""

    gpu: GPUType = "H100"
    python_version: str = "3.12"
    pip_deps: tuple[str, ...] = _DEFAULT_TRAIN_PIP

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class AccelerateConfig:
    """Config for an `accelerate launch`-driven multi-node run.

    Mirrors `TorchrunConfig` for cross-launcher compatibility — the same
    `dataset` / `model` / `wandb` containers and `train_script_source` can be
    reused across both frameworks. Accelerate-specific knobs (e.g.
    `mixed_precision`) live on this class only.
    """

    # ── Containers ───────────────────────────────────────────────────────────
    dataset: "DatasetConfig | None" = None
    model: "Model | None" = None
    wandb: "WandbConfig | None" = None

    # ── Infrastructure ───────────────────────────────────────────────────────
    n_nodes: int = 2
    gpus_per_node: int = 8
    master_port: int = 29500

    # ── Training script ──────────────────────────────────────────────────────
    train_script_source: str = ""
    train_script_name: str = "train.py"

    # ── Script args ──────────────────────────────────────────────────────────
    script_args: list[str] = []

    # ── Accelerate-specific ──────────────────────────────────────────────────
    mixed_precision: MixedPrecision = "bf16"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
