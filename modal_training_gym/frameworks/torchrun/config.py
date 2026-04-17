"""Configuration for generic `torchrun`-launched multi-node training on Modal.

Unlike the framework-specific launchers (slime, ms-swift, megatron, verl),
this one doesn't speak a specific training framework's CLI. It just runs a
user-provided Python script under `torchrun` on a Modal cluster. The script
source is uploaded to a scripts volume by `app.upload_script` and the
launcher points torchrun at the materialized file.
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


class TorchrunModalConfig:
    """Modal infrastructure for torchrun — GPU family, Python, pip deps."""

    gpu: GPUType = "H100"
    python_version: str = "3.12"
    # Additional pip deps for the training image. Override to add / replace.
    pip_deps: tuple[str, ...] = _DEFAULT_TRAIN_PIP

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class TorchrunConfig:
    """Config for a torchrun-launched multi-node run.

    Containers (`dataset`, `model`, `wandb`) are interpreted explicitly — the
    launcher reads fields off them rather than merging `.to_fields()`. The
    training script decides how to consume them via `script_args`.
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
    # Source written to the scripts volume by `app.upload_script`.
    train_script_source: str = ""
    train_script_name: str = "train.py"

    # ── Script args (passed after the torchrun launcher args) ────────────────
    # Values here are forwarded verbatim; the launcher also injects
    # `--data_dir` / `--output_dir` / `--model_cache_dir` from the containers.
    script_args: list[str] = []

    # ── Modal app tags ───────────────────────────────────────────────────────
    # Merged with the framework's default tags (training/source/framework) at
    # app-build time. Use for per-run tagging (experiment id, user, env, …).
    app_tags: dict = {}

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
