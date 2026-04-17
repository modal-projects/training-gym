"""Configuration for NVIDIA NeMo Megatron (`megatron.bridge`) training on Modal.

`MegatronConfig` holds the training/parallelism/LoRA/optimizer knobs. Unlike
the SLIME and ms-swift frameworks, Megatron's training entrypoint is a Python
script (`train_script.py`) that imports `megatron.bridge` directly rather
than a CLI. The launcher serializes a `MegatronConfig` to a JSON file that
the script loads inside each container.

Containers (`dataset`, `model`, `wandb`) are interpreted explicitly — no
blind merge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Model
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE = Path("/root/.cache/huggingface")
MODELS_DIR = Path("/models")
DATA_DIR = Path("/data")
CHECKPOINTS_DIR = Path("/checkpoints")

# ── Types ─────────────────────────────────────────────────────────────────────

GPUType = Literal["H100", "H200", "B200", "B300", "A100"]


class MegatronModalConfig:
    """Modal infrastructure for Megatron — GPU family + image overrides."""

    gpu: GPUType = "B200"
    # NVIDIA NeMo image that ships megatron-core + megatron-bridge.
    nemo_image: str = "nvcr.io/nvidia/nemo:25.11"

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class MegatronConfig:
    """Megatron (NeMo bridge) training configuration.

    Subclass and override class attributes. Most fields map to the Python
    config constructed in `train_script.py` (`TrainingConfig`,
    `DistributedDataParallelConfig`, `LoRA`, etc.). The launcher serializes
    this config to a JSON file that the script reads at startup.
    """

    # ── Containers (interpreted in the launcher) ─────────────────────────────
    dataset: "DatasetConfig | None" = None
    model: "Model | None" = None
    wandb: "WandbConfig | None" = None

    # ── Infrastructure ───────────────────────────────────────────────────────
    n_nodes: int = 4
    gpus_per_node: int = 8

    # ── Parallelism (TP × PP × EP × CP must divide total GPUs) ───────────────
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 4
    expert_model_parallel_size: int = 4
    context_parallel_size: int = 4

    # ── Batch ────────────────────────────────────────────────────────────────
    micro_batch_size: int = 1
    global_batch_size: int = 32

    # ── Training ─────────────────────────────────────────────────────────────
    train_iters: int = 650
    seq_length: int = 131_072

    # ── Optimizer (cosine-annealed fused Adam) ───────────────────────────────
    lr: float = 1e-4
    min_lr: float = 0.0
    lr_warmup_iters: int = 50
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # ── Recompute (long-context friendly default: full recompute) ────────────
    recompute_num_layers: int = 1  # 1 = checkpoint every layer
    no_recompute: bool = False

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_dim: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ── Checkpointing ────────────────────────────────────────────────────────
    save_interval: int = 130

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ── Serialization for train_script.py ────────────────────────────────────

    def to_script_config(self) -> dict[str, Any]:
        """Dict serialized to JSON and consumed by `train_script.py`."""
        assert self.model is not None, "MegatronConfig.model must be set"
        return {
            "hf_checkpoint": self.model.hf_checkpoint,
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "expert_model_parallel_size": self.expert_model_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "train_iters": self.train_iters,
            "seq_length": self.seq_length,
            "lr": self.lr,
            "min_lr": self.min_lr,
            "lr_warmup_iters": self.lr_warmup_iters,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "recompute_num_layers": self.recompute_num_layers,
            "no_recompute": self.no_recompute,
            "lora_dim": self.lora_dim,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "save_interval": self.save_interval,
            "wandb_project": self.wandb.project if self.wandb else "",
        }

    def write_script_config(self, path: str) -> None:
        """Write the serialized config to a JSON file for the train script."""
        with open(path, "w") as f:
            json.dump(self.to_script_config(), f, indent=2)
