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
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modal_training_gym.common import GPUType
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    from modal import App

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration
    from modal_training_gym.common.wandb import WandbConfig

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE = Path("/root/.cache/huggingface")
MODELS_DIR = Path("/models")
DATA_DIR = Path("/data")
CHECKPOINTS_DIR = Path("/checkpoints")

@dataclass(config=ConfigDict(extra="forbid", validate_assignment=True))
class MegatronFrameworkConfig:
    """Megatron (NeMo bridge) configuration, including Modal infrastructure.

    Subclass and override class attributes. Most fields map to the Python
    config constructed in `train_script.py` (`TrainingConfig`,
    `DistributedDataParallelConfig`, `LoRA`, etc.). The launcher serializes
    this config to a JSON file that the script reads at startup.
    """

    # ── Modal infrastructure ────────────────────────────────────────────────
    gpu: GPUType = "B200"
    # NVIDIA NeMo image that ships megatron-core + megatron-bridge.
    nemo_image: str = "nvcr.io/nvidia/nemo:25.11"

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

    # ── Modal app tags ───────────────────────────────────────────────────────
    # Merged with the framework's default tags (training/source/framework) at
    # app-build time. Use for per-run tagging.
    app_tags: dict = field(default_factory=dict)


class MegatronConfig:
    dataset: "DatasetConfig | None"
    model: "ModelConfiguration | None"
    wandb: "WandbConfig | None"
    framework_config: MegatronFrameworkConfig

    def __init__(
        self,
        dataset: "DatasetConfig | None" = None,
        model: "ModelConfiguration | None" = None,
        wandb: "WandbConfig | None" = None,
        framework_config: MegatronFrameworkConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.wandb = wandb
        self.framework_config = framework_config or MegatronFrameworkConfig()

    def to_script_config(self) -> dict[str, Any]:
        assert self.model is not None, "MegatronConfig.model must be set"
        cfg = self.framework_config
        return {
            "hf_checkpoint": self.model.model_name,
            "tensor_model_parallel_size": cfg.tensor_model_parallel_size,
            "pipeline_model_parallel_size": cfg.pipeline_model_parallel_size,
            "expert_model_parallel_size": cfg.expert_model_parallel_size,
            "context_parallel_size": cfg.context_parallel_size,
            "micro_batch_size": cfg.micro_batch_size,
            "global_batch_size": cfg.global_batch_size,
            "train_iters": cfg.train_iters,
            "seq_length": cfg.seq_length,
            "lr": cfg.lr,
            "min_lr": cfg.min_lr,
            "lr_warmup_iters": cfg.lr_warmup_iters,
            "weight_decay": cfg.weight_decay,
            "adam_beta1": cfg.adam_beta1,
            "adam_beta2": cfg.adam_beta2,
            "adam_eps": cfg.adam_eps,
            "recompute_num_layers": cfg.recompute_num_layers,
            "no_recompute": cfg.no_recompute,
            "lora_dim": cfg.lora_dim,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "save_interval": cfg.save_interval,
            "wandb_project": self.wandb.project if self.wandb else "",
        }

    def write_script_config(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_script_config(), f, indent=2)

    def build_app(
        self,
        *,
        name: str | None = None,
    ) -> "App":
        from .launcher import build_megatron_app

        return build_megatron_app(megatron=self, name=name)
