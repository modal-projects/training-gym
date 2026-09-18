from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.models import ModelConfig, Qwen3_30B
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_30B_A3B_Tinker_Recipe(MilesRecipe):
    """Qwen3-30B-A3B multi-LoRA Tinker gateway on one 8-GPU node.

    Mirrors Miles' ``examples/multi_lora/serve_qwen3_30b_a3b_tinker.py``: four
    training GPUs (TP2 for the dense layers, EP4 for the routed experts), four
    sampling GPUs as two 2-GPU SGLang engines, and four adapter slots of rank
    up to 32 covering attention, the per-expert MLP projections, and the
    output layer. Clients connect with ``tinker==0.26.2``.
    """

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_30B

    # Multi-LoRA v2 (the Tinker gateway) landed upstream on 2026-09-16; the
    # ref pin keeps the served protocol fixed until an image bundles it.
    docker_image: str = "radixark/miles:dev-202609161228"
    miles_git_ref: str | None = "f6d0257d83b7c14f4ce43ecfcd71d955112f6e0e"
    miles_model_name: str = "qwen3-30B-A3B"

    colocate: bool = False
    actor_num_gpus_per_node: int = 4
    rollout_num_gpus: int | None = 4
    rollout_num_gpus_per_engine: int = 2

    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    expert_model_parallel_size: int = 4

    # Slot pool; each client's rank comes from the SDK, capped by lora_rank.
    multi_lora_n_adapters: int | None = 4
    lora_rank: int | None = 32
    lora_alpha: int | None = 64
    lora_dropout: float | None = 0.0
    no_gradient_accumulation_fusion: bool = True
    sglang_lora_backend: str | None = "triton"

    # Initial optimizer config only; AdamParams arrive per optim_step request.
    lr: float = 1e-4

    max_tokens_per_gpu: int = 8192

    sglang_mem_fraction_static: float = 0.7

    attention_backend: str | None = "flash"
