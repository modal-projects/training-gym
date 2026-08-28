from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.models import ModelConfig, Qwen3_5_4B
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_5_4b_Miles_Recipe(MilesRecipe):
    """Qwen3.5-4B on 1x8xH100, following Miles' colocated GRPO recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Qwen3_5_4B

    miles_model_name: str = "qwen3.5-4B"
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "CONVERT_KEEP_PP1": "1",
        }
    )
    ref_load: str = "/checkpoints/Qwen3.5-4B_torch_dist"
    megatron_to_hf_mode: str = ""

    num_rollout: int = 3000
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 8192
    global_batch_size: int = 256
    balance_data: bool = True

    eval_interval: int | None = 20
    n_samples_per_eval_prompt: int = 16

    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    context_parallel_size: int | None = 1
    expert_model_parallel_size: int | None = 1
    expert_tensor_parallel_size: int | None = 1

    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1

    rm_type: str | None = "deepscaler"
    use_kl_loss: bool = True

    sglang_mem_fraction_static: float = 0.7

    attention_backend: str | None = "flash"

    save_interval: int = 20
