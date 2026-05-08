from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_14b_Recipe(SlimeRecipe):
    """Qwen3-14B on 1×8×H100 with TP2, colocated GRPO."""

    gpu_type: str = "H100"
    colocate: bool = True
    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    rollout_num_gpus_per_engine: int = 2

    num_rollout: int = 1
    rollout_batch_size: int = 8
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.7

    save_interval: int = 10

    n_samples_per_prompt: int = 8
    lr: float = 5e-7
    max_tokens_per_gpu: int = 4608
    eval_interval: int | None = 10
    eval_max_response_len: int = 4096
