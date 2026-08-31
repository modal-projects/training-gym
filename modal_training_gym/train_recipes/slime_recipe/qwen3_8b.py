from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_8b_Recipe(SlimeRecipe):
    """Qwen3-8B GRPO recipe for 1 node with 8 H100 GPUs."""

    sequence_parallel: bool = False

    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.72

    save_interval: int = 10

    n_samples_per_prompt: int = 8
    lr: float = 5e-7
    max_tokens_per_gpu: int = 6144
    eval_interval: int | None = 10
    eval_max_response_len: int = 4096
