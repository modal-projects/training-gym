from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_4b_Recipe(SlimeRecipe):
    """Qwen3-4B GRPO recipe for 1 node with 8 H100 GPUs."""

    # ── Required fields with Qwen3-4B defaults ─────────────────────────────
    sequence_parallel: bool = False

    rollout_batch_size: int = 16
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0

    save_interval: int = 10

    # ── Overrides from SlimeRecipe defaults ─────────────────────────────────
    n_samples_per_prompt: int = 8
    lr: float = 5e-7
    max_tokens_per_gpu: int = 8192
    eval_interval: int | None = 10
    eval_max_response_len: int = 4096
