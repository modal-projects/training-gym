"""Qwen3-VL-8B recipe for vision-language GRPO on 1x8xH100."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_VL_8B_Recipe(SlimeRecipe):
    """Qwen3-VL-8B GRPO recipe for 1 node with 8 H100 GPUs."""

    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True

    num_rollout: int = 15
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 256
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.55

    save_interval: int = 10

    # AutoBridge loads the VL checkpoint (incl. ViT) at the configured TP; skips
    # slime's torch_dist pre-conversion, which mis-assigns the VL pipeline stage.
    megatron_to_hf_mode: str = "bridge"
