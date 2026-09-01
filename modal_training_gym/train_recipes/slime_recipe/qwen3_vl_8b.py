"""Qwen3-VL-8B recipe for vision-language GRPO on 1x8xH100."""

from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_VL_8b_Recipe(SlimeRecipe):
    """Qwen3-VL-8B vision-language GRPO on 1×8×H100, colocated."""

    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True

    num_rollout: int = 15
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 256
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.55

    # VL image patches expand to many tokens; padded (bshd) batches avoid the
    # THD packing path, which needs dynamic batching off + explicit micro batch.
    use_dynamic_batch_size: bool = False
    extra_config: dict | None = field(
        default_factory=lambda: {"qkv_format": "bshd", "micro_batch_size": 1}
    )

    save_interval: int = 10

    # AutoBridge loads the VL checkpoint (incl. ViT) at the configured TP; skips
    # slime's torch_dist pre-conversion, which mis-assigns the VL pipeline stage.
    megatron_to_hf_mode: str = "bridge"

    # Freeze the vision tower; RL only updates the language backbone.
    freeze_params_name_list: list[str] | None = field(
        default_factory=lambda: ["vision_model"]
    )
