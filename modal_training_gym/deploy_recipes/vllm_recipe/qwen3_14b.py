from dataclasses import dataclass

from modal_training_gym.deploy_recipes.vllm_recipe.recipe import VllmRecipe

_QWEN3_14B_DEFAULTS = {
    "gpu": "H100",
    "n_gpu": 2,
}

_VLLM_DEFAULTS = VllmRecipe()


@dataclass
class Qwen3_14b_VllmRecipe(VllmRecipe):
    """Qwen3-14B on 2×H100 — sensible vLLM defaults for a 14B model."""

    def __post_init__(self) -> None:
        for key, val in _QWEN3_14B_DEFAULTS.items():
            if getattr(self, key) == getattr(_VLLM_DEFAULTS, key):
                object.__setattr__(self, key, val)
