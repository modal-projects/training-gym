from dataclasses import dataclass

from modal_training_gym.deploy_recipes.vllm_recipe.recipe import VllmRecipe

_QWEN3_4B_DEFAULTS = {
    "gpu": "H100",
    "n_gpu": 1,
}

_VLLM_DEFAULTS = VllmRecipe()


@dataclass
class Qwen3_4b_VllmRecipe(VllmRecipe):
    """Qwen3-4B vLLM recipe for 1×H100."""

    def __post_init__(self) -> None:
        for key, val in _QWEN3_4B_DEFAULTS.items():
            if getattr(self, key) == getattr(_VLLM_DEFAULTS, key):
                object.__setattr__(self, key, val)
