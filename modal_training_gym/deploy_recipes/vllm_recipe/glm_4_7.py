from dataclasses import dataclass

from modal_training_gym.deploy_recipes.vllm_recipe.recipe import VllmRecipe

_GLM_4_7_DEFAULTS = {
    "gpu": "H100",
    "n_gpu": 8,
    "extra_vllm_args": ["--trust-remote-code"],
}

_VLLM_DEFAULTS = VllmRecipe()


@dataclass
class GLM_4_7_VllmRecipe(VllmRecipe):
    """GLM-4.7 (355B) on 8×H100 — tensor-parallel MoE serving."""

    def __post_init__(self) -> None:
        for key, val in _GLM_4_7_DEFAULTS.items():
            if getattr(self, key) == getattr(_VLLM_DEFAULTS, key):
                object.__setattr__(self, key, val)
