from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

_GLM_4_7_DEFAULTS = {
    "gpu": "H200",
    "tp": 8,
    "context_length": 32768,
    "mem_fraction_static": 0.70,
    "chunked_prefill_size": 8192,
    "max_running_requests": 8,
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class GLM_4_7_SglangRecipe(SglangRecipe):
    """GLM-4.7 (355B-A32B MoE) on 8xH200 — sensible SGLang defaults."""

    def __post_init__(self) -> None:
        for key, val in _GLM_4_7_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
