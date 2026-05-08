from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

_QWEN3_32B_DEFAULTS = {
    "gpu": "H100",
    "tp": 4,
    "context_length": 32768,
    "mem_fraction_static": 0.75,
    "chunked_prefill_size": 8192,
    "max_running_requests": 16,
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class Qwen3_32b_SglangRecipe(SglangRecipe):
    """Qwen3-32B on 4×H100 — sensible SGLang defaults."""

    def __post_init__(self) -> None:
        for key, val in _QWEN3_32B_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
