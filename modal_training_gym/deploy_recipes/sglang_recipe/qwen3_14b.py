from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

_QWEN3_14B_DEFAULTS = {
    "gpu": "H100",
    "tp": 2,
    "context_length": 32768,
    "mem_fraction_static": 0.8,
    "chunked_prefill_size": 8192,
    "max_running_requests": 16,
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class Qwen3_14b_SglangRecipe(SglangRecipe):
    """Qwen3-14B on 2×H100 — sensible SGLang defaults for a 14B model."""

    def __post_init__(self) -> None:
        for key, val in _QWEN3_14B_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
