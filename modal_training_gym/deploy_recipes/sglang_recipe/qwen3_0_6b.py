from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

_QWEN3_0_6B_DEFAULTS = {
    "gpu": "H100",
    "tp": 1,
    "context_length": 32768,
    "mem_fraction_static": 0.88,
    "chunked_prefill_size": 8192,
    "max_running_requests": 64,
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class Qwen3_0_6B_SglangRecipe(SglangRecipe):
    """Qwen3-0.6B SGLang recipe for 1×H100."""

    def __post_init__(self) -> None:
        for key, val in _QWEN3_0_6B_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
