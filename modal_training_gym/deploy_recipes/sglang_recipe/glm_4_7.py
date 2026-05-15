from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

_GLM_4_7_DEFAULTS = {
    "gpu": "H100",
    "tp": 8,
    "context_length": 32768,
    "mem_fraction_static": 0.80,
    "chunked_prefill_size": 8192,
    "max_running_requests": 16,
    "extra_server_args": {"--trust-remote-code": ""},
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class GLM_4_7_SglangRecipe(SglangRecipe):
    """GLM-4.7 (355B) on 8×H100 — tensor-parallel MoE serving."""

    def __post_init__(self) -> None:
        for key, val in _GLM_4_7_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
