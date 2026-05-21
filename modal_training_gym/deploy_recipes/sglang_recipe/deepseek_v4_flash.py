from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

_DEEPSEEK_V4_FLASH_DEFAULTS = {
    "gpu": "B200",
    "tp": 4,
    "context_length": 32768,
    "mem_fraction_static": 0.80,
    "chunked_prefill_size": 8192,
    "max_running_requests": 16,
    "extra_server_args": {
        "--trust-remote-code": "",
        "--moe-a2a-backend": "deepep",
    },
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class DeepSeek_V4_Flash_SglangRecipe(SglangRecipe):
    """DeepSeek-V4-Flash (284B MoE, 13B active) on 4×B200 — sensible SGLang defaults."""

    def __post_init__(self) -> None:
        for key, val in _DEEPSEEK_V4_FLASH_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
