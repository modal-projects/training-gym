from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

_QWEN3_6_35B_DEFAULTS = {
    "gpu": "H100",
    "tp": 4,
    "context_length": 262144,
    "mem_fraction_static": 0.80,
    "chunked_prefill_size": 8192,
    "max_running_requests": 16,
}

# Native tool calling + reasoning parsing for the mini-SWE agent's bash tool.
# Qwen3 series (non-coder) uses the `qwen` tool-call parser; `--reasoning-parser
# qwen3` separates <think> blocks (the agent opts out per-request with
# separate_reasoning=False to keep them inline).
_QWEN3_6_35B_PARSER_ARGS = {
    "--tool-call-parser": "qwen",
    "--reasoning-parser": "qwen3",
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class Qwen3_6_35B_SglangRecipe(SglangRecipe):
    """Qwen3.6-35B-A3B (MoE) SGLang recipe for 4×H100."""

    def __post_init__(self) -> None:
        for key, val in _QWEN3_6_35B_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
        merged = dict(_QWEN3_6_35B_PARSER_ARGS)
        merged.update(self.extra_server_args or {})
        object.__setattr__(self, "extra_server_args", merged)
