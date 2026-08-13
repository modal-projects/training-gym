from dataclasses import dataclass

from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe

# Production-tuned SGLang recipe for DeepSeek-V4-Flash on 4×B200 (FP4 MoE).
# Mirrors the Modal LLM endpoint config (sglang v0.5.12.post1-cu130 + MegaMoE
# DeepGEMM kernels + DP attention), minus EAGLE — speculative decoding only
# helps the decode path, and this recipe's primary consumer is OPD teacher
# logprob prefills (``max_new_tokens=0``). Relative to the earlier cookbook-
# style defaults (pure TP=4, flashinfer_mxfp4, no DP), the prefill wins are:
#   - ``--moe-a2a-backend megamoe`` + DeepGEMM FP4 env vars (Blackwell MoE)
#   - ``dp=4`` / DP attention (4 independent prefill streams on one node)
#   - ``--enable-mixed-chunk`` + chunked prefill (better prefill packing)
#   - higher ``mem_fraction_static`` (more KV → more concurrent prefills)
_DEEPSEEK_V4_FLASH_DEFAULTS = {
    "gpu": "B200",
    "tp": 4,
    "dp": 4,
    "context_length": 262144,
    "mem_fraction_static": 0.85,
    "chunked_prefill_size": 4096,
    "max_running_requests": 16,
    # Matches the production Modal LLM endpoint image (cu130 / Blackwell).
    "sglang_image": "lmsysorg/sglang:v0.5.12.post1-cu130",
    # That image already ships DeepSeek-V4-aware transformers; forcing a git
    # install double-registers ``qwen3_asr`` and crashloops on import.
    "install_transformers_from_git": False,
    "env_vars": {
        "NCCL_CUMEM_ENABLE": "1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": "1",
        "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": "1",
    },
}

_DEEPSEEK_V4_FLASH_SERVER_ARGS = {
    "--trust-remote-code": "",
    "--moe-a2a-backend": "megamoe",
    "--enable-breakable-cuda-graph": "",
    "--enable-mixed-chunk": "",
    "--piecewise-cuda-graph-max-tokens": "4096",
    # Parse DeepSeek-V4's native DSML tool-call markup + reasoning blocks.
    "--tool-call-parser": "deepseekv4",
    "--reasoning-parser": "deepseek-v4",
}


_SGLANG_DEFAULTS = SglangRecipe()


@dataclass
class DeepSeek_V4_Flash_SglangRecipe(SglangRecipe):
    """DeepSeek-V4-Flash (284B MoE, 13B active) on 4×B200 — production SGLang defaults."""

    def __post_init__(self) -> None:
        for key, val in _DEEPSEEK_V4_FLASH_DEFAULTS.items():
            if getattr(self, key) == getattr(_SGLANG_DEFAULTS, key):
                object.__setattr__(self, key, val)
        merged = dict(_DEEPSEEK_V4_FLASH_SERVER_ARGS)
        merged.update(self.extra_server_args or {})
        object.__setattr__(self, "extra_server_args", merged)
