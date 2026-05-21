from collections.abc import Callable

import modal
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


def _glm_image_overlay(image: modal.Image) -> modal.Image:
    return image.run_commands(
        "uv pip install --system 'transformers>=5' 'tokenizers>=0.21'"
    )


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_4_7_Recipe(SlimeRecipe):
    """GLM-4.7 (355B-A32B MoE) on 8x8xH200, colocated GSPO."""

    image_overlay: Callable[[modal.Image], modal.Image] | None = _glm_image_overlay
    gpu_type: str = "H200"
    colocate: bool = True
    tensor_model_parallel_size: int = 8
    sequence_parallel: bool = True
    rollout_num_gpus_per_engine: int = 32

    num_rollout: int = 3000
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 8192
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.7

    save_interval: int = 10

    actor_num_nodes: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    lr: float = 1e-6
    max_tokens_per_gpu: int = 16384

    # MoE parallelism
    pipeline_model_parallel_size: int = 4
    context_parallel_size: int = 2
    expert_model_parallel_size: int = 16
    expert_tensor_parallel_size: int = 1
    attention_backend: str = "flash"

    # RL algorithm (GSPO)
    advantage_estimator: str = "gspo"
    eps_clip: float = 1e-4
    eps_clip_high: float = 2e-4
    use_tis: bool = True
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0

    # Optimizer
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # Rollout sglang
    sglang_enable_dp_attention: bool = True
    sglang_dp_size: int = 4
    sglang_ep_size: int = 32
    sglang_enable_dp_lm_head: bool = True
    sglang_moe_dense_tp_size: int = 1
    sglang_speculative_algorithm: str = "EAGLE"
    sglang_speculative_num_steps: int = 3
    sglang_speculative_eagle_topk: int = 1
    sglang_speculative_num_draft_tokens: int = 4

    # Data
    num_steps_per_rollout: int = 4
    balance_data: bool = True
    rollout_stop_token_ids: list[int] | None = None
    skip_eval_before_train: bool = True

    eval_interval: int | None = 20
    eval_max_response_len: int = 8192

    def __post_init__(self) -> None:
        if self.rollout_stop_token_ids is None:
            object.__setattr__(self, "rollout_stop_token_ids", [151329, 151336, 151338])
