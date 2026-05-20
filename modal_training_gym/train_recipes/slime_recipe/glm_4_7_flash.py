from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_4_7_Flash_Recipe(SlimeRecipe):
    """GLM-4.7-Flash (30B-A3B MoE) on 1x8xH200, colocated GRPO."""

    gpu_type: str = "H200"
    colocate: bool = True
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus_per_engine: int = 8

    num_rollout: int = 3000
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 32768
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.7

    save_interval: int = 10

    n_samples_per_prompt: int = 8
    global_batch_size: int = 512
    lr: float = 1e-6
    max_tokens_per_gpu: int = 32768

    # MoE parallelism
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1

    # MoE training
    moe_token_dispatcher_type: str = "flex"
    moe_enable_deepep: bool = True
    attention_backend: str = "flash"

    # MTP
    mtp_num_layers: int = 1
    enable_mtp_training: bool = True
    mtp_loss_scaling_factor: float = 0.2

    # Optimizer
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # Rollout sglang
    sglang_enable_dp_attention: bool = True
    sglang_dp_size: int = 8
    sglang_enable_dp_lm_head: bool = True
    sglang_moe_dense_tp_size: int = 1
    sglang_speculative_algorithm: str = "EAGLE"
    sglang_speculative_num_steps: int = 2
    sglang_speculative_eagle_topk: int = 1
    sglang_speculative_num_draft_tokens: int = 3
    sglang_max_running_requests: int = 512

    eval_interval: int | None = 20
    eval_max_response_len: int = 32768
