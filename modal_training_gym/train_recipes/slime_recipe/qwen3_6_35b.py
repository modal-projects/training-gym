from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_35B_Recipe(SlimeRecipe):
    """Qwen3.6-35B-A3B recipe."""

    slime_model_script: str = "scripts/models/qwen3.5-35B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    actor_num_gpus_per_node: int = 4
    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 2
    expert_tensor_parallel_size: int = 1

    rollout_num_gpus_per_engine: int = 2
    sglang_enable_dp_attention: bool = True
    sglang_dp_size: int | None = 2
    sglang_ep_size: int | None = 2
    sglang_enable_dp_lm_head: bool = True
    sglang_cuda_graph_bs: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 8))
    )
    sglang_max_running_requests: int | None = 512
    sglang_speculative_algorithm: str | None = "EAGLE"
    sglang_speculative_num_steps: int | None = 3
    sglang_speculative_eagle_topk: int | None = 1
    sglang_speculative_num_draft_tokens: int | None = 4
    sglang_mamba_scheduler_strategy: str = "extra_buffer"

    max_tokens_per_gpu: int = 8192
    balance_data: bool = True
    moe_token_dispatcher_type: str = "flex"
    moe_enable_deepep: bool = True
    use_kl_loss: bool = True

    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    attention_backend: str = "flash"

    ref_load: str = "/checkpoints/Qwen3.6-35B-A3B_torch_dist_tp2pp2"
    no_save_optim: bool = True
