from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_4_7_Flash_Recipe(SlimeRecipe):
    """GLM-4.7-Flash (30B-A3B MoE) on 1×8×H100, colocated GRPO.

    TP=1, PP=1, EP=8 fits on a single 8-GPU node.
    Uses MTP training, DeepEP, and CPU optimizer offloading.
    """

    gpu_type: str = "H100"
    colocate: bool = True
    tensor_model_parallel_size: int = 1
    sequence_parallel: bool = False
    rollout_num_gpus_per_engine: int = 8

    # MoE parallelism
    expert_model_parallel_size: int = 8
    expert_tensor_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    moe_token_dispatcher_type: str | None = "flex"
    moe_enable_deepep: bool = True

    # MTP
    enable_mtp_training: bool = True
    mtp_num_layers: int = 1
    mtp_loss_scaling_factor: float = 0.2

    # Rollout
    num_rollout: int = 1
    rollout_batch_size: int = 64
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.70

    save_interval: int = 10

    # Training
    n_samples_per_prompt: int = 8
    global_batch_size: int = 512
    lr: float = 1e-6
    max_tokens_per_gpu: int = 32768
    attention_backend: str | None = "flash"

    # Optimizer offloading (MoE models are memory-heavy)
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    eval_interval: int | None = 10
    eval_max_response_len: int = 4096
