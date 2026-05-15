from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_4_7_Recipe(SlimeRecipe):
    """GLM-4.7 (355B-A32B MoE) on 8×8×H100, colocated GRPO.

    TP=8, PP=4, CP=2, EP=16 across 8 nodes (64 GPUs).
    Uses CPU optimizer offloading for the large parameter count.
    """

    gpu_type: str = "H100"
    colocate: bool = True
    actor_num_nodes: int = 8
    actor_num_gpus_per_node: int = 8
    tensor_model_parallel_size: int = 8
    sequence_parallel: bool = True
    rollout_num_gpus_per_engine: int = 32

    # MoE parallelism
    expert_model_parallel_size: int = 16
    expert_tensor_parallel_size: int = 1
    pipeline_model_parallel_size: int = 4
    context_parallel_size: int = 2
    attention_backend: str | None = "flash"

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
    max_tokens_per_gpu: int = 16384

    # Optimizer offloading (required for 355B model)
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    eval_interval: int | None = 10
    eval_max_response_len: int = 4096
