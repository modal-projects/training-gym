from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class GLM_4_7_Recipe(SlimeRecipe):
    """GLM-4.7 recipe."""

    gpu_type: str = "H200"
    memory: int | tuple[int, int] | None = (128, 2_097_152)
    slime_model_script: str = "scripts/models/glm4.5-355B-A32B.sh"
    hf_checkpoint: str = "zai-org/GLM-4.7"
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "DEPRECATED_MEGATRON_COMPATIBLE": "1",
        }
    )
    colocate: bool = False
    actor_num_nodes: int = 8
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int | None = 64
    tensor_model_parallel_size: int = 8
    sequence_parallel: bool = True
    rollout_num_gpus_per_engine: int = 32

    sglang_mem_fraction_static: float = 0.7

    async_save: bool = True
    use_persistent_ckpt_worker: bool = True

    max_tokens_per_gpu: int = 8192

    pipeline_model_parallel_size: int = 4
    context_parallel_size: int = 2
    expert_model_parallel_size: int = 16
    expert_tensor_parallel_size: int = 1
    attention_backend: str = "flash"

    advantage_estimator: str = "gspo"
    eps_clip: float = 1e-4
    eps_clip_high: float = 2e-4
    use_tis: bool = True

    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    sglang_enable_dp_attention: bool = True
    sglang_dp_size: int | None = 4
    sglang_ep_size: int | None = 32
    sglang_enable_dp_lm_head: bool = True
    sglang_moe_dense_tp_size: int = 1
    sglang_speculative_algorithm: str | None = None
    sglang_speculative_num_steps: int | None = None
    sglang_speculative_eagle_topk: int | None = None
    sglang_speculative_num_draft_tokens: int | None = None

    balance_data: bool = True
    rollout_stop_token_ids: list[int] | None = None
    skip_eval_before_train: bool = True

    def __post_init__(self) -> None:
        if self.rollout_stop_token_ids is None:
            object.__setattr__(self, "rollout_stop_token_ids", [151329, 151336, 151338])
