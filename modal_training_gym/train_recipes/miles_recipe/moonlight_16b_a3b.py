from dataclasses import field
from typing import ClassVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.models import ModelConfig, Moonlight_16B_A3B_Instruct
from modal_training_gym.train_recipes.miles_recipe.recipe import MilesRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Moonlight_16B_A3B_Recipe(MilesRecipe):
    """Moonlight-16B-A3B on 1x8xH100, following Miles' DAPO recipe."""

    model_config_class: ClassVar[type[ModelConfig]] = Moonlight_16B_A3B_Instruct

    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "CONVERT_KEEP_PP1": "1",
        }
    )
    ref_load: str = "/checkpoints/Moonlight-16B-A3B-Instruct_torch_dist"
    megatron_to_hf_mode: str = ""

    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    use_miles_router: bool = True

    num_rollout: int = 3000
    rollout_batch_size: int = 128
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 4096
    rollout_temperature: float = 1.0
    over_sampling_batch_size: int | None = 256
    dynamic_sampling_filter_path: str | None = (
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    )
    num_steps_per_rollout: int | None = 4
    global_batch_size: int | None = None
    balance_data: bool = True

    eval_interval: int | None = 20
    n_samples_per_eval_prompt: int = 8
    eval_max_response_len: int = 4096
    eval_top_p: float = 1.0

    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int | None = 1
    expert_model_parallel_size: int | None = 8
    expert_tensor_parallel_size: int | None = 1

    recompute_granularity: str | None = "full"
    recompute_method: str | None = "uniform"
    recompute_num_layers: int | None = 1
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 8192

    advantage_estimator: str = "grpo"
    rm_type: str | None = "math"
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    eps_clip: float = 0.2
    eps_clip_high: float = 0.28

    optimizer: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    rollout_num_gpus_per_engine: int = 8
    sglang_mem_fraction_static: float = 0.7
    sglang_cuda_graph_bs: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 8))
    )

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    accumulate_allreduce_grads_in_fp32: bool = True
    attention_softmax_in_fp32: bool = True
    attention_backend: str | None = None
    moe_enable_deepep: bool = True
    moe_token_dispatcher_type: str = "flex"

    save_interval: int = 20
