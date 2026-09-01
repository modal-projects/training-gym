from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_35b_Recipe_Long_Context(SlimeRecipe):
    """Qwen3.6-35B-A3B (MoE) on 2x8xH200 with TP2/PP2/CP1/EP2."""

    gpu_type: str = "H200"
    slime_model_script: str = "scripts/models/qwen3.5-35B-A3B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    colocate: bool = False
    rollout_num_gpus: int = 8

    # ── Parallelism ───────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 2
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    context_parallel_size: int = 2
    expert_model_parallel_size: int = 4
    expert_tensor_parallel_size: int = 1

    # ── Rollout ───────────────────────────────────────────────────────────
    num_rollout: int = 5
    rollout_batch_size: int = 16
    rollout_num_gpus_per_engine: int = 4
    rollout_max_response_len: int = 8192
    rollout_temperature: float = 1.0
    global_batch_size: int = 128
    sglang_ep_size: int | None = 4
    sglang_disable_custom_all_reduce: bool = True
    sglang_cuda_graph_bs: list[int] | None = field(
        default_factory=lambda: [1, 2, 4, 8] + list(range(16, 257, 8))
    )
    sglang_max_running_requests: int | None = 256
    # EAGLE/MTP speculative decoding disabled: the base Qwen3.6-35B-A3B ships no
    # MTP weights, and converting a randomly-initialized MTP head at any tp/pp>1
    # corrupts the torch_dist save (duplicate keys in determine_global_metadata) —
    # the same MTP/checkpoint incompatibility that forced GLM-4.7 to disable it.
    sglang_speculative_algorithm: str | None = None
    sglang_speculative_num_steps: int | None = None
    sglang_speculative_eagle_topk: int | None = None
    sglang_speculative_num_draft_tokens: int | None = None
    sglang_mamba_scheduler_strategy: str = "extra_buffer"
    mtp_num_layers: int | None = None
    enable_mtp_training: bool = False
    mtp_loss_scaling_factor: float | None = None

    # ── Training ──────────────────────────────────────────────────────────
    n_samples_per_prompt: int = 8
    max_tokens_per_gpu: int = 16384
    calculate_per_token_loss: bool = True
    moe_token_dispatcher_type: str = "flex"
    moe_enable_deepep: bool = True

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True

    # ── Attention ─────────────────────────────────────────────────────────
    attention_backend: str = "flash"

    # ── Checkpointing / eval ──────────────────────────────────────────────
    ref_load: str = "/checkpoints/Qwen3.6-35B-A3B_torch_dist_tp2pp2"
    save_interval: int = 1

    # ── Chat template ─────────────────────────────────────────────────────
    apply_chat_template_kwargs: dict | str = field(
        default_factory=lambda: {"enable_thinking": True}
    )

    # ── Environment ───────────────────────────────────────────────────────
    environment: dict = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
