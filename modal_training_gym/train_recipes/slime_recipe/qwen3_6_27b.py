from dataclasses import field

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.train_recipes.slime_recipe.recipe import SlimeRecipe


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_27b_Recipe(SlimeRecipe):
    """Qwen3.6-27B on 4×8×H100 using Slime's Qwen3.5-27B recipe."""

    gpu_type: str = "H100"
    memory: int | tuple[int, int] | None = (128, 2_097_152)
    slime_model_script: str = "scripts/models/qwen3.5-27B.sh"
    hf_checkpoint: str = "Qwen/Qwen3.6-27B"
    ref_load: str = "/checkpoints/Qwen3.6-27B_torch_dist_tp4pp2"
    train_function_kwargs: dict[str, int] = field(
        default_factory=lambda: {"ephemeral_disk": 1_048_576}
    )

    colocate: bool = True
    actor_num_nodes: int = 4
    actor_num_gpus_per_node: int = 8

    # ── Parallelism ───────────────────────────────────────────────────────
    tensor_model_parallel_size: int = 4
    sequence_parallel: bool = True
    pipeline_model_parallel_size: int = 2
    conversion_tensor_model_parallel_size: int = 4
    conversion_pipeline_model_parallel_size: int = 2
    decoder_last_pipeline_num_layers: int | None = 30
    context_parallel_size: int = 4
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1

    # ── Rollout ───────────────────────────────────────────────────────────
    num_rollout: int = 3000
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    rollout_num_gpus_per_engine: int = 2
    rollout_max_response_len: int = 32768
    rollout_temperature: float = 1.0
    sglang_mem_fraction_static: float = 0.75
    sglang_speculative_algorithm: str | None = "EAGLE"
    sglang_speculative_num_steps: int | None = 3
    sglang_speculative_eagle_topk: int | None = 1
    sglang_speculative_num_draft_tokens: int | None = 4
    sglang_mamba_scheduler_strategy: str | None = "extra_buffer"

    # ── Training / optimizer ──────────────────────────────────────────────
    lr: float = 1e-6
    max_tokens_per_gpu: int = 8192
    calculate_per_token_loss: bool = True
    balance_data: bool = True
    rm_type: str | None = "deepscaler"
    use_kl_loss: bool = False
    optimizer_cpu_offload: bool = True
    overlap_cpu_optimizer_d2h_h2d: bool = True
    use_precision_aware_optimizer: bool = True
    attention_backend: str = "flash"
    eps_clip_high: float | None = None

    # ── Checkpointing / eval ──────────────────────────────────────────────
    megatron_to_hf_mode: str = ""
    save_interval: int = 20
    eval_interval: int | None = None
