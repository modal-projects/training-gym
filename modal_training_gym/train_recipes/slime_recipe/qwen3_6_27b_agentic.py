from dataclasses import field
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from modal_training_gym.common.trackio import TrackioConfig
from modal_training_gym.train_recipes.slime_recipe.qwen3_6_27b import (
    Qwen3_6_27b_Recipe,
)


@dataclass(config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class Qwen3_6_27b_Agentic_Recipe(Qwen3_6_27b_Recipe):
    """Qwen3.6-27B with fork-backed Harbor agent rollouts on 6×8×H200."""

    slime_git_repository: str | None = "https://github.com/modal-projects/slime.git"
    slime_git_revision: str | None = "ba324bebdd3a3cbfc1946b58404a012ad607f38b"
    data_volume_name: str | None = "slime-data"
    trackio: TrackioConfig | None = field(
        default_factory=lambda: TrackioConfig(project="agentic-harbor")
    )

    gpu_type: str = "H200"
    memory: int | tuple[int, int] | None = (1024, 2 * 1024 * 1024)
    train_function_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"ephemeral_disk": 2 * 1024 * 1024}
    )
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "NCCL_RAS_ENABLE": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ASYNC_RL_TASK_ROOT": "/data",
            "SLIME_AGENT_SANDBOX_CPU": "2",
            "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
            "ASYNC_RL_REWARD_SHAPE": "binary",
        }
    )
    image_run_commands: list[str] = field(
        default_factory=lambda: [
            "apt-get update && apt-get install -y --no-install-recommends "
            "rdma-core libibverbs1 ibverbs-providers",
            "uv pip install --system modal mini-swe-agent datasets",
        ]
    )
    image_env: dict[str, str] = field(
        default_factory=lambda: {"MSWEA_SILENT_STARTUP": "1"}
    )
    app_tags: dict[str, str] = field(
        default_factory=lambda: {"agentic_rollout": "harbor"}
    )
    capture_trace: bool = True
    dashboard_auth: bool = True

    colocate: bool = False
    actor_num_nodes: int = 2
    actor_num_gpus_per_node: int = 8
    rollout_num_gpus: int | None = 32
    rollout_num_gpus_per_engine: int = 2
    sglang_server_concurrency: int = 32
    tensor_model_parallel_size: int = 4
    pipeline_model_parallel_size: int = 2
    conversion_pipeline_model_parallel_size: int | None = 2
    decoder_last_pipeline_num_layers: int | None = 30
    context_parallel_size: int = 2

    num_rollout: int = 500
    rollout_batch_size: int = 32
    rollout_max_response_len: int = 8192
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    max_tokens_per_gpu: int = 16384
    log_probs_chunk_size: int = 128
    rm_type: str | None = None

    eval_interval: int | None = None
    eval_max_response_len: int = 8192
    save_interval: int = 5
    save_debug_rollout_data: str = (
        "/checkpoints/agentic_rollout_dumps/rollout_{rollout_id}.pt"
    )
    ref_load: str = ""
    lr: float = 4e-6

    custom_rollout_log_function: str | None = "agentic_rl.metrics.log_rollout_data"
    extra_config: dict[str, Any] | None = field(
        default_factory=lambda: {
            "custom_generate_function_path": "agentic_rl.generate.generate",
            "agentic_max_steps": 75,
            "agentic_episode_timeout": 1800,
            "agentic_eval_timeout": 300,
            "agentic_exec_timeout": 120,
            "router_policy": "consistent_hashing",
        }
    )
