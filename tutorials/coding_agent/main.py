# ---
# order: 2
# ---
#
# # Training multi-turn coding agents
#
# This tutorial trains [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) on
# [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2).
# During rollouts, the agent inspects repositories, edits code, and runs commands
# in a [Modal Sandbox](https://modal.com/docs/guide/sandboxes) via 
# [Harbor](https://docs.harborframework.com/).

import json

from pathlib import Path
from uuid import uuid4

import modal

from modal_training_gym import (
    DatasetConfig,
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe,
    TrainConfig,
)

from tutorials.coding_agent.dataset import (
    DATA_VOLUME_NAME,
    SLIME_GIT_REPOSITORY,
    SLIME_GIT_REVISION,
)

# ## Get the dataset
#
# We must first convert SWE-rebench into Harbor tasks, and split/sample
# the data to create balanced, repository-disjoint train/eval sets.
# Since this is verbose, we have a
# [separate preprocessing script](https://github.com/modal-projects/training-gym/blob/main/tutorials/coding_agent/dataset.py).
# 
# Run with:
#
# ```bash
# uv run -m tutorials.coding_agent.dataset prepare --limit 100
# uv run -m tutorials.coding_agent.dataset prepare
# ```

DATASET_ROOT = "swe_rebench_v2"
DATA_ROOT = Path("/data") / DATASET_ROOT

# ## Select tasks with mixed rewards
#
# GRPO needs successes and failures for the same task. First set `PROBE = True`
# and run this tutorial to evaluate all 300 training tasks with eight attempts
# each, followed by one training update. The initial evaluation uses base weights.
# Then filter that evaluation dump with the existing dataset command:
#
# ```bash
# uv run -m tutorials.coding_agent.dataset mixed \
#   --source train-300 --n-samples 8 \
#   --probe-dump /checkpoints/agentic_rollout_dumps/<run-name>/rollout_eval_0.pt \
#   --checkpoints-volume slime-qwen3_6_27b_recipe-checkpoints
# ```
#
# Use the printed rollout dump directory for `<run-name>`. The filter keeps tasks
# with eight gradeable attempts and at least one success and one failure.
# Set `PROBE = False` for full training on the resulting subset. Held-out
# evaluation remains unfiltered; selected training reward is not held-out accuracy.

PROBE = False
TRAIN_SUBSET = (
    "train-300" if PROBE else "train-300-mixed-reward-qwen3-6-27b-agentic-n8"
)
EVAL_SUBSETS = ("train-300",) if PROBE else ("eval",)

class AgentTaskDataset(DatasetConfig):
    def __init__(self, path: Path):
        self.path = path

    def input_key(self) -> str:
        return "prompt"

    def label_key(self) -> str:
        return "label"

    def apply_chat_template(self) -> bool:
        return False

    def rows(self):
        with self.path.open() as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)

# ## Start training
# 
# With the [Qwen3_6_27B_Recipe](https://gym.modal.dev/reference/qwen3_6_27b_recipe),
# recipe class, it's just that simple.

RUN_NAME = f"coding-agent-{uuid4().hex}"

config = TrainConfig(
    model=Qwen3_6_27B(),
    dataset=AgentTaskDataset(DATA_ROOT / f"{TRAIN_SUBSET}.jsonl"),
    recipe=Qwen3_6_27B_Recipe(
        slime_git_repository=SLIME_GIT_REPOSITORY,
        slime_git_revision=SLIME_GIT_REVISION,
        data_volume_name=DATA_VOLUME_NAME,
        memory=(1024, 2 * 1024 * 1024),
        train_function_kwargs={"ephemeral_disk": 2 * 1024 * 1024},
        environment={
            "PYTHONPATH": "/root/Megatron-LM/:/root/slime",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
            "NCCL_RAS_ENABLE": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ASYNC_RL_TASK_ROOT": "/data",
            "SLIME_AGENT_SANDBOX_CPU": "2",
            "SLIME_AGENT_SANDBOX_MEMORY_MB": "4096",
            "ASYNC_RL_REWARD_SHAPE": "binary",
        },
        image_run_commands=[
            "apt-get update && apt-get install -y --no-install-recommends "
            "rdma-core libibverbs1 ibverbs-providers",
            "uv pip install --system modal mini-swe-agent datasets",
        ],
        image_env={"MSWEA_SILENT_STARTUP": "1"},
        app_tags={"agentic_rollout": "harbor"},
        gpu_type="H200",
        colocate=False,
        actor_num_nodes=2,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=32,
        rollout_num_gpus_per_engine=2,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        context_parallel_size=2,
        sequence_parallel=True,
        conversion_tensor_model_parallel_size=4,
        conversion_pipeline_model_parallel_size=2,
        decoder_last_pipeline_num_layers=30,
        attention_backend="flash",
        ref_load="",
        sglang_speculative_algorithm="EAGLE",
        sglang_speculative_num_steps=3,
        sglang_speculative_eagle_topk=1,
        sglang_speculative_num_draft_tokens=4,
        num_rollout=1 if PROBE else 500,
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        global_batch_size=256,
        advantage_estimator="grpo",
        lr=4e-6,
        lr_decay_style="constant",
        weight_decay=0.1,
        eps_clip=0.2,
        eps_clip_high=None,
        use_kl_loss=False,
        kl_coef=0.0,
        entropy_coef=0.0,
        rollout_temperature=1.0,
        rollout_max_response_len=8192,
        eval_max_response_len=8192,
        eval_interval=5,
        n_samples_per_eval_prompt=8 if PROBE else 1,
        save=None if PROBE else "/checkpoints",
        save_interval=None if PROBE else 5,
        sglang_server_concurrency=32,
        max_tokens_per_gpu=16384,
        log_probs_chunk_size=128,
        capture_trace=True,
        custom_rollout_log_function="agentic_rl.metrics.log_rollout_data",
        extra_config={
            "custom_generate_function_path": "agentic_rl.generate.generate",
            "agentic_max_steps": 75,
            "agentic_episode_timeout": 1800,
            "agentic_eval_timeout": None,
            "agentic_exec_timeout": 120,
            "router_policy": "consistent_hashing",
            "skip_eval_before_train": False,
        },
        eval_config={
            "defaults": {
                "n_samples_per_eval_prompt": 8 if PROBE else 1,
                "temperature": 0.6,
                "top_p": 1.0,
            },
            "datasets": [
                {
                    "name": subset,
                    "path": str(DATA_ROOT / f"{subset}.jsonl"),
                    "metadata_overrides": {"eval_dataset": subset},
                }
                for subset in EVAL_SUBSETS
            ],
        },
        save_debug_rollout_data=(
            f"/checkpoints/agentic_rollout_dumps/{RUN_NAME}/rollout_{{rollout_id}}.pt"
        ),
    ),
)

volume = modal.Volume.from_name(DATA_VOLUME_NAME)
prepared = {Path(entry.path).name for entry in volume.listdir(DATASET_ROOT) if entry.size}
missing = [
    subset for subset in (TRAIN_SUBSET, *EVAL_SUBSETS)
    if f"{subset}.jsonl" not in prepared
]
if missing:
    raise FileNotFoundError(
        f"Missing or empty subsets: {missing}. Run dataset prepare, then follow "
        "the mixed-reward probe and filtering steps above before full training."
    )

print(f"rollout dumps: /checkpoints/agentic_rollout_dumps/{RUN_NAME}/")
run = config.launch()
print(f"run id: {run.training_run_id}")
