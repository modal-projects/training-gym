# ---
# order: 2
# ---
#
# # Training coding agents in sandboxed environments
#
# Train [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) on
# [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2).
# The agent inspects repositories, edits code, and runs commands in
# [Harbor](https://github.com/laude-institute/harbor) sandboxes on Modal.
#
# Loop per rollout:
# 1. Start a task environment in a [Modal Sandbox](https://modal.com/docs/guide/sandbox),
#    an isolated container with its own filesystem.
# 2. Ask the model for the next action using the task and conversation so far.
# 3. Execute its shell command in the sandbox and append the output as feedback.
# 4. Repeat until the agent finishes or reaches its action or time budget.
# 5. Run the task verifier against the edited repository and use its reward
#    to train the model.
#
# The entire stack runs on Modal — model serving, tool execution,
# and the sandbox — so you control cost, latency, and data privacy.

import json

from pathlib import Path
from uuid import uuid4

import modal

from modal_training_gym import (
    DatasetConfig,
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe,
    TrackioConfig,
    TrainConfig,
)

from tutorials.coding_agent.dataset import (
    DATA_VOLUME_NAME,
    SLIME_GIT_REPOSITORY,
    SLIME_GIT_REVISION,
)

# ## Prepare sandboxed tasks
#
# `dataset.py` converts SWE-rebench into Harbor tasks and uses
# the shared splitting and sampling utilities to prepare balanced,
# repository-disjoint train/eval data. The converter grades Python tasks.
#
# ```bash
# uv run -m tutorials.coding_agent.dataset prepare
# ```
#
# Subsets live in `/data/swe_rebench_v2` on the `slime-data` volume.
# For a small check, use `prepare --limit 100`; rerun without the limit
# before selecting larger subsets.

# ## Configure training
#
# The default checks two training steps with two prompts, then evaluates once
# on four held-out tasks.
# One sample per prompt gives GRPO no advantage variance. Set `SMOKE = False`
# for learning, or `NUM_ROLLOUT = 1` for a one-step check.

SMOKE = True
DATASET_ROOT = "swe_rebench_v2"
TRAIN_SUBSET = "train-4" if SMOKE else "train-300"
EVAL_SUBSETS = ("eval-4",) if SMOKE else ("eval",)
NUM_ROLLOUT = 2 if SMOKE else 500
ROLLOUT_BATCH_SIZE = 2 if SMOKE else 32
N_SAMPLES_PER_PROMPT = 1 if SMOKE else 8
MAX_STEPS = 2 if SMOKE else 75
RUN_NAME = f"coding-agent-{uuid4().hex}"
DATA_ROOT = Path("/data") / DATASET_ROOT

# ## Load the training tasks
#
# `AgentTaskDataset` reads the prepared JSONL, preserving prompts, labels,
# and task metadata. The agent loop formats conversations, so the reader
# disables chat templating. Evaluation reads the files via `eval_config`.


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


train_dataset = AgentTaskDataset(DATA_ROOT / f"{TRAIN_SUBSET}.jsonl")

# ## Multi-turn rollouts and rewards
#
# The pinned Slime fork's `agentic_rl.generate.generate` runs the agent loop
# and computes rewards inline. It also records a `loss_mask`: model-generated
# tokens contribute to training (`1`), while task text and environment feedback
# are masked out (`0`). The model learns its actions, not the tool's output.
#
# `ASYNC_RL_REWARD_SHAPE="binary"` selects the task's pass/fail reward. GRPO
# compares samples for the same task, giving higher-reward samples a positive
# advantage. The full configuration samples eight episodes per task.
#
# See the pinned [rollout implementation](https://github.com/modal-projects/slime/blob/ba324bebdd3a3cbfc1946b58404a012ad607f38b/agentic_rl/generate.py)
# and [Harbor environment](https://github.com/modal-projects/slime/blob/ba324bebdd3a3cbfc1946b58404a012ad607f38b/agentic_rl/environment/harbor.py)
# for the interaction and verification code.
#
# ## Configure the recipe
#
# Use `Qwen3_6_27B_Recipe` with the workload overrides below. Smoke uses two
# B300 GPUs: one for Megatron training and one for SGLang inference. Full
# training uses 16 H200s for training and 32 for rollouts. `colocate=False`
# separates training and inference; sandbox commands run on CPUs.
#
# The main controls are:
# - `num_rollout`: number of rollout/training iterations.
# - `rollout_batch_size`: tasks sampled per iteration; `n_samples_per_prompt`
#   sets how many episodes to generate for each task.
# - `MAX_STEPS`: agent action budget; the episode and command timeouts below
#   also bound tool execution.
# - `eval_interval` and `save_interval`: evaluation and checkpoint frequency.
#
# Smoke evaluates once at the end and saves no checkpoints. Full evaluation
# is disabled; checkpoints save every 20 steps. Evaluation uses the same
# agent loop and verifier on the held-out tasks.
# Set up Trackio and the dashboard with `uv run training-gym setup` before
# launching to inspect metrics and trajectories.

hardware_overrides = (
    {
        "gpu_type": "B300",
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "rollout_num_gpus": 1,
        "rollout_num_gpus_per_engine": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "conversion_tensor_model_parallel_size": 1,
        "conversion_pipeline_model_parallel_size": 1,
        "decoder_last_pipeline_num_layers": None,
        "attention_backend": "unfused",
        "ref_load": "/checkpoints/Qwen3.6-27B_torch_dist_tp1pp1",
        "sglang_speculative_algorithm": None,
        "sglang_speculative_num_steps": None,
        "sglang_speculative_eagle_topk": None,
        "sglang_speculative_num_draft_tokens": None,
    }
    if SMOKE
    else {
        "gpu_type": "H200",
        "actor_num_nodes": 2,
        "actor_num_gpus_per_node": 8,
        "rollout_num_gpus": 32,
        "rollout_num_gpus_per_engine": 2,
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "context_parallel_size": 2,
        "sequence_parallel": True,
        "conversion_tensor_model_parallel_size": 4,
        "conversion_pipeline_model_parallel_size": 2,
        "decoder_last_pipeline_num_layers": 30,
        "attention_backend": "flash",
        "ref_load": "",
        "sglang_speculative_algorithm": "EAGLE",
        "sglang_speculative_num_steps": 3,
        "sglang_speculative_eagle_topk": 1,
        "sglang_speculative_num_draft_tokens": 4,
    }
)

recipe = Qwen3_6_27B_Recipe(
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
    colocate=False,
    metrics=TrackioConfig(project="coding-agent"),
    num_rollout=NUM_ROLLOUT,
    rollout_batch_size=ROLLOUT_BATCH_SIZE,
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
    global_batch_size=ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT,
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
    rollout_max_response_len=1024 if SMOKE else 8192,
    eval_max_response_len=1024 if SMOKE else 8192,
    eval_interval=NUM_ROLLOUT if SMOKE else None,
    n_samples_per_eval_prompt=1,
    save=None if SMOKE else "/checkpoints",
    save_interval=None if SMOKE else 20,
    sglang_server_concurrency=4 if SMOKE else 32,
    max_tokens_per_gpu=16384,
    log_probs_chunk_size=128,
    capture_trace=True,
    custom_rollout_log_function="agentic_rl.metrics.log_rollout_data",
    extra_config={
        "custom_generate_function_path": "agentic_rl.generate.generate",
        "agentic_max_steps": MAX_STEPS,
        "agentic_episode_timeout": 300 if SMOKE else 1800,
        "agentic_eval_timeout": 120 if SMOKE else None,
        "agentic_exec_timeout": 60 if SMOKE else 120,
        "router_policy": "consistent_hashing",
        "skip_eval_before_train": SMOKE,
    },
    eval_config={
        "defaults": {
            "n_samples_per_eval_prompt": 1,
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
    **hardware_overrides,
)

# ## Check the dataset and launch
#
# Check the subsets before allocating GPUs. The launcher handles model
# download, dataset materialization, and checkpoint conversion.

volume = modal.Volume.from_name(recipe.data_volume_name)
try:
    prepared = {Path(entry.path).name for entry in volume.listdir(DATASET_ROOT)}
except modal.exception.NotFoundError:
    prepared = set()
missing = [
    subset
    for subset in (TRAIN_SUBSET, *EVAL_SUBSETS)
    if f"{subset}.jsonl" not in prepared
]
if missing:
    raise FileNotFoundError(
        f"{recipe.data_volume_name}:/{DATASET_ROOT} is missing {missing}; "
        "run uv run -m tutorials.coding_agent.dataset prepare first"
    )

run = TrainConfig(
    model=Qwen3_6_27B(),
    dataset=train_dataset,
    recipe=recipe,
).launch()
print(f"run id: {run.training_run_id}")
print(f"Modal app: {run.modal_app_url}")
print(f"rollout dumps: /checkpoints/agentic_rollout_dumps/{RUN_NAME}/")

# Run with `uv run -m tutorials.coding_agent.main`. Open the training run in the
# dashboard to inspect tool calls, verifier outcomes, and rewards. Before
# increasing the training horizon, check that tasks are gradeable and sample
# groups contain both successes and failures; uniform rewards give GRPO no
# learning signal.
#
# ## Select tasks with informative rewards
#
# To probe a training subset, use a short run with `EVAL_SUBSETS = ("train-300",)`
# and set both `n_samples_per_eval_prompt` and the eval config's
# `n_samples_per_eval_prompt` to 8. Use full episode budgets (`SMOKE = False`)
# and `NUM_ROLLOUT = 1`, and set `eval_interval=1` to enable evaluation.
# The initial evaluation produces an eval dump before
# training. Use the printed dump directory in this command:
#
# ```bash
# uv run -m tutorials.coding_agent.dataset --dataset-root swe_rebench_v2 mixed \
#   --source train-300 --recipe qwen3-6-27b-agentic --n-samples 8 \
#   --probe-dump /checkpoints/agentic_rollout_dumps/<run-name>/rollout_eval_0.pt \
#   --checkpoints-volume slime-qwen3_6_27b_recipe-checkpoints
# ```
#
# This keeps tasks whose eight gradeable samples include both successes and
# failures. Set `TRAIN_SUBSET` to the resulting
# `train-300-mixed-reward-qwen3-6-27b-agentic-n8` subset for subsequent training.
# A sidecar records the source, checkpoint, and selection criterion; incomplete
# or non-evaluation dumps are rejected.
