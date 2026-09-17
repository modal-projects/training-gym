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

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import modal

from modal_training_gym import (
    DatasetConfig,
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe_Agentic,
    TrackioConfig,
    TrainConfig,
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
# The default checks one step with two prompts and four eval tasks.
# One sample per prompt gives GRPO no advantage variance. Set `SMOKE = False`
# for learning, or `NUM_ROLLOUT = 10` for a longer smoke run.

SMOKE = True
DATASET_ROOT = "swe_rebench_v2"
TRAIN_SUBSET = "train-4" if SMOKE else "train-300"
EVAL_SUBSETS = ("eval-4",) if SMOKE else ("eval",)
NUM_ROLLOUT = 1 if SMOKE else 500
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
# Both configurations use two B300 GPUs on one node: one for Megatron training
# and one for SGLang inference. `colocate=False` gives each its own GPU;
# sandbox commands run separately on CPUs.
#
# The main controls are:
# - `num_rollout`: number of rollout/training iterations.
# - `rollout_batch_size`: tasks sampled per iteration; `n_samples_per_prompt`
#   sets how many episodes to generate for each task.
# - `MAX_STEPS`: agent action budget; the episode and command timeouts below
#   also bound tool execution.
# - `eval_interval` and `save_interval`: evaluation and checkpoint frequency.
#
# Evaluation uses the same agent loop and verifier on the held-out tasks.
# Run `uv run training-gym setup` and `uv run training-gym trackio setup` before
# launching to inspect metrics and trajectories.

recipe = Qwen3_6_27B_Recipe_Agentic(
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
    eval_interval=1 if SMOKE else 5,
    n_samples_per_eval_prompt=1,
    save_interval=1 if SMOKE else 5,
    sglang_server_concurrency=4 if SMOKE else 32,
    capture_trace=not SMOKE,
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
)
recipe = replace(
    recipe,
    extra_config={
        **(recipe.extra_config or {}),
        "agentic_max_steps": MAX_STEPS,
        "agentic_episode_timeout": 300 if SMOKE else 1800,
        "agentic_eval_timeout": 120 if SMOKE else 300,
        "agentic_exec_timeout": 60 if SMOKE else 120,
    },
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
# and `NUM_ROLLOUT = 1`. The initial evaluation produces an eval dump before
# training. Use the printed dump directory in this command:
#
# ```bash
# uv run -m tutorials.coding_agent.dataset --dataset-root swe_rebench_v2 mixed \
#   --source train-300 --recipe qwen3-6-27b-agentic --n-samples 8 \
#   --probe-dump /checkpoints/agentic_rollout_dumps/<run-name>/rollout_eval_0.pt \
#   --checkpoints-volume slime-qwen3_6_27b_recipe_agentic-checkpoints
# ```
#
# This keeps tasks whose eight gradeable samples include both successes and
# failures. Set `TRAIN_SUBSET` to the resulting
# `train-300-mixed-reward-qwen3-6-27b-agentic-n8` subset for subsequent training.
# A sidecar records the source, checkpoint, and selection criterion; incomplete
# or non-evaluation dumps are rejected.
