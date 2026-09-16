# ---
# order: 2
# ---
#
# # Training coding agents in sandboxed environments
#
# Train a model to solve coding tasks through multi-turn interaction with a
# sandbox: inspect a repository, edit files, run commands, and learn from the
# task's tests. Each episode gets its own [Harbor](https://github.com/laude-institute/harbor)
# environment on Modal, so tool execution is isolated and rewards come from
# executing tests against the agent's changes.
#
# This example uses [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) and
# [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2).
# The pinned Slime fork supplies the agent loop and Harbor environment; the
# tutorial configures the dataset, episode budget, training objective, and
# evaluation. The reward is binary: one for passing the task's tests, zero
# otherwise.

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
# The partition script streams SWE-rebench V2, renders each accepted row into
# a Harbor task directory, and translates it into the agent loop's input
# format. Each task includes an instruction, a container environment, and
# verification tests. The pinned converter currently grades Python tasks.
#
# Run preparation once before launching training:
#
# ```bash
# uv run scripts/partition_swe_dataset.py prepare
# ```
#
# It writes repository-disjoint train/eval splits to `/data/swe_rebench_v2`
# on the `slime-data` Modal Volume. `eval` contains approximately 20% of tasks;
# `eval-4` is its smoke subset. Nested `train-4`, `train-100`, `train-300`,
# `train-1000`, and `train-full` subsets preserve the source language mix.
# Sized subsets are written only when enough tasks are available.
#
# For a smaller preparation check, use `prepare --limit 100`. This converts
# only 100 accepted tasks, so it does not produce the larger training subsets.
# Rerun preparation without the limit before selecting those subsets.

# ## Choose the experiment
#
# Edit these constants to configure the experiment. The default proof runs
# one training step with two prompts and one sample per prompt, plus eval on
# four held-out tasks. Two agent actions and short timeouts keep it small.
# A single sample per prompt gives GRPO no within-group advantage variance:
# this checks execution, grading, checkpoints, and reporting, not learning.
#
# For a learning experiment, set `SMOKE = False`. This selects 300 training
# tasks, eight samples per prompt, a 75-action budget, and 500 rollout steps.
# After a successful one-step proof, keep `SMOKE = True` and set
# `NUM_ROLLOUT = 10` for a separate smoke run before increasing the horizon.

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

# ## Read a prepared task subset
#
# The partition script already translates Harbor tasks into the agent loop's
# JSONL format. This small dataset reader preserves those rows, including
# task paths, container images, and verifier metadata. It bypasses the model
# chat template because the agent loop formats its own conversation.
# Evaluation reads the prepared eval files directly through `eval_config`.


class PreparedTaskSubset(DatasetConfig):
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


train_dataset = PreparedTaskSubset(DATA_ROOT / f"{TRAIN_SUBSET}.jsonl")

# ## Configure multi-turn training
#
# The model recipe supplies model-specific parallelism and infrastructure:
# by default one B300 for training and a separate B300 for rollouts. The
# settings below describe this coding task: GRPO, the batch and episode
# budgets, sampling, learning rate, and checkpoint/evaluation intervals.
#
# `extra_config` replaces the recipe dictionary, so merge the episode settings
# into it to retain the fork's custom generation function and routing policy.
# Rollout dumps get a unique directory per launch to avoid overwriting other
# experiments. The dashboard renders the assistant and tool turns.
#
# Trackio hosts the fork's `rollout/*` and `eval/<subset>` metrics. Deploy it
# once with `uv run training-gym setup` before launching this tutorial.

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
# Check that every requested subset exists before allocating GPUs. The
# launcher materializes the training rows remotely on the same data volume.
# `launch()` starts a detached app and handles model download, conversion,
# dataset materialization, and training; cached model conversion is reused.

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
        "run uv run scripts/partition_swe_dataset.py prepare first"
    )

run = TrainConfig(
    model=Qwen3_6_27B(),
    dataset=train_dataset,
    recipe=recipe,
).launch()
print(f"run id: {run.training_run_id}")
print(f"Modal app: {run.modal_app_url}")
print(f"rollout dumps: /checkpoints/agentic_rollout_dumps/{RUN_NAME}/")

# Run with `uv run tutorials/coding_agent.py`. Open the training run in the
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
# uv run scripts/partition_swe_dataset.py --dataset-root swe_rebench_v2 mixed \
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
