# ---
# order: 9
# ---
#
# # Multi-turn RL for coding agents on SWE-rebench tasks
#
# This tutorial trains [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
# to solve [SWE-rebench V2](https://huggingface.co/datasets/nebius/SWE-rebench-V2)
# coding tasks, rendered as [Harbor](https://github.com/laude-institute/harbor)
# tasks, as a multi-turn agent. Every rollout is a full agent episode: the policy reads the
# task, edits and runs code in its own Modal Sandbox for up to 75 steps, and is
# graded by the task's tests. The reward is binary, so the model is rewarded
# only for tasks it actually solves.
#
# Three pieces are specific to this workload, and each has its own section
# below:
#
# 1. deterministic train/eval subsets partitioned from SWE-rebench V2,
#    consumed by filename;
# 2. `Qwen3_6_27B_Recipe_Agentic`, which pins a Slime fork that ships the agent
#    loop and the Harbor sandbox environment;
# 3. Trackio experiment tracking, so the fork's native `rollout/*` and
#    `eval/<dataset>` charts land on a server you host on Modal.
#
# The tutorial defaults to a one-step smoke on one 8×H200 node. Set
# `AGENTIC_SMOKE=0` for six 8×H200 nodes (48 GPUs), with two trainer
# nodes and four rollout nodes; both profiles are described below.

import json
import os
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

# ## Partition the dataset
#
# The partition script streams SWE-rebench V2 from the Hugging Face Hub,
# renders each row into a Harbor task directory with the pinned fork's
# converter, then writes a repository-disjoint 20% `eval.jsonl` alongside
# a smoke-test `eval-4` and nested, language-balanced `train-4`,
# `train-100`, `train-300`, `train-1000`, `train-full` subsets:
#
# ```bash
# uv run scripts/partition_swe_dataset.py prepare
# ```
#
# The subsets land at `/data/swe_rebench_v2/<subset>.jsonl` on the `slime-data`
# Modal Volume, next to the rendered task directories. The script reads the
# fork pin and the volume name from the recipe, so what it writes is always
# what training mounts.

# ## Configure the run
#
# The dataset root defaults to `swe_rebench_v2` and can be overridden through
# the environment. The remaining knobs follow suit so
# the same file can launch a full run, a one-node experiment, a smoke test, or
# a probe. The dataset class below is shipped to remote workers by value, and
# those workers may import this module after launcher-only variables are gone,
# so everything is read at module level with defaults.
#
# `AGENTIC_NODES` is 6 for the recipe's disaggregated topology or 1 for a
# single colocated 8×H200 node. The batch shape and the agent step budget are
# separate knobs so a one-node run can still use realistic episodes.
# `AGENTIC_SMOKE=1` pins all of them to the cheapest shape that exercises the
# plumbing.

SMOKE = os.environ.get("AGENTIC_SMOKE", "1") == "1"
DATASET_ROOT = os.environ.get("AGENTIC_HARBOR_DATASET_ROOT", "swe_rebench_v2")
TRAIN_SUBSET = os.environ.get("AGENTIC_TRAIN_SUBSET", "train-4" if SMOKE else "train-300")
EVAL_SUBSETS = tuple(
    value
    for value in os.environ.get("AGENTIC_EVAL_SUBSETS", "eval-4" if SMOKE else "eval").split(",")
    if value
)
NUM_ROLLOUT = int(os.environ.get("AGENTIC_NUM_ROLLOUT", "1" if SMOKE else "500"))
EVAL_SAMPLES = int(os.environ.get("AGENTIC_EVAL_SAMPLES", "1"))
EVAL_INTERVAL_RAW = os.environ.get("AGENTIC_EVAL_INTERVAL", "1" if SMOKE else "")
EVAL_INTERVAL = int(EVAL_INTERVAL_RAW) if EVAL_INTERVAL_RAW else None
NODES = int(os.environ.get("AGENTIC_NODES", "6"))
ROLLOUT_BATCH_SIZE = int(os.environ.get("AGENTIC_ROLLOUT_BATCH_SIZE", "32"))
N_SAMPLES_PER_PROMPT = int(os.environ.get("AGENTIC_N_SAMPLES_PER_PROMPT", "8"))
MAX_STEPS = int(os.environ.get("AGENTIC_MAX_STEPS", "75"))
RUN_NAME = os.environ.get("AGENTIC_RUN_NAME") or f"agentic-harbor-{uuid4().hex}"
TRACKIO_PROJECT = os.environ.get("AGENTIC_TRACKIO_PROJECT", "agentic-harbor")
LOAD = os.environ.get("AGENTIC_LOAD", "")

if SMOKE:
    NODES, ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, MAX_STEPS = 1, 2, 1, 2
    EVAL_SAMPLES = 1
if NODES not in (1, 6):
    raise ValueError("AGENTIC_NODES must be 6 (disaggregated) or 1 (one colocated node)")
for name in (DATASET_ROOT, TRAIN_SUBSET, *EVAL_SUBSETS, RUN_NAME):
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"expected a single directory or subset name, got {name!r}")
DATA_ROOT = f"/data/{DATASET_ROOT}"

# ## Select a prepared subset by filename
#
# The dataset reads an already partitioned JSONL file from the data volume.
# The launcher materializes these rows for training using the DatasetConfig
# API. Evaluation reads the prepared files directly through `eval_config`.


class PreparedHarborSubset(DatasetConfig):
    def __init__(self, subset: str):
        if not subset or "/" in subset or "\\" in subset or subset in {".", ".."}:
            raise ValueError(f"invalid subset name: {subset!r}")
        self.path = Path(DATA_ROOT) / f"{subset}.jsonl"

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


train_dataset = PreparedHarborSubset(TRAIN_SUBSET)

# ## Track the run with Trackio
#
# The recipe logs to [Trackio](https://huggingface.co/docs/trackio) by default,
# and a `TrackioConfig` that only names a project resolves at launch to the
# `training-gym-trackio` server deployed in your workspace. Deploy one once with
# `training-gym setup` (see the
# [metrics guide](https://gym.modal.dev/guides/tools/metric)); the launch fails
# fast if no server exists rather than logging to a database that dies with the
# training container. The fork's native `rollout/*` train charts and one
# `eval/<subset>` chart per evaluation dataset appear there under the Training
# Gym run id.

metrics = TrackioConfig(project=TRACKIO_PROJECT)

# ## Configure the recipe
#
# `Qwen3_6_27B_Recipe_Agentic` inherits model conversion, parallelism, and
# speculative decoding from `Qwen3_6_27B_Recipe`, and adds what the agent
# workload needs: the commit-pinned Slime fork, a disaggregated topology
# (2 trainer nodes + 32 rollout GPUs), the Harbor sandbox environment, and the
# fork's `agentic_rl.generate.generate` rollout function.
#
# Evaluation is driven by `eval_config`: each listed subset is scored
# separately and reported as `eval/<subset>`. `save_debug_rollout_data` dumps
# every rollout under a per-run directory, which the probe workflow at the end
# of this tutorial reads back. The global batch is always one optimizer step
# over every sample of the rollout. `extra_config` replaces the recipe's
# dictionary rather than merging into it, so the agent step budget is layered
# onto the recipe's own values to keep the rollout function and timeouts.

recipe = Qwen3_6_27B_Recipe_Agentic(
    num_rollout=NUM_ROLLOUT,
    eval_interval=EVAL_INTERVAL,
    load=LOAD,
    metrics=metrics,
    rollout_batch_size=ROLLOUT_BATCH_SIZE,
    n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
    global_batch_size=ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT,
    eval_config={
        "defaults": {
            "n_samples_per_eval_prompt": EVAL_SAMPLES,
            "temperature": 1.0 if EVAL_SAMPLES > 1 else 0.6,
            "top_p": 1.0,
        },
        "datasets": [
            {
                "name": subset,
                "path": f"{DATA_ROOT}/{subset}.jsonl",
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
    extra_config={**(recipe.extra_config or {}), "agentic_max_steps": MAX_STEPS},
)

# ## Run on one node
#
# `AGENTIC_NODES=1` colocates the actor and four rollout engines on a single
# 8×H200 node, and context parallelism drops to 1 so TP4 × PP2 × CP1 fills
# exactly eight GPUs. Pipeline parallelism stays at 2, so the cached checkpoint
# conversion remains valid. Colocation parks the rollout engines in host memory
# while the actor trains, using SGLang's memory saver, which does not work
# under PyTorch's `expandable_segments` allocator setting, so that setting is
# cleared here. Size the batch to the node: two prompts with eight samples each
# is 16 episodes per step, which keeps GRPO's within-group advantage meaningful
# while a full 75-step agent budget stays affordable.
#
# `AGENTIC_SMOKE=1` goes further: two-step agents with short timeouts, one
# sample per prompt, short responses, and tracing off. With binary rewards and
# a single sample there is no advantage variance, so a smoke checks that
# rollouts, grading, and metric routing work; it is not a learning experiment.

if NODES == 1:
    recipe = replace(
        recipe,
        actor_num_nodes=1,
        rollout_num_gpus=8,
        colocate=True,
        context_parallel_size=1,
        environment={**recipe.environment, "PYTORCH_CUDA_ALLOC_CONF": ""},
    )

if SMOKE:
    recipe = replace(
        recipe,
        n_samples_per_eval_prompt=1,
        rollout_max_response_len=1024,
        eval_max_response_len=1024,
        sglang_server_concurrency=4,
        capture_trace=False,
        extra_config={
            **(recipe.extra_config or {}),
            "agentic_episode_timeout": 300,
            "agentic_eval_timeout": 120,
            "agentic_exec_timeout": 60,
        },
    )

print(f"training and rollout gpus colocated: {recipe.colocate}")
print(f"rollout dumps: /checkpoints/agentic_rollout_dumps/{RUN_NAME}/")
print(f"nodes: {recipe.total_nodes}, gpus: {recipe.gpu_allocation.total_gpus}")
print(
    f"parallelism: tp={recipe.tensor_model_parallel_size}, "
    f"pp={recipe.pipeline_model_parallel_size}, cp={recipe.context_parallel_size}"
)
print(
    f"episodes per step: {recipe.rollout_batch_size} prompts x "
    f"{recipe.n_samples_per_prompt} samples, up to {MAX_STEPS} agent steps each"
)

# ## Check the subsets before allocating the cluster
#
# Dataset preparation runs inside the training function, after the cluster is
# up, and the training function retries on failure. A missing subset would
# therefore allocate the full 48-GPU cluster several times before the file
# error surfaced. Listing the dataset root on the data volume from the
# launching shell costs nothing and catches a typo in the subset names first.

data_volume = modal.Volume.from_name(recipe.data_volume_name)
try:
    prepared = {os.path.basename(entry.path) for entry in data_volume.listdir(DATASET_ROOT)}
except modal.exception.NotFoundError:
    prepared = set()
missing = [
    subset
    for subset in (TRAIN_SUBSET, *EVAL_SUBSETS)
    if f"{subset}.jsonl" not in prepared
]
if missing:
    raise FileNotFoundError(
        f"{recipe.data_volume_name}:/{DATASET_ROOT} has no "
        f"{', '.join(f'{subset}.jsonl' for subset in missing)}; "
        "run scripts/partition_swe_dataset.py prepare first"
    )

# ## Launch
#
# `launch()` starts a detached Modal app and handles model download, dataset
# preparation, and training. Model conversion is cached on the recipe's
# checkpoints volume, so only the first launch pays for it.

run = TrainConfig(
    model=Qwen3_6_27B(),
    dataset=train_dataset,
    recipe=recipe,
).launch()
print(f"run id: {run.training_run_id}")
print(f"Modal app: {run.modal_app_url}")

# ## Run it
#
# A full run on the recipe's 48-GPU topology (the default command runs a smoke):
#
# ```bash
# AGENTIC_SMOKE=0 uv run tutorials/coding_agent.py
# ```
#
# The same topology on a small subset, here the four-task `train-4` split, for
# a handful of steps. Slime fills a rollout batch from at most one pass over
# the subset plus the start of the next, so keep the rollout batch no larger
# than twice the subset: four prompts from four tasks gives each task one
# 8-sample GRPO group per step, and `rollout/rewards` in Trackio and the
# dashboard's reward curve should move within a few steps:
#
# ```bash
# AGENTIC_SMOKE=0 AGENTIC_TRAIN_SUBSET=train-4 \
# AGENTIC_EVAL_SUBSETS=eval-4 \
# AGENTIC_ROLLOUT_BATCH_SIZE=4 \
# AGENTIC_NUM_ROLLOUT=8 \
# AGENTIC_EVAL_INTERVAL=4 \
#   uv run tutorials/coding_agent.py
# ```
#
# Add `AGENTIC_NODES=1 AGENTIC_ROLLOUT_BATCH_SIZE=2` to run the same experiment
# colocated on a single node when the full cluster is not available.
#
# A one-node smoke test against the four-task splits, producing `rollout/*`
# train charts and separate `eval/train-4` and `eval/eval-4` charts in Trackio
# after two rollouts:
#
# ```bash
# AGENTIC_SMOKE=1 \
# AGENTIC_TRAIN_SUBSET=train-4 \
# AGENTIC_EVAL_SUBSETS=train-4,eval-4 \
# AGENTIC_NUM_ROLLOUT=2 \
# AGENTIC_EVAL_INTERVAL=1 \
# AGENTIC_EVAL_SAMPLES=1 \
#   uv run tutorials/coding_agent.py
# ```
#
# ## Probe for a mixed-reward subset
#
# GRPO learns nothing from a prompt whose samples all succeed or all fail, so
# it pays to train on tasks the base model solves only sometimes. The same
# script is the probe launcher: point evaluation at a train subset, ask for
# eight samples per task, and run one rollout so Megatron's optimizer
# scheduler initializes. The initial eval dump is written before that training
# step:
#
# ```bash
# AGENTIC_SMOKE=0 AGENTIC_TRAIN_SUBSET=train-300 \
# AGENTIC_EVAL_SUBSETS=train-300 \
# AGENTIC_NUM_ROLLOUT=1 \
# AGENTIC_EVAL_SAMPLES=8 \
# AGENTIC_EVAL_INTERVAL=5 \
# AGENTIC_RUN_NAME=qwen3-6-27b-agentic-probe \
#   uv run tutorials/coding_agent.py
# ```
#
# Once the probe finishes, keep only the tasks whose eight fully gradeable
# outcomes contain both successes and failures. `mixed` reads that eval dump
# and requires it to cover every task in `--source` with a fixed sample count;
# a training dump or a truncated eval is rejected.
#
# ```bash
# uv run scripts/partition_swe_dataset.py \
#   --dataset-root swe_rebench_v2 \
#   mixed \
#   --source train-300 \
#   --recipe qwen3-6-27b-agentic \
#   --n-samples 8 \
#   --probe-dump /checkpoints/agentic_rollout_dumps/qwen3-6-27b-agentic-probe/rollout_eval_0.pt \
#   --checkpoints-volume slime-qwen3_6_27b_recipe_agentic-checkpoints
# ```
#
# This writes `train-300-mixed-reward-qwen3-6-27b-agentic-n8.jsonl` next to the
# other subsets, plus a JSON sidecar recording the source hash, recipe,
# checkpoint, sample count, selection criterion, and probe dump. Pass it as
# `AGENTIC_TRAIN_SUBSET` to train on it. `--checkpoint` defaults to `base`;
# pass the Training Gym run and checkpoint identity when probing trained
# weights.
