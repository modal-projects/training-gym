# ---
# order: 11
# ---
#
# # Multi-turn RL for coding agents on Harbor tasks
#
# This tutorial trains [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)
# to solve [Harbor](https://github.com/laude-institute/harbor) coding tasks as a
# multi-turn agent. Every rollout is a full agent episode: the policy reads the
# task, edits and runs code in its own Modal Sandbox for up to 75 steps, and is
# graded by the task's tests. The reward is binary, so the model is rewarded
# only for tasks it actually solves.
#
# Three pieces are specific to this workload, and each has its own section
# below:
#
# 1. deterministic train/eval subsets partitioned from an archived Harbor
#    dataset, consumed by filename;
# 2. `Qwen3_6_27B_Recipe_Agentic`, which pins a Slime fork that ships the agent
#    loop and the Harbor sandbox environment;
# 3. Trackio experiment tracking, so the fork's native `rollout/*` and
#    `eval/<dataset>` charts land on a server you host on Modal.
#
# The recipe trains on six 8×H200 nodes (48 GPUs) by default, with two trainer
# nodes and four rollout nodes. A one-node smoke profile is described at the
# end for checking the plumbing before committing that much hardware.

import os
from dataclasses import replace

import modal

from modal_training_gym import (
    DatasetConfig,
    Qwen3_6_27B,
    Qwen3_6_27B_Recipe_Agentic,
    TrackioConfig,
    TrainConfig,
)

# ## Partition the Harbor dataset
#
# Harbor tasks are archived on the Hugging Face Hub. The partition script
# downloads and converts them once with the pinned fork's translator, then
# writes a repository-disjoint 20% `eval.jsonl` alongside nested,
# language-balanced `train-100.jsonl`, `train-300.jsonl`, `train-1000.jsonl`,
# and `train-full.jsonl` subsets:
#
# ```bash
# uv run scripts/partition_harbor_dataset.py prepare \
#   --hf-repo example-org/private-harbor-tasks \
#   --dataset-key private-harbor-tasks
# ```
#
# The subsets land at `/data/<dataset-root>/<subset>.jsonl` on the `slime-data`
# Modal Volume, where `<dataset-root>` defaults to the Hub repo id with `/`
# replaced by `_`. The script reads the fork pin and the volume name from the
# recipe, so what it writes is always what training mounts.

# ## Configure the run
#
# The dataset root is private to your workspace, so it is read from the
# environment rather than hard-coded, and the remaining knobs follow suit so
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

DATASET_ROOT = os.environ.get("AGENTIC_HARBOR_DATASET_ROOT", "")
TRAIN_SUBSET = os.environ.get("AGENTIC_TRAIN_SUBSET", "train-300")
EVAL_SUBSETS = tuple(
    value for value in os.environ.get("AGENTIC_EVAL_SUBSETS", "eval").split(",") if value
)
NUM_ROLLOUT = int(os.environ.get("AGENTIC_NUM_ROLLOUT", "500"))
EVAL_SAMPLES = int(os.environ.get("AGENTIC_EVAL_SAMPLES", "1"))
EVAL_INTERVAL_RAW = os.environ.get("AGENTIC_EVAL_INTERVAL", "")
EVAL_INTERVAL = int(EVAL_INTERVAL_RAW) if EVAL_INTERVAL_RAW else None
NODES = int(os.environ.get("AGENTIC_NODES", "6"))
ROLLOUT_BATCH_SIZE = int(os.environ.get("AGENTIC_ROLLOUT_BATCH_SIZE", "32"))
N_SAMPLES_PER_PROMPT = int(os.environ.get("AGENTIC_N_SAMPLES_PER_PROMPT", "8"))
MAX_STEPS = int(os.environ.get("AGENTIC_MAX_STEPS", "75"))
SMOKE = os.environ.get("AGENTIC_SMOKE", "") == "1"
RUN_NAME = os.environ.get("AGENTIC_RUN_NAME", "agentic-harbor")
TRACKIO_PROJECT = os.environ.get("AGENTIC_TRACKIO_PROJECT", "agentic-harbor")
LOAD = os.environ.get("AGENTIC_LOAD", "")

if SMOKE:
    NODES, ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, MAX_STEPS = 1, 2, 1, 2
if NODES not in (1, 6):
    raise ValueError("AGENTIC_NODES must be 6 (disaggregated) or 1 (one colocated node)")
if not DATASET_ROOT:
    raise RuntimeError(
        "AGENTIC_HARBOR_DATASET_ROOT must name the dataset root written by "
        "scripts/partition_harbor_dataset.py, e.g. example-org_private-harbor-tasks"
    )
DATA_ROOT = f"/data/{DATASET_ROOT}"

# ## Select a prepared subset by filename
#
# The recipes resolve a dataset's `hf_repo` and `hf_split` to
# `/data/<hf_repo>/<hf_split>.jsonl`, so a `DatasetConfig` that sets those two
# attributes to the dataset root and subset name points straight at a
# partitioned file. Its `prepare()` refuses to run: if the file is missing, the
# fix is to run the partition script, not to silently materialize a different
# dataset. Evaluation subsets are listed in the recipe's `eval_config` instead,
# so `writes_eval_paths` is off and no companion `eval.jsonl` is expected.


class PreparedHarborSubset(DatasetConfig):
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = False
    output_format = "jsonl"
    writes_eval_paths = False

    def __init__(self, subset: str):
        if not subset or "/" in subset or subset in {".", ".."}:
            raise ValueError(f"invalid subset name: {subset!r}")
        self.hf_repo = DATASET_ROOT
        self.hf_split = subset
        self.dataset_id = f"{DATASET_ROOT}-{subset}"
        self._validate()

    def prepare(self, path: str, eval_paths=None):
        raise FileNotFoundError(
            f"prepared Harbor subset is missing: {path}; "
            "run scripts/partition_harbor_dataset.py prepare first"
        )


train_dataset = PreparedHarborSubset(TRAIN_SUBSET)

# ## Track the run with Trackio
#
# The recipe logs to [Trackio](https://huggingface.co/docs/trackio) by default,
# and a `TrackioConfig` that only names a project resolves at launch to the
# `training-gym-trackio` server deployed in your workspace. Deploy one once with
# `TrackioConfig.deploy_to_modal(project=...)` (see the
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
        "run scripts/partition_harbor_dataset.py prepare first"
    )

# ## Launch
#
# `launch()` starts a detached Modal app and returns immediately. With
# `prepare_inputs=True` the model download and Megatron conversion run first,
# so a stale conversion also fails before the cluster is allocated. The
# conversion is cached on the recipe's checkpoints volume, so only the first
# launch pays for it.

run = TrainConfig(
    model=Qwen3_6_27B(),
    dataset=train_dataset,
    recipe=recipe,
).launch(prepare_inputs=True)
print(f"run id: {run.training_run_id}")
print(f"Modal app: {run.modal_app_url}")

# ## Run it
#
# A full run on the default 48-GPU topology:
#
# ```bash
# AGENTIC_HARBOR_DATASET_ROOT=example-org_private-harbor-tasks \
#   uv run tutorials/agentic_harbor.py
# ```
#
# A one-node learning experiment on a small subset, here the two mixed-reward
# tasks used for smoke tests, with real 75-step agents. Two prompts with eight
# samples each per step; `rollout/rewards` in Trackio and the dashboard's
# reward curve should move within a handful of steps:
#
# ```bash
# AGENTIC_HARBOR_DATASET_ROOT=example-org_private-harbor-tasks \
# AGENTIC_NODES=1 \
# AGENTIC_TRAIN_SUBSET=train-2-mixed-smoke \
# AGENTIC_EVAL_SUBSETS=eval-2-smoke \
# AGENTIC_ROLLOUT_BATCH_SIZE=2 \
# AGENTIC_N_SAMPLES_PER_PROMPT=8 \
# AGENTIC_NUM_ROLLOUT=8 \
# AGENTIC_EVAL_INTERVAL=4 \
#   uv run tutorials/agentic_harbor.py
# ```
#
# A one-node smoke test against two-row subsets, producing `rollout/*` train
# charts and separate `eval/train-2-smoke` and `eval/eval-2-smoke` charts in
# Trackio after two rollouts:
#
# ```bash
# AGENTIC_HARBOR_DATASET_ROOT=example-org_private-harbor-tasks \
# AGENTIC_SMOKE=1 \
# AGENTIC_TRAIN_SUBSET=train-2-mixed-smoke \
# AGENTIC_EVAL_SUBSETS=train-2-smoke,eval-2-smoke \
# AGENTIC_NUM_ROLLOUT=2 \
# AGENTIC_EVAL_INTERVAL=1 \
# AGENTIC_EVAL_SAMPLES=1 \
#   uv run tutorials/agentic_harbor.py
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
# AGENTIC_HARBOR_DATASET_ROOT=example-org_private-harbor-tasks \
# AGENTIC_TRAIN_SUBSET=train-300 \
# AGENTIC_EVAL_SUBSETS=train-300 \
# AGENTIC_NUM_ROLLOUT=1 \
# AGENTIC_EVAL_SAMPLES=8 \
# AGENTIC_EVAL_INTERVAL=5 \
# AGENTIC_RUN_NAME=qwen3-6-27b-agentic-probe \
#   uv run tutorials/agentic_harbor.py
# ```
#
# Once the probe finishes, keep only the tasks whose eight fully gradeable
# outcomes contain both successes and failures:
#
# ```bash
# uv run scripts/partition_harbor_dataset.py \
#   --dataset-root example-org_private-harbor-tasks \
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
