# pyright: reportUndefinedVariable=false
"""Tutorial source for `010_agentic_harbor` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "6 × 8×H200",
    "summary": "Fork-backed multi-turn coding agents on partitioned Harbor tasks",
    "difficulty": "Advanced",
    "order": 100,
    "api_classes": [
        "DatasetConfig",
        "Qwen3_6_27B",
        "Qwen3_6_27b_Agentic_Recipe",
        "TrackioConfig",
        "TrainConfig",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Fork-backed coding agents on Harbor tasks

    This tutorial follows the standard Training Gym recipe pattern while using
    a commit-pinned Slime fork for multi-turn coding-agent rollouts.

    The three pieces specific to this workload are:
    1. deterministic train/eval subsets prepared from archived Harbor tasks;
    2. `Qwen3_6_27b_Agentic_Recipe`, which overlays the pinned Slime fork;
    3. the fork's `agentic_rl.generate.generate` rollout function.

    The recipe uses six 8×H200 nodes by default. Ensure the Modal workspace has
    multi-node access and sufficient GPU quota before launching it.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    export AGENTIC_HARBOR_DATASET_ROOT=example-org_private-harbor-tasks
    uv run tutorials/rl/010_agentic_harbor/010_agentic_harbor.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    import os

    from modal_training_gym import (
        DatasetConfig,
        Qwen3_6_27B,
        Qwen3_6_27b_Agentic_Recipe,
        TrackioConfig,
        TrainConfig,
    )


@markdown
def _partition_intro():
    """
    ## Partition the Harbor dataset

    The partition script downloads and converts the archived tasks once, then
    writes a repository-disjoint 20% `eval.jsonl` and nested,
    language-balanced `train-100.jsonl`, `train-300.jsonl`,
    `train-1000.jsonl`, and `train-full.jsonl` subsets.

    ```
    uv run python scripts/partition_harbor_dataset.py prepare \\
      --hf-repo example-org/private-harbor-tasks \\
      --dataset-key private-harbor-tasks
    ```

    The script and recipe both use the shared `slime-data` Modal volume.
    """


@code
def _settings():
    DATASET_ROOT = os.environ["AGENTIC_HARBOR_DATASET_ROOT"]
    TRAIN_SUBSET = os.environ.get("AGENTIC_TRAIN_SUBSET", "train-300")
    EVAL_SUBSETS = tuple(
        value
        for value in os.environ.get("AGENTIC_EVAL_SUBSETS", "eval").split(",")
        if value
    )
    NUM_ROLLOUT = int(os.environ.get("AGENTIC_NUM_ROLLOUT", "500"))
    EVAL_SAMPLES = int(os.environ.get("AGENTIC_EVAL_SAMPLES", "1"))
    EVAL_INTERVAL_RAW = os.environ.get("AGENTIC_EVAL_INTERVAL", "")
    EVAL_INTERVAL = int(EVAL_INTERVAL_RAW) if EVAL_INTERVAL_RAW else None
    RUN_NAME = os.environ.get("AGENTIC_RUN_NAME", "agentic-harbor")
    TRACKIO_PROJECT = os.environ.get("AGENTIC_TRACKIO_PROJECT", "agentic-harbor")


@markdown
def _dataset_intro():
    """
    ## Select subsets by filename

    `PreparedHarborSubset` resolves `train-300` directly to
    `/data/<dataset-root>/train-300.jsonl`. If the file is missing, it reports
    the partition command instead of silently creating a different dataset.
    """


@code
def _dataset():
    class PreparedHarborSubset(DatasetConfig):
        input_key = "prompt"
        label_key = "label"
        apply_chat_template = False
        output_format = "jsonl"
        writes_eval_paths = False
        always_prepare = False

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


@markdown
def _recipe_intro():
    """
    ## Configure the normal Qwen recipe subclass

    Model conversion, speculative decoding, and Qwen-specific settings remain
    inherited from `Qwen3_6_27b_Recipe`. The agentic subclass adds the pinned
    fork, non-colocated topology, Harbor environment,
    `agentic_rl.generate.generate`, and Trackio experiment tracking.

    The launcher runs one Trackio server on the Ray head and prints a dashboard
    URL that stays live for the duration of training. All Ray actors send
    metrics to that server, and its database is persisted in the
    `training-gym-trackio` Modal Volume under the training run id. No W&B key or
    account is required.

    Dataset selection is independent from rollout count and eval sampling.
    """


@code
def _build_and_run():
    data_root = f"/data/{DATASET_ROOT.replace('/', '_')}"
    recipe = Qwen3_6_27b_Agentic_Recipe(
        num_rollout=NUM_ROLLOUT,
        eval_interval=EVAL_INTERVAL,
        trackio=TrackioConfig(project=TRACKIO_PROJECT, run_name=RUN_NAME),
        eval_config={
            "defaults": {
                "n_samples_per_eval_prompt": EVAL_SAMPLES,
                "temperature": 1.0 if EVAL_SAMPLES > 1 else 0.6,
                "top_p": 1.0,
            },
            "datasets": [
                {
                    "name": subset,
                    "path": f"{data_root}/{subset}.jsonl",
                    "metadata_overrides": {"eval_dataset": subset},
                }
                for subset in EVAL_SUBSETS
            ],
        },
        save_debug_rollout_data=(
            f"/checkpoints/agentic_rollout_dumps/{RUN_NAME}/"
            "rollout_{rollout_id}.pt"
        ),
    )
    run = TrainConfig(
        model=Qwen3_6_27B(),
        dataset=train_dataset,
        recipe=recipe,
        merge_model_recipe=False,
    ).launch(prepare_inputs=True)
    print(f"Training run: {run.training_run_id}")
    print(f"Modal app: {run.modal_app_url}")


@markdown
def _mixed_subset():
    """
    ## Probe and make a mixed-reward subset

    The same tutorial is also the probe launcher. Select a train subset as eval
    data and request multiple samples. Use one rollout so Megatron's optimizer
    scheduler initializes; the initial eval dump is written before that training
    step:

    ```
    AGENTIC_TRAIN_SUBSET=train-300 \\
    AGENTIC_EVAL_SUBSETS=train-300 \\
    AGENTIC_NUM_ROLLOUT=1 \\
    AGENTIC_EVAL_SAMPLES=8 \\
    AGENTIC_EVAL_INTERVAL=5 \\
    AGENTIC_RUN_NAME=qwen3-6-27b-agentic-probe \\
      uv run tutorials/rl/010_agentic_harbor/010_agentic_harbor.py
    ```

    After the probe finishes, filter the source subset to tasks whose eight
    fully gradeable outcomes contain both successes and failures:

    ```
    uv run python scripts/partition_harbor_dataset.py \\
      --dataset-root example-org_private-harbor-tasks \\
      mixed \\
      --source train-300 \\
      --recipe qwen3-6-27b-agentic \\
      --n-samples 8 \\
      --probe-dump /checkpoints/agentic_rollout_dumps/qwen3-6-27b-agentic-probe/rollout_eval_0.pt \\
      --checkpoints-volume slime-qwen3_6_27b_agentic_recipe-checkpoints
    ```

    This writes
    `train-300-mixed-reward-qwen3-6-27b-agentic-n8.jsonl` and a JSON sidecar
    containing the source hash, recipe, checkpoint, sample count, selection
    criterion, and probe dump. `--checkpoint` defaults to `base`; pass the
    Training Gym run/checkpoint identity when probing trained weights.
    """
