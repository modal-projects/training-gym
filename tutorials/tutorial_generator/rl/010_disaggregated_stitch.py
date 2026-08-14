# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `010_disaggregated_stitch` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`miles` + `stitch`",
    "cluster_shape": "1 × 8×B200 + B200 pool",
    "summary": "Disaggregated DAPO-math RL with sparse weight deltas",
    "difficulty": "Advanced",
    "order": 60,
    "api_classes": [
        "Qwen3_30B",
        "Qwen3_30B_A3B_Stitch_Recipe",
        "HuggingFaceDataset",
        "StitchRecipe",
        "StitchTrainConfig",
        "StitchServeConfig",
        "TrainConfig",
        "WandbConfig",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Disaggregated RL: training and rollouts on separate GPUs

    Every other RL tutorial here is *colocated*: the trainer and the SGLang
    rollout engines share the same GPUs, and a weight update is an NCCL
    broadcast inside one job. That is simple, and it is also why the trainer
    sits idle while rollouts generate, why the rollout pool can only be as large
    as the training cluster, and why one bad engine takes the run down.

    **stitch** splits the two halves apart:

    - The **trainer** is a plain miles job. It owns no rollout engines.
    - The **rollout pool** is a Modal Flash service of SGLang replicas, which
      autoscales independently and serves the run over one gateway URL.
    - After each step the trainer writes a **sparse weight delta** to a shared
      "bulletin board" Volume and advances a pointer. Each replica notices the
      new version and applies it **in place**, without restarting and without
      dropping traffic.

    Two consequences worth naming, since they are the reason to pay the
    complexity:

    1. **Rollout capacity is elastic.** Sampling is throughput-bound and
       training is memory-bound; disaggregating lets you scale replicas to the
       rollout batch instead of buying trainer GPUs to get sampling throughput.
    2. **A weight update is bytes on a Volume, not a collective.** Only the
       parameters that actually changed are published (xor-encoded,
       checksummed), so sync cost tracks the *update*, not the model size — the
       thing that makes MoE weight sync expensive.

    The cost is a new correctness question: which weights served a given
    rollout? stitch answers it by stamping every rollout request with a weight
    version, so a replica that has not caught up returns a retryable 409 rather
    than quietly sampling from stale weights.

    This tutorial trains **Qwen3-30B-A3B** with GRPO on DAPO-math-17k, in NVFP4,
    as a port of stitch's own `miles_disagg/configs/qwen3_30b_a3b_nvfp4_46`
    cookbook config.
    """


@markdown
def _warning():
    """
    <div class="admonition warning">

    This is the most expensive tutorial in the repo: 8×B200 for the trainer, a
    B200 rollout pool on top, and a one-off NVFP4 conversion of the base
    checkpoint before the first step. Read it as the reference for the
    disaggregated path; run it when you actually want the run.

    </div>
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run tutorials/rl/010_disaggregated_stitch/010_disaggregated_stitch.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        HuggingFaceDataset,
        Qwen3_30B,
        Qwen3_30B_A3B_Stitch_Recipe,
        TrainConfig,
        WandbConfig,
    )


@markdown
def _dataset_intro():
    """
    ## Dataset

    The same competition-math set as the DAPO tutorial:
    [`zhuzilin/dapo-math-17k`](https://huggingface.co/datasets/zhuzilin/dapo-math-17k).
    Each row holds a chat-formatted `prompt` and an integer `label`, and the
    reward comes from miles' built-in deepscaler math scorer, so there is no
    reward function to write here.
    """


@code
def _dataset():
    class DAPOMath(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_column = "prompt"
        output_column = "label"
        output_format = "jsonl"
        apply_chat_template = True


@markdown
def _recipe_intro():
    """
    ## The recipe: two halves

    A `StitchRecipe` is deliberately not one flat pile of flags. It is the two
    sides of the run, and they are validated against each other when you
    construct it:

    ```python
    StitchRecipe(
        train=StitchTrainConfig(...),  # a miles trainer that publishes deltas
        serve=StitchServeConfig(...),  # the Flash pool that applies them
    )
    ```

    `StitchTrainConfig` is a `MilesRecipe`, so every miles flag you already know
    applies. What it pins is the disaggregation contract: `rollout_num_gpus=0`
    (the trainer owns no engines), `update_weight_transfer_mode="disk-delta"`,
    and the publish/request hooks.

    `StitchServeConfig` wraps an `SglangRecipe` for the engine arguments and
    adds what only a weight-syncing pool needs: how many replicas to keep warm,
    whether a version is applied `in_place` or after draining, and where the
    bulletin board is mounted.

    The values that *must* agree across the halves — the delta encoding, the
    bulletin path, the byte-exact served baseline, the pool's tensor-parallel
    degree — are derived and checked at construction, so a mismatch fails on
    your laptop instead of twenty minutes into a Modal run.

    `Qwen3_30B_A3B_Stitch_Recipe` is that pairing, already filled in for this
    model: TP4/EP8 with a CPU-offloaded optimizer on the trainer, NVFP4 on the
    routed experts with the last 7 layers kept in BF16, and single-B200 replicas
    serving the quantized baseline.
    """


@code
def _recipe():
    recipe = Qwen3_30B_A3B_Stitch_Recipe(
        wandb=WandbConfig(project="training-gym", group="stitch-qwen3-30b-a3b-nvfp4"),
    )
    print(f"trainer: {recipe.train.actor_num_nodes} × "
          f"{recipe.train.actor_num_gpus_per_node}×{recipe.train.gpu_type}")
    print(f"pool:    {recipe.serve.min_containers}+ × "
          f"{recipe.serve.gpus_per_replica}×{recipe.serve.gpu}, "
          f"commit_mode={recipe.serve.commit_mode}")


@markdown
def _quantization_intro():
    """
    ## Why there is a checkpoint-preparation step

    In a quantized run the trainer and the pool must agree on the *bytes* of the
    starting checkpoint, because a sparse delta is defined against it. So the
    app has a step the colocated tutorials do not: `prepare_checkpoints`
    materializes the trainer's BF16 masters and converts the pool's NVFP4
    baseline with the same quantizer the trainer exports with.

    You do not call it yourself — `train()` runs the model download, the dataset
    prep, and `prepare_checkpoints` before the run starts, exactly like the
    colocated path.
    """


@markdown
def _watch():
    """
    ## What to watch during the run

    Beyond the usual reward curve, the disaggregated path has its own health
    signals, all reported per replica by the pool's `/server_info`:

    - **`stage_s`** — fetching and materializing a published version. It scales
      with the *size of the delta*, so a slow creep over hundreds of steps means
      updates are getting denser, not that the pool is degrading.
    - **`commit_s`** — swapping the staged weights into the live engine. In
      `in_place` mode this is the only moment the engine is not sampling.
    - **the served version per replica** — every replica should march
      monotonically to the newest published version. A replica stuck behind is
      why you would see 409s: the gate is doing its job, and the trainer retries.

    If you are comparing against the colocated path, the number that matters is
    not raw step time but trainer idle time: the pool keeps sampling while the
    trainer is in its optimizer step.
    """


@markdown
def _train_intro():
    """
    ## Train

    One call, the same as every other recipe in the gym: it brings up the Flash
    pool alongside the trainer, so there is no separate deploy step.
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3_30B(),
        dataset=DAPOMath(),
        recipe=recipe,
    )
    result = training_run.train()
    print(f"Checkpoints: {result.checkpoint_dir}")
