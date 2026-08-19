# pyright: reportUndefinedVariable=false, reportMissingImports=false
"""Tutorial source for `007_param_sweep` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "Sweep hyperparameters across runs",
    "difficulty": "Intermediate",
    "order": 40,
    "api_classes": [
        "HuggingFaceDataset",
        "Qwen3_5_4B",
        "SlimeRecipe",
        "TrainConfig",
        "TrainingGroup",
    ],
}

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Sweeping hyperparameters with TrainingGroup

    Tuning an RL run usually means launching the *same* recipe several times
    with one or two knobs changed — learning rate, rollout temperature, KL
    coefficient — and comparing the curves. Doing that by hand is tedious and
    error-prone: you copy a `TrainConfig`, tweak a field, and hope you didn't
    fat-finger a name or leave two runs sharing a checkpoint.

    `TrainingGroup` makes this a first-class operation. You give it **one base
    `TrainConfig`** and a **`grid`** of field overrides, and it expands them
    into the cross-product of concrete runs — one independent training job per
    combination, each with its own `training_run_id` but a shared `group_id` so
    the dashboard can show them side by side.

    Three things make it safe to launch a big sweep:

    * `get_train_configs()` returns the derived `TrainConfig`s so you can see
      *exactly* what will run before spending a single GPU-second.
    * Invalid overrides — a misspelled field, or a value of the wrong type —
      raise immediately, before any run starts. A sweep never dies three
      variants deep because of a typo.
    * `launch()` starts every variant as a detached Modal run with
      `TrainingRun` handles you can inspect or wait on later.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run locally (your machine drives the Modal GPU workers):

    ```
    cd training-gym
    uv sync
    uv run tutorials/rl/007_param_sweep/007_param_sweep.py
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
    from typing import Any

    from modal_training_gym import (
        HuggingFaceDataset,
        Qwen3_5_4B,
        SlimeRecipe,
        TrainConfig,
        TrainingGroup,
    )


@markdown
def _base_intro():
    """
    ## 1. Define the base run

    Start with an ordinary `TrainConfig` — the recipe you'd launch if you
    weren't sweeping. Everything the sweep *doesn't* override is inherited from
    here, so set the fields you want held constant once. We use a short Qwen3.5-4B
    DAPO-math run as the base.
    """


@code
def _dataset():
    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_key = "prompt"
        label_key = "label"
        output_format = "jsonl"
        apply_chat_template = True
        always_prepare = True

        def load(self, split: str = "all") -> Any:
            from datasets import load_dataset

            ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
            stop = len(ds) if not self.n_rows else min(self.n_rows, len(ds))
            return ds.select(range(stop))

    train_dataset = MathDataset(n_rows=2_000)


@code
def _base():
    base = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=train_dataset,
        recipe=SlimeRecipe(
            rm_type="dapo",
            gpu_type="H100",
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            rollout_num_gpus_per_engine=1,
            num_rollout=15,
            rollout_batch_size=16,
            n_samples_per_prompt=8,
            rollout_max_response_len=8192,
            rollout_temperature=1.0,
            global_batch_size=32,
            lr=1e-6,
            advantage_estimator="grpo",
            use_kl_loss=False,
            kl_coef=0.0,
            use_dynamic_batch_size=True,
            max_tokens_per_gpu=9216,
            sglang_mem_fraction_static=0.75,
            save_interval=10,
        ),
    )


@markdown
def _group_intro():
    """
    ## 2. Wrap it in a `TrainingGroup`

    The `grid` maps **dotted field paths** to the list of values to try. Paths
    address the composed config: `recipe.lr` sets `lr` on the recipe,
    `model.model_name` would set it on the model, and so on. The group is the
    cross-product of every axis — here `2 × 2 = 4` runs.

    The grid is validated the moment you construct the group: misspell
    `recipe.lr` as `recipe.learning_rate` and you get an immediate error with a
    "did you mean" hint, not a four-hour run that crashes at the end.

    Pass a `name` to give the group a stable, human-readable id — it's slugified
    and stamped onto every run, so you can find and filter the whole sweep by it
    in the dashboard's **Group** filter. Omit it and one is autogenerated.
    """


@code
def _group():
    group = TrainingGroup(
        base=base,
        grid={
            "recipe.lr": [1e-6, 5e-6],
            "recipe.rollout_temperature": [0.8, 1.0],
        },
        name="qwen4b-lr-temp-sweep",
    )


@markdown
def _preview_intro():
    """
    ## 3. Preview the derived runs (no GPU)

    `get_train_configs()` expands and fully validates the grid, returning the
    concrete `TrainConfig`s. Inspect them before launching anything — confirm
    the values are what you expect and that each run is distinct.
    """


@code
def _preview():
    configs = group.get_train_configs()
    print(f"{len(configs)} runs in group {group.group_id}:")
    for cfg in configs:
        print(
            f"  lr={cfg.recipe.lr:<8} "
            f"temp={cfg.recipe.rollout_temperature}"
        )


@notebook_only
@markdown
def _validation_intro():
    """
    ### Validation catches mistakes up front

    A bad override fails immediately rather than mid-sweep. Both a misspelled
    field and a wrong-typed value raise before any training starts:
    """


@notebook_only
@code
def _validation_demo():
    from modal_training_gym.common.training_group import TrainingGroupError

    try:
        TrainingGroup(base=base, grid={"recipe.learning_rate": [1e-6]})
    except TrainingGroupError as err:
        print("misspelled field rejected:")
        print(f"  {err}")


@markdown
def _launch_intro():
    """
    ## 4. Launch the sweep

    ### Blocking param sweep

    `train()` runs every variant and returns the successful `TrainResult`s.
    With `max_parallel > 1` the variants run concurrently — each is a detached,
    independent Modal app — and a single failure is recorded in
    `group.failures` instead of sinking the rest of the sweep.

    Every result carries the shared `group_id`, so you can pull the whole sweep
    back together afterwards (and the dashboard groups them automatically).


    ### Background param sweep

    `launch()` starts every variant as a detached Modal run and returns a list of
    `TrainingRun` handles. Each handle has the `training_run_id`, Modal app URL,
    function-call id, and shared `group_id`.

    A single launch failure is recorded in `group.failures` instead of sinking
    the rest of the sweep.

    Call `launch.result()` to wait for a handle's trained `TrainResult`. If you
    don't need the handles, `group.train(max_parallel=...)` wraps this pattern
    and returns the successful `TrainResult`s directly.
    """


@code
def _launch():
    launches = group.launch()
    print(f"group {group.group_id}: {len(launches)} runs launched")
    for launch in launches:
        print(
            f"  {launch.training_run_id}  "
            f"app={launch.modal_app_id}  "
            f"group_id={launch.group_id}"
        )
    if group.failures:
        for overrides, err in group.failures:
            print(f"  FAILED {overrides}: {err}")

    results = []
    for launch in launches:
        result = launch.result()
        results.append(result)
        print(f"completed {result.training_run_id}  (group_id={result.group_id})")

    print(f"group {group.group_id}: {len(results)} runs completed")


@markdown
def _outro():
    """
    ## Recap

    * One base `TrainConfig` + a `grid` of dotted overrides → the cross-product
      of independent runs, each with a unique `training_run_id` and a shared
      `group_id` (set from `name`, or autogenerated).
    * `get_train_configs()` shows you the derived runs, and validation rejects
      bad fields/values *before* anything launches.
    * `train(max_parallel=...)` fans the sweep out across Modal; `group.failures`
      isolates any run that didn't make it.
    * `launch()` fans the sweep out across Modal and returns
      `TrainingRun` handles; `group.failures` isolates any run that didn't
      launch.
    * Use `launch.result()` to wait on a specific launched run, or
      `train(max_parallel=...)` when you want one blocking call that returns the
      successful `TrainResult`s.
    * Every run is tagged with the `group_id`, so the dashboard's **Group**
      filter pulls the whole sweep together for side-by-side comparison.

    Sweep any composed field, not just the recipe — `model.*` and `dataset.*`
    paths work too, so you can vary the dataset size or a model setting the same
    way. Keep the grid small to start: cost scales with the product of the axes.
    """
