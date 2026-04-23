"""Tutorial source for `002_customizing_your_slime_run` — parsed by generate_tutorial.py.

Teaches users how to override SlimePreset defaults to scale up training
with more nodes, adjust parallelism, and tune data throughput.
"""

TUTORIAL_METADATA = {
    'framework': '`slime`',
    'cluster_shape': '4 × 8×H100',
    'summary': 'Customizing your SLIME run — scaling nodes, parallelism, and throughput',
    'difficulty': 'Intermediate',
    'order': 12,
    'api_classes': ['SlimeConfig', 'DatasetConfig', 'Qwen3_4B', 'WandbConfig'],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Customizing your SLIME run

    The [intro tutorial](../001_slime_intro/001_slime_intro.ipynb) runs
    Qwen3-4B GRPO on GSM8K with all defaults — one node, colocated,
    the model's built-in `SlimePreset`. This tutorial shows how to
    override those defaults when you need more from your training run.
    """


@markdown
def _why():
    """
    ## Why would you want this?

    The default `SlimePreset` for Qwen3-4B gives you a working run on
    a single 8×H100 node. That's great for prototyping — you can
    validate your reward function, check that loss moves, and iterate
    on your dataset in minutes.

    But a single-node run has limits:

    - **Speed.** GRPO generates rollouts for every prompt, scores them,
      then trains. On one node, rollout and training take turns on the
      same GPUs. A 7k-prompt dataset with 8 samples per prompt means
      56k generations *per training step*. On 8 GPUs that's slow.

    - **Batch diversity.** Larger global batches give more diverse
      advantage estimates per step, which stabilizes GRPO training.
      But you can only fit so much in 8 GPUs' worth of memory.

    - **Rollout throughput.** More rollout GPUs = more generations per
      second. Separating rollout onto dedicated GPUs (non-colocated)
      lets training and inference overlap.

    Scaling to 4 nodes (32 GPUs) addresses all three: 4× the data
    parallelism, 4× the rollout capacity, and room for much larger
    batches. The only thing that changes in your code is a few numbers
    on `SlimeConfig`.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@code
def _imports():
    from modal_training_gym.common.dataset import HuggingFaceDataset
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.slime import SlimeConfig
    from modal_training_gym.frameworks.slime.config import DATA_PATH


@markdown
def _explain_defaults():
    """
    ## What the model provides

    `Qwen3_4B` ships a `SlimePreset` with these values:

    | Field | Value | Why |
    |---|---|---|
    | `gpu_type` | `"H100"` | Qwen3-4B fits comfortably on H100 |
    | `actor_num_nodes` | `1` | 4B params → no need for multi-node |
    | `actor_num_gpus_per_node` | `8` | Standard Modal GPU node |
    | `colocate` | `True` | Actor + rollout share GPUs |
    | `tensor_model_parallel_size` | `1` | Model fits on one GPU |
    | `sequence_parallel` | `False` | Not needed at TP=1 |

    These are the **minimum viable** settings. For a quick experiment
    they're fine. For a production run you'll want to scale up.
    """


@markdown
def _explain_scaling():
    """
    ## Scaling up with more nodes

    The simplest way to speed up training: add more nodes. This gives
    you more GPUs for **data parallelism** — each node processes a
    different batch slice in parallel, then gradients are all-reduced.

    ```python
    # 1 node (8 GPUs) — default from Qwen3_4B preset
    run = SlimeConfig(model=Qwen3_4B(), ...)

    # 4 nodes (32 GPUs) — 4x data parallelism
    run = SlimeConfig(model=Qwen3_4B(), actor_num_nodes=4, ...)
    ```

    With more GPUs you'll typically also increase `global_batch_size`
    and `num_rollout` to keep each GPU busy:

    | Nodes | GPUs | Suggested `global_batch_size` | Suggested `num_rollout` |
    |---|---|---|---|
    | 1 | 8 | 16 | 50 |
    | 2 | 16 | 64 | 200 |
    | 4 | 32 | 128 | 3000 |
    """


@markdown
def _explain_colocate():
    """
    ## Colocated vs non-colocated rollout

    By default, actor (training) and rollout (SGLang inference) share
    the same GPUs (`colocate=True`). This is memory-efficient but
    means training and inference can't overlap.

    For larger runs, separating them lets training and rollout happen
    in parallel on different GPU pools:

    ```python
    # Non-colocated: 3 nodes train, 1 node rolls out
    run = SlimeConfig(
        model=Qwen3_4B(),
        actor_num_nodes=4,
        colocate=False,
        # rollout_num_gpus auto-derived from spare nodes
    )
    ```
    """


@markdown
def _explain_throughput():
    """
    ## Throughput knobs

    - **`use_dynamic_batch_size=True`** + **`max_tokens_per_gpu=9216`** —
      pack variable-length prompts into a per-GPU token budget instead
      of a fixed batch count. Better GPU utilization for datasets with
      mixed prompt lengths.

    - **`recompute_granularity="full"`** — recompute activations during
      backward pass instead of storing them. Trades compute for memory,
      letting you fit larger batches.

    - **`n_samples_per_prompt=8`** — generate more rollout samples per
      prompt for better advantage estimation (GRPO benefits from
      larger groups).
    """


@markdown
def _example():
    """
    ## Putting it together

    Here's a production-scale config that overrides the preset defaults
    for a 4-node run with larger batches and non-colocated rollout:
    """


@code
def _define_dataset():
    class GSM8KDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/gsm8k"
        input_key = "messages"
        label_key = "label"

        def __init__(self, data_root):
            super().__init__(data_root)
            test_path = self.prompt_data.replace("train.parquet", "test.parquet")
            self.eval_prompt_data = ["gsm8k", test_path]

        def prepare(self):
            from datasets import load_dataset

            super().prepare()
            test_path = self.prompt_data.replace("train.parquet", "test.parquet")
            ds = load_dataset(self.hf_repo, split="test")
            ds.to_parquet(test_path)


@code
def _define_config():
    base_model = Qwen3_4B()
    my_training_run = SlimeConfig(
        model=base_model,
        dataset=GSM8KDataset(DATA_PATH),
        wandb=WandbConfig(project="slime-grpo", group="qwen3-4b-scaled"),
        ref_load=base_model.model_name,
        # Override preset defaults for a larger run:
        actor_num_nodes=4,
        global_batch_size=128,
        num_rollout=3000,
        n_samples_per_prompt=8,
        rollout_batch_size=64,
    )


@code
def _build_app():
    app = my_training_run.build_app()


@py_only
@markdown
def _run_cli():
    """
    From the CLI:

    ```bash
    uv run modal run tutorials/rl/002_customizing_your_slime_run/002_customizing_your_slime_run.py::app.download_model
    uv run modal run tutorials/rl/002_customizing_your_slime_run/002_customizing_your_slime_run.py::app.prepare_dataset
    uv run modal run --detach tutorials/rl/002_customizing_your_slime_run/002_customizing_your_slime_run.py::app.train
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Run interactively from a notebook:
    """


@notebook_only
@code
def _invoke_download_model():
    import modal
    with app.run():
        app.download_model.remote()

@notebook_only
@code
def _invoke_prepare_dataset():
    import modal
    with app.run():
        app.prepare_dataset.remote()

@notebook_only
@code
def _invoke_train():
    import modal
    with app.run():
        app.train.remote()
