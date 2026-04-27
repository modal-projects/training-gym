"""Tutorial source for `005_miles_intro` — parsed by generate_tutorial.py.

Single-node Qwen3-0.6B GRPO on GSM8K using Miles (Ray + Megatron).
Uses 8×A100 instead of H100 to cut costs while still running the full RL
stack.
"""

TUTORIAL_METADATA = {
    'framework': '`miles`',
    'cluster_shape': '1 × 8×A100',
    'summary': 'Qwen3-0.6B GRPO on GSM8K with Miles — single-node A100',
    'difficulty': 'Intermediate',
    'order': 18,
    'gpu_type': 'A100',
    'n_gpus': 8,
    'api_classes': [
        'MilesConfig', 'MilesFrameworkConfig',
        'DatasetConfig', 'Qwen3_0_6B', 'WandbConfig',
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-0.6B GRPO on GSM8K with Miles on Modal

    **What Miles is.** [Miles](https://github.com/radixark/miles) is an
    RLVR training framework built on Megatron-LM + Ray.
    `modal-training-gym`'s Miles launcher wires that stack onto a Modal
    GPU cluster, handling the image, Ray bring-up, volume mounts, and
    checkpointing.

    **What this tutorial does.** GRPO-tunes Qwen3-0.6B on
    [GSM8K](https://huggingface.co/datasets/openai/gsm8k) on 1 node ×
    8×A100. The math-correctness reward parses boxed numeric answers and
    compares them to gold labels. Colocated mode — actor and rollout
    share the same GPUs.

    **Why A100?** Qwen3-0.6B is small enough to train on A100s. This
    drops the hourly cost from ~$31.60 (8×H100) to **~$20.00** (8×A100).
    Miles requires 8 GPUs per node for its Ray cluster topology.

    **What you'll need.**
    - A Modal account with A100 access (1 node × 8 GPUs).
    - A `wandb` Modal secret holding your W&B API key.

    **What to watch.** W&B project `miles-grpo`, group
    `qwen3-0.6b-gsm8k`. Rollout reward should climb within the first
    few training steps.

    **Estimated cost.** ~$20.00/hr on 8×A100. A smoke run
    (`train_iters=5`) completes in <15 min ≈ **~$5.00**.
    See [Modal pricing](https://modal.com/pricing) for current rates.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.common.dataset import HuggingFaceDataset
    from modal_training_gym.common.models import (
        ModelTrainingConfig,
        Qwen3_0_6B,
    )
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.miles import (
        MilesConfig,
        MilesFrameworkConfig,
    )
    from modal_training_gym.frameworks.miles.config import DATA_PATH


@markdown
def _explain_model():
    """
    ## Retarget the model to A100

    `Qwen3_0_6B` ships with `training.gpu_type="H100"`. Override it to
    target A100 instead. Miles reads `gpu_type` from the model's
    training config to provision the right GPU on Modal.
    """


@code
def _define_model():
    class Qwen3_0_6B_A100(Qwen3_0_6B):
        """Qwen3-0.6B retargeted for A100 GPUs."""

        training = ModelTrainingConfig(
            gpu_type="A100",
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    Miles uses the same GSM8K data format as slime — parquet files with
    a `messages` column (chat template) and a `label` column for the
    math-correctness reward.
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


@markdown
def _explain_config():
    """
    ## Define the experiment

    `MilesFrameworkConfig` wraps Miles-specific CLI flags. The
    `recipe_args` block passes model architecture flags that Megatron
    needs — these match `Qwen3_0_6B`'s `ModelArchitecture`.

    Key settings:

    - **`n_nodes=1`** — single-node training.
    - **`colocate=True`** — actor and rollout share the same GPUs.
    - **`train_iters=5`** — smoke run; bump for real training.
    - **`global_batch_size=4`** — small batch for quick iteration.
    """


@code
def _define_config():
    base_model = Qwen3_0_6B_A100()
    arch = Qwen3_0_6B.architecture

    miles_framework = MilesFrameworkConfig(
        n_nodes=1,
        colocate=True,
        train_iters=5,
        global_batch_size=4,
        recipe_args=f"""
            --num-layers {arch.num_layers}
            --hidden-size {arch.hidden_size}
            --ffn-hidden-size {arch.ffn_hidden_size}
            --num-attention-heads {arch.num_attention_heads}
            --group-query-attention
            --num-query-groups {arch.num_query_groups}
            --kv-channels {arch.kv_channels}
            --vocab-size {arch.vocab_size}
            --normalization {arch.normalization}
            --norm-epsilon {arch.norm_epsilon}
            --disable-bias-linear
            --swiglu
            --use-rotary-position-embeddings
            --rotary-base {arch.rotary_base}
            --qk-layernorm
        """,
    )

    my_training_run = MilesConfig(
        dataset=GSM8KDataset(DATA_PATH),
        model=base_model,
        wandb=WandbConfig(project="miles-grpo", group="qwen3-0.6b-gsm8k"),
        framework_config=miles_framework,
        name="qwen3-0.6b-gsm8k-miles",
    )


@markdown
def _cost_estimate_section():
    """
    ## Cost estimate

    Run the cell below to see what this training run will cost before
    you launch it. Adjust `estimated_minutes` to match your expected
    wall-clock time.
    """


@code
def _cost_estimate():
    from modal_training_gym.common.cost import GPU_HOURLY_PRICES, estimate_cost

    gpu_type = "A100"
    n_gpus = 8
    estimated_minutes = 15  # smoke run; increase for full training

    hourly = estimate_cost(gpu_type, n_gpus, hours=1)
    smoke = estimate_cost(gpu_type, n_gpus, hours=estimated_minutes / 60)
    print(f"GPU: {n_gpus}×{gpu_type} @ ${GPU_HOURLY_PRICES[gpu_type]:.2f}/GPU/hr")
    print(f"Estimated cost ({estimated_minutes} min smoke run): ${smoke:.2f}")
    print(f"Estimated cost (1 hr full run):  ${hourly:.2f}")


@markdown
def _build_section():
    """
    ## Build and run
    """


@code
def _build_app():
    app = my_training_run.build_app()


@py_only
@markdown
def _run_cli():
    """
    From the CLI:

    ```bash
    uv run modal run tutorials/rl/005_miles_intro/005_miles_intro.py::app.download_model
    uv run modal run tutorials/rl/005_miles_intro/005_miles_intro.py::app.prepare_dataset
    uv run modal run --detach tutorials/rl/005_miles_intro/005_miles_intro.py::app.train_multi_node
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — one stage per cell:
    """


@notebook_only
@code
def _invoke_download_model():
    with modal.enable_output():
        with app.run():
            app.download_model.remote()


@notebook_only
@code
def _invoke_prepare_dataset():
    with modal.enable_output():
        with app.run():
            app.prepare_dataset.remote()


@notebook_only
@code
def _invoke_train():
    with modal.enable_output():
        with app.run():
            app.train_multi_node.remote()
