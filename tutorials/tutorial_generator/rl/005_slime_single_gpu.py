"""Tutorial source for `005_slime_single_gpu` — parsed by generate_tutorial.py.

Single-node Qwen3-0.6B GRPO on GSM8K using slime, targeting 2×A100 instead
of the usual 8×H100. Demonstrates that RL post-training is accessible on
cheaper hardware when the model is small enough.
"""

TUTORIAL_METADATA = {
    'framework': '`slime`',
    'cluster_shape': '1 × 2×A100',
    'summary': 'Qwen3-0.6B GRPO on GSM8K — 2×A100 budget setup',
    'difficulty': 'Beginner',
    'order': 8,
    'gpu_type': 'A100',
    'n_gpus': 2,
    'api_classes': ['SlimeConfig', 'DatasetConfig', 'Qwen3_0_6B', 'WandbConfig'],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-0.6B GRPO on GSM8K with slime — budget A100 setup

    **What this tutorial does.** The same GRPO training as
    [`001_slime_intro`](../001_slime_intro/001_slime_intro.ipynb), but on
    **2×A100** instead of 8×H100. This drops the hourly cost from
    ~$31.60 to **~$5.00** — a 6× saving that makes RL experimentation
    affordable for smaller budgets.

    **How it works.** Qwen3-0.6B is small enough (0.6 B params, ~1.2 GB
    in bf16) that the full RL pipeline — Megatron actor + SGLang rollout
    — fits on 2 GPUs in colocated mode. No tensor parallelism needed
    (`tensor_model_parallel_size=1`).

    **What you'll need.**
    - A Modal account with A100 access.
    - A `wandb` Modal secret holding your W&B API key.

    **What to watch.** W&B project `slime-grpo`, group
    `qwen3-0.6b-gsm8k-a100`. Rollout reward should start climbing
    within the first few training steps.

    **Estimated cost.** ~$5.00/hr on 2×A100. A smoke run
    (`train_iters=5`) completes in <10 min ≈ **~$0.83**.
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
    from modal_training_gym.common.models import Qwen3_0_6B, ModelTrainingConfig
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.slime import SlimeConfig
    from modal_training_gym.frameworks.slime.config import DATA_PATH
    from modal_training_gym.frameworks.slime.preset import SlimePreset


@markdown
def _explain_model():
    """
    ## Retarget the model to A100

    `Qwen3_0_6B` ships with a slime preset for 8×H100. Override it to
    target 2×A100 with colocated actor + rollout.
    """


@code
def _define_model():
    class Qwen3_0_6B_A100(Qwen3_0_6B):
        """Qwen3-0.6B retargeted for 2×A100."""

        slime = SlimePreset(
            gpu_type="A100",
            actor_num_nodes=1,
            actor_num_gpus_per_node=2,
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
        )


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    Same GSM8K setup as the full-scale slime intro — nothing changes on
    the data side when you shrink the cluster.
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

    Key differences from the 8×H100 setup:

    - **`global_batch_size=4`** — halved to fit the smaller GPU count.
    - **`train_iters=5`** — smoke run; bump for real training.
    - **`test_freq=-1`, `save_freq=-1`** — skip eval and checkpointing
      for the quick smoke test.
    """


@code
def _define_config():
    base_model = Qwen3_0_6B_A100()
    my_training_run = SlimeConfig(
        name="qwen3-0.6b-gsm8k-a100",
        model=base_model,
        dataset=GSM8KDataset(DATA_PATH),
        wandb=WandbConfig(project="slime-grpo", group="qwen3-0.6b-gsm8k-a100"),
        ref_load=base_model.model_name,
        global_batch_size=4,
        train_iters=5,
        test_freq=-1,
        save_freq=-1,
    )


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
    uv run modal run tutorials/rl/005_slime_single_gpu/005_slime_single_gpu.py::app.download_model
    uv run modal run tutorials/rl/005_slime_single_gpu/005_slime_single_gpu.py::app.prepare_dataset
    uv run modal run --detach tutorials/rl/005_slime_single_gpu/005_slime_single_gpu.py::app.train
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
            app.train.remote()
