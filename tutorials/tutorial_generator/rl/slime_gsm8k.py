"""Tutorial source for `slime_gsm8k` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). Function names are arbitrary. Cell order = source order.
"""

TUTORIAL_METADATA = {
    'framework': '`slime`',
    'cluster_shape': '4 × 8×H100',
    'summary': 'Qwen3-4B GRPO on GSM8K (colocated)',
    'difficulty': 'Advanced',
    'order': 10,
    'api_classes': ['SlimeConfig', 'DatasetConfig', 'Qwen3_4B', 'WandbConfig'],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-4B GRPO on GSM8K with SLIME on Modal

    **What SLIME is.** [SLIME](https://github.com/THUDM/slime) is an RL
    post-training framework that pairs Megatron (training) with SGLang
    (rollouts) and orchestrates both with Ray. `modal-training-gym`'s
    `slime` launcher wires that stack onto a Modal multi-node cluster.

    **What this tutorial does.** GRPO-tunes Qwen3-4B against
    [GSM8K](https://huggingface.co/datasets/openai/gsm8k) on 4 nodes ×
    8×H100 with actor and rollout **colocated** on the same GPUs. GSM8K
    is the canonical target for math-RL: short prompts, short answers,
    and a deterministic correctness check. This is the "everything works
    end-to-end" reference for the `slime` framework — a medium-scale RL
    post-training run with SLIME's built-in math reward (no custom
    reward code). For a custom-reward example see
    [`slime_haiku`](../slime_haiku/slime_haiku.ipynb); for the shared
    primitives (DatasetConfig, volumes, the 3-stage pipeline) see
    [`quickstart`](../../intro/quickstart/quickstart.ipynb).

    **What you'll need.**
    - Access to Modal's multi-node training preview (4 × 8×H100).
    - A `wandb` Modal secret holding your W&B API key (the SLIME launcher
      mounts it automatically when `WandbConfig` is present).
    - Patience: multi-hour run — use `modal run --detach`.

    **What to watch.**
    - **Weights & Biases** under project `slime-grpo`, group
      `qwen3-4b-gsm8k`. Rollout reward should climb steadily; eval fires
      every 20 training steps (see `eval_interval` below) against
      GSM8K's test split.
    - **Modal dashboard** — per-node GPU utilization and live logs. On a
      healthy run you'll see SGLang warm up, then alternating rollout /
      training phases.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")

def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.slime import (
        ModalConfig,
        SlimeConfig,
    )
    from modal_training_gym.frameworks.slime.config import DATA_PATH


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    The non-obvious choices for GSM8K under SLIME:

    - `input_key="messages"` + `apply_chat_template=True` — the prompt
      column holds a list of chat messages; SLIME runs the model's chat
      template over them before tokenizing. The upstream
      [`zhuzilin/gsm8k`](https://huggingface.co/datasets/zhuzilin/gsm8k)
      mirror we load below is already in that shape.
    - `label_key="label"` — the column SLIME scores against.
    - `rm_type="math"` — selects SLIME's built-in math correctness
      reward. It parses the boxed numeric answer out of the rollout and
      compares to `label`. No custom reward code needed.
    - `rollout_shuffle=True` — matters for GRPO's group-sampling
      stability.
    """


@code
def _define_dataset():
    class GSM8KDataset(DatasetConfig):
        input_key = "messages"
        label_key = "label"
        apply_chat_template = True
        rollout_shuffle = True
        rm_type = "math"

        def __init__(self, data_path):
            self._data_path = str(data_path)
            self.prompt_data = f"{self._data_path}/gsm8k/train.parquet"
            self.eval_prompt_data = ["gsm8k", f"{self._data_path}/gsm8k/test.parquet"]

        def prepare(self):
            import os

            from datasets import load_dataset

            os.makedirs(f"{self._data_path}/gsm8k", exist_ok=True)
            ds = load_dataset("zhuzilin/gsm8k")
            ds["train"].to_parquet(f"{self._data_path}/gsm8k/train.parquet")
            ds["test"].to_parquet(f"{self._data_path}/gsm8k/test.parquet")


@markdown
def _explain_config():
    """
    ## Define the experiment

    `SlimeConfig` is a pydantic dataclass — hover over any field in
    your IDE to see its type, default, and description. Only non-default
    values need to be specified; everything else inherits sensible
    defaults.

    **Cluster**
    - `actor_num_nodes=4` — 32 H100s (4 × 8 GPUs).
    - `colocate=True` — actor and rollout share the same GPUs.

    **Throughput**
    - `use_dynamic_batch_size=True` + `max_tokens_per_gpu=9216` — pack
      prompts up to a per-GPU token budget.
    - `recompute_granularity="full"` — activation recomputation for
      memory savings.

    **RL**
    - `use_kl_loss=True` — KL divergence is computed (but `kl_loss_coef`
      defaults to 0.0, so it's tracked but not penalized).
    - `weight_decay=0.1`, `adam_beta2=0.98` — optimizer overrides.
    """


@code
def _define_config():
    base_model = Qwen3_4B()
    my_training_run = SlimeConfig(
        model=base_model,
        dataset=GSM8KDataset(DATA_PATH),
        wandb=WandbConfig(project="slime-grpo", group="qwen3-4b-gsm8k"),
        ref_load=base_model.model_name,
        actor_num_nodes=4,
        modal=ModalConfig(gpu="H100"),
    )



@markdown
def _build_section():
    """
    ## Build and run

    `build_app()` returns a Modal app with `download_model`,
    `prepare_dataset`, and `train`. (Bridge mode means there's no
    separate `convert_checkpoint` step to call — see quickstart for the
    general pattern.)
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
    uv run modal run tutorials/rl/slime_gsm8k/slime_gsm8k.py::app.download_model
    uv run modal run tutorials/rl/slime_gsm8k/slime_gsm8k.py::app.prepare_dataset
    uv run modal run --detach tutorials/rl/slime_gsm8k/slime_gsm8k.py::app.train
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
