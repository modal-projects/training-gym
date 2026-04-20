"""Tutorial source for `slime_gsm8k` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). Function names are arbitrary. Cell order = source order.
"""

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-4B GRPO on GSM8K with SLIME on Modal

    Trains Qwen3-4B with GRPO on the GSM8K math dataset using
    [SLIME](https://github.com/slimerl/slime) on Modal. 4 nodes × 8×H200,
    actor and rollout colocated.

    Invoke any function on the returned `app` via `modal run`, or
    interactively with `app.run()` + `.remote()`.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")

def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import BaseModelType, Model
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.slime import (
        ModalConfig,
        SlimeConfig,
        build_slime_app,
    )
    from modal_training_gym.frameworks.slime.config import DATA_PATH


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    Subclass `DatasetConfig` with the path + interpretation fields this
    experiment uses, and implement `prepare()` to materialize the data
    into the shared data volume.
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
            from datasets import load_dataset

            ds = load_dataset("zhuzilin/gsm8k")
            ds["train"].to_parquet(f"{self._data_path}/gsm8k/train.parquet")
            ds["test"].to_parquet(f"{self._data_path}/gsm8k/test.parquet")


@markdown
def _explain_config():
    """
    ## Define the experiment

    Subclass `SlimeConfig` and set class attributes. Every attribute except
    `environment`, `async_mode`, and `slime_model_script` is forwarded to SLIME
    automatically (`field_name` → `--field-name`).
    """


@code
def _define_config():
    class _Slime(SlimeConfig):
        # ── Model, dataset, logging ───────────────────────────────────────────
        # Architecture + hf_checkpoint are inferred from the BaseModelType.
        model = Model(BaseModelType.Qwen3_4B)
        dataset = GSM8KDataset(DATA_PATH)
        wandb = WandbConfig(project="slime-grpo", group="qwen3-4b-gsm8k")

        # ── Checkpointing ─────────────────────────────────────────────────────
        ref_load = model.hf_checkpoint  # bridge mode: ref = base checkpoint
        megatron_to_hf_mode = "bridge"

        # ── Infrastructure ────────────────────────────────────────────────────
        actor_num_nodes = 4
        actor_num_gpus_per_node = 8
        colocate = True

        # ── Rollout ───────────────────────────────────────────────────────────
        num_rollout = 2
        rollout_batch_size = 64
        rollout_max_response_len = 8192
        rollout_temperature = 1.0
        rollout_num_gpus_per_engine = 1
        sglang_mem_fraction_static = 0.7
        n_samples_per_prompt = 8
        global_batch_size = 128
        use_fault_tolerance = True

        # ── Eval ──────────────────────────────────────────────────────────────
        eval_interval = 20
        n_samples_per_eval_prompt = 4
        eval_max_response_len = 16384
        eval_top_p = 1.0

        # ── Training ──────────────────────────────────────────────────────────
        tensor_model_parallel_size = 1
        sequence_parallel = False
        use_dynamic_batch_size = True
        max_tokens_per_gpu = 9216
        recompute_granularity = "full"
        recompute_method = "uniform"
        recompute_num_layers = 1
        attention_dropout = 0.0
        hidden_dropout = 0.0
        accumulate_allreduce_grads_in_fp32 = True
        attention_softmax_in_fp32 = True

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer = "adam"
        lr = 1e-6
        lr_decay_style = "constant"
        weight_decay = 0.1
        adam_beta1 = 0.9
        adam_beta2 = 0.98

        # ── Algorithm ─────────────────────────────────────────────────────────
        advantage_estimator = "grpo"
        eps_clip = 0.2
        eps_clip_high = 0.28
        use_kl_loss = True
        kl_loss_coef = 0.0
        kl_loss_type = "low_var_kl"
        entropy_coef = 0.0



@markdown
def _build_section():
    """
    ## Build the Modal app
    """


@code
def _build_app():
    app = build_slime_app(modal=ModalConfig(gpu="H200"), slime=_Slime())


@markdown
def _run_section():
    """
    ## Run it
    """


@py_only
@markdown
def _run_cli():
    """
    From the CLI:

    ```
    uv run modal run tutorials/slime_gsm8k/slime_gsm8k/slime_gsm8k.py::app.download_model
    uv run modal run tutorials/slime_gsm8k/slime_gsm8k/slime_gsm8k.py::app.prepare_dataset
    uv run modal run tutorials/slime_gsm8k/slime_gsm8k/slime_gsm8k.py::app.train
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — open an ephemeral app and call functions remotely:
    """

@notebook_only
@code
def _invoke():
    with modal.enable_output():
        with app.run():
            app.download_model.remote()

@notebook_only
@code
def _invoke():
    with modal.enable_output():
        with app.run():
            app.train.remote()
