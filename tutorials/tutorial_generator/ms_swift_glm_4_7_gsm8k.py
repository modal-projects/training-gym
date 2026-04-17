"""Tutorial source for `ms_swift_glm_4_7_gsm8k` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). Function names are arbitrary. Cell order = source order.
"""

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # GLM-4.7 LoRA on GSM8K with ms-swift on Modal

    Fine-tunes GLM-4.7 with LoRA via ms-swift's Megatron backend. 4 nodes ×
    8×B200, TP=2 / EP=4 / PP=4 / CP=1. `megatron sft` is launched under
    `torchrun` on each clustered node.

    Invoke any function on the returned `app` via `modal run`, or
    interactively with `app.run()` + `.remote()`.
    """


@notebook_only
@shell("%uv pip install -q 'git+https://x-access-token:${GITHUB_TOKEN}@github.com/modal-projects/training-gym.git@joy/initial-setup'")

def _install():
    pass


@code
def _imports():
    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import BaseModelType, Model
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.ms_swift import (
        MsSwiftConfig,
        MsSwiftModalConfig,
        build_ms_swift_app,
    )
    from modal_training_gym.frameworks.ms_swift.config import HF_CACHE_PATH


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    ms-swift expects training data as a JSONL file with a `messages` field.
    Subclass `DatasetConfig`, set `prompt_data` to the output path, and
    implement `prepare()` to convert a HuggingFace dataset into that format.
    """


@code
def _define_dataset():
    class GSM8KDataset(DatasetConfig):
        def __init__(
            self,
            hf_cache_root,
            data_folder="gsm8k",
            hf_dataset="openai/gsm8k",
            split="train",
            input_col="question",
            output_col="answer",
        ):
            self._hf_cache_root = str(hf_cache_root)
            self._data_folder = data_folder
            self._hf_dataset = hf_dataset
            self._split = split
            self._input_col = input_col
            self._output_col = output_col
            self.prompt_data = (
                f"{self._hf_cache_root}/msswift-data/{data_folder}/training.jsonl"
            )

        def prepare(self):
            import json
            import os

            from datasets import load_dataset

            output_dir = f"{self._hf_cache_root}/msswift-data/{self._data_folder}"
            os.makedirs(output_dir, exist_ok=True)

            try:
                ds = load_dataset(self._hf_dataset, split=self._split)
            except ValueError:
                ds = load_dataset(self._hf_dataset, "main", split=self._split)

            out_path = f"{output_dir}/training.jsonl"
            with open(out_path, "w") as f:
                for row in ds:
                    messages = [
                        {"role": "user", "content": row[self._input_col]},
                        {"role": "assistant", "content": row[self._output_col]},
                    ]
                    f.write(json.dumps({"messages": messages}) + "\n")
            print(f"Wrote {out_path}")


@markdown
def _explain_config():
    """
    ## Define the experiment

    Subclass `MsSwiftConfig` and set class attributes. Every attribute except
    containers (`dataset`, `model`, `wandb`) and launcher-only fields
    (`environment`, `n_nodes`, `gpus_per_node`) is forwarded to `megatron sft`
    as a `--flag value` CLI arg (ms-swift uses underscore-style names and
    string-valued booleans).
    """


@code
def _define_config():
    class _MsSwift(MsSwiftConfig):
        # ── Containers ────────────────────────────────────────────────────────
        # hf_checkpoint comes from the Model; ms-swift reads architecture from
        # the HF config at train time.
        model = Model(BaseModelType.GLM_4_7)
        dataset = GSM8KDataset(HF_CACHE_PATH)
        wandb = WandbConfig(project="glm-4-7-sft")

        # ── Infrastructure ────────────────────────────────────────────────────
        n_nodes = 4
        gpus_per_node = 8

        # ── Parallelism (TP=2, EP=4, PP=4, CP=1) ──────────────────────────────
        tensor_model_parallel_size = 2
        expert_model_parallel_size = 4
        pipeline_model_parallel_size = 4
        context_parallel_size = 1
        sequence_parallel = True

        # ── Training ──────────────────────────────────────────────────────────
        # train_iters caps total Megatron iterations; set low for smoke testing.
        train_iters = 5
        num_train_epochs = 1
        lr = 1e-4
        global_batch_size = 8
        max_length = 2048

        # ── LoRA ──────────────────────────────────────────────────────────────
        tuner_type = "lora"
        lora_rank = 128
        lora_alpha = 32
        merge_lora = False


@markdown
def _build_section():
    """
    ## Build the Modal app
    """


@code
def _build_app():
    app = build_ms_swift_app(modal=MsSwiftModalConfig(gpu="B200"), swift=_MsSwift())


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
    uv run modal run tutorials/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.download_model
    uv run modal run tutorials/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.prepare_dataset
    uv run modal run tutorials/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.train
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
    import modal
    
    with modal.enable_output():
        with app.run():
            app.download_model.remote()
            # app.prepare_dataset.remote()
            # app.train.remote()
