"""Tutorial source for `ms_swift_glm_4_7_gsm8k` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). Function names are arbitrary. Cell order = source order.
"""

TUTORIAL_METADATA = {
    'framework': '`ms_swift`',
    'cluster_shape': '4 × 8×B200',
    'summary': 'GLM-4.7 LoRA SFT on GSM8K (Megatron)',
    'difficulty': 'Advanced',
    'order': 20,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # GLM-4.7 LoRA SFT on GSM8K with ms-swift on Modal

    **What ms-swift is.**
    [ms-swift](https://github.com/modelscope/ms-swift) is ModelScope's
    end-to-end fine-tuning toolkit. It supports PEFT methods (LoRA /
    QLoRA / DoRA) and full finetuning across HuggingFace and
    **Megatron-LM** backends. The Megatron backend is what makes it
    interesting at large scale — it gives you 4-D parallelism (TP / PP /
    EP / CP) for models that don't fit on one GPU under HF's data
    parallelism alone.

    **What this tutorial does.** LoRA SFT of GLM-4.7 (a large MoE model)
    on GSM8K, on 4 nodes × 8×B200 (32 GPUs). The interesting piece is
    the parallelism split: TP=2, EP=4, PP=4, CP=1 — tensor parallel
    across pairs of GPUs, 4-way expert parallel for the MoE layers,
    4-stage pipeline parallel for the transformer blocks. Under the
    hood this launches `megatron sft` via `torchrun` on each clustered
    node. For the shared primitives (`DatasetConfig`, `Model`, 3-stage
    pipeline) see [`quickstart`](../../intro/quickstart/quickstart.ipynb).

    **What you'll need.**
    - Access to Modal's multi-node training preview (4 × 8×B200).
    - `wandb` Modal secret.

    **What to watch.** W&B project `glm-4-7-sft`. Watch `train/loss`
    and `train/grad_norm`; LoRA converges quickly on GSM8K so expect
    loss to fall off within the first few hundred iters.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")

def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import GLM_4_7
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.ms_swift import (
        MsSwiftConfig,
        MsSwiftFrameworkConfig,
    )
    from modal_training_gym.frameworks.ms_swift.config import HF_CACHE_PATH


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    ms-swift reads a JSONL file where each line is a chat-format object:
    `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.
    `prepare()` converts GSM8K's `(question, answer)` columns into that
    shape and writes it under the HF cache volume so both download and
    dataset prep share the same mount.
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

    `MsSwiftFrameworkConfig` holds ms-swift-specific knobs; the launcher
    forwards them to `megatron sft` as `--flag value` args. ms-swift
    uses underscore-style names (e.g. `--tensor_model_parallel_size 2`)
    and expects booleans as strings.

    ### Parallelism (32 GPUs = 4 nodes × 8 B200)

    GLM-4.7 is a large MoE — it needs all four axes:

    | Axis | Setting | Why |
    |---|---|---|
    | Tensor (TP)   | 2 | Shard individual weight matrices across 2 GPUs |
    | Expert (EP)   | 4 | Spread MoE experts across 4 GPUs |
    | Pipeline (PP) | 4 | 4-stage pipeline over transformer blocks |
    | Context (CP)  | 1 | No sequence-dim parallelism at this context length |
    | Data (implicit) | 32/(2·4·4·1) = 1 | No data parallelism at this scale |

    `sequence_parallel=True` saves activation memory in the TP groups.

    ### LoRA

    - `tuner_type="lora"` + `lora_rank=128`, `lora_alpha=32` — higher
      rank than the usual 8–16; GLM-4.7 is large enough that a bigger
      rank pays for itself without running out of memory.
    - `merge_lora=False` — keep the LoRA weights separate at save time
      (smaller checkpoints, easier to A/B against the base).

    ### Throughput

    - `global_batch_size=8`, `max_length=2048` — GSM8K is short so we
      don't need long context; batch is small because GLM-4.7 is big.
    - `lr=1e-4` — standard LoRA LR (higher than a full-finetune LR
      because only the adapter params update).
    - `train_iters=1`, `num_train_epochs=1` — set for a quick smoke run;
      bump either for real training.
    """


@code
def _define_config():
    swift_framework_config = MsSwiftFrameworkConfig(
        gpu="B200",
        n_nodes=4,
        gpus_per_node=8,
        tensor_model_parallel_size=2,
        expert_model_parallel_size=4,
        pipeline_model_parallel_size=4,
        context_parallel_size=1,
        sequence_parallel=True,
        train_iters=1,
        num_train_epochs=1,
        lr=1e-4,
        global_batch_size=8,
        max_length=2048,
        tuner_type="lora",
        lora_rank=128,
        lora_alpha=32,
        merge_lora=False,
    )

    my_training_run = MsSwiftConfig(
        dataset=GSM8KDataset(HF_CACHE_PATH),
        model=GLM_4_7(),
        wandb=WandbConfig(project="glm-4-7-sft"),
        framework_config=swift_framework_config,
    )


@markdown
def _build_section():
    """
    ## Build and run

    `build_app()` returns a Modal app with `download_model`,
    `prepare_dataset`, and `train`. See
    [`quickstart`](../../intro/quickstart/quickstart.ipynb) for the pattern.
    """


@code
def _build_app():
    app = my_training_run.build_app()


@py_only
@markdown
def _run_cli():
    """
    From the CLI:

    ```
    uv run modal run tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.download_model
    uv run modal run tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.prepare_dataset
    uv run modal run --detach tutorials/sft/ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.py::app.train
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
