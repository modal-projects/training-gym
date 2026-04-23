"""Tutorial source for `ms_swift_custom_hf` — parsed by generate_tutorial.py.

Demonstrates plugging a *custom* HuggingFace model into an ms-swift LoRA SFT
run by subclassing `ModelConfiguration` — no entry in the built-in model
catalog, no registry mutation, just a subclass defined inline in the
tutorial.
"""

TUTORIAL_METADATA = {
    'framework': '`ms_swift`',
    'cluster_shape': '1 × 1×H100',
    'summary': 'Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfiguration` subclass, no catalog entry',
    'difficulty': 'Beginner',
    'order': 25,
    'api_classes': [
        'MsSwiftConfig', 'MsSwiftFrameworkConfig',
        'ModelConfiguration', 'DatasetConfig', 'WandbConfig',
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Custom HuggingFace model with ms-swift LoRA SFT on Modal

    **What this tutorial teaches.** How to train a model that isn't in
    the built-in catalog. The built-in classes (`Qwen3_4B`, `GLM_4_7`,
    `Llama2_7B`, `KimiK2_5`, `Qwen3_32B`) are concrete
    `HFModelConfiguration` subclasses, but nothing stops you from
    writing your own inline — no registry, no library fork, just a
    subclass in your tutorial file.

    **What it actually runs.** LoRA SFT of
    [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
    (a 135M-param Llama-family model) on a 4-example slice of GSM8K on
    1×H100. Not a serious run — the goal is to prove end-to-end that the
    custom-model seam works, cheaply. Bump `split="train[:4]"` and
    `num_train_epochs` for a real run.

    **What to watch.** W&B project `custom-hf-smoke`. `train/loss`
    should drop within the first 1–2 steps since the dataset is tiny.
    See [`quickstart`](../../intro/quickstart/quickstart.ipynb) for the shared
    primitives; [`ms_swift_glm_4_7_gsm8k`](../ms_swift_glm_4_7_gsm8k/ms_swift_glm_4_7_gsm8k.ipynb)
    for a full-scale ms-swift run with 4-D parallelism.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import ModelConfiguration
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.ms_swift import (
        MsSwiftConfig,
        MsSwiftFrameworkConfig,
    )
    from modal_training_gym.frameworks.ms_swift.config import HF_CACHE_PATH


@markdown
def _explain_model():
    """
    ## Define the custom model

    Subclass `ModelConfiguration` with:

    - `model_name` — the HF repo id. Every framework reads this off
      your subclass to know what to tokenize/train.
    - `download_model()` — how to materialize the weights. For
      HF-hosted models this is one line: `snapshot_download(repo_id=...)`.
      (If you'd rather not write this body, subclass `HFModelConfiguration`
      instead — it implements `download_model()` for you.)

    That's the whole seam. The subclass is a regular Python class; you
    can stack architecture metadata, tokenizer overrides, or custom
    download logic on top as needed.
    """


@code
def _define_model():
    from modal_training_gym.common.models import ModelTrainingConfig

    class SmolLM2_135M(ModelConfiguration):
        model_name = "HuggingFaceTB/SmolLM2-135M"
        training = ModelTrainingConfig(
            gpu_type="H100",
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            lora_rank=8,
            lora_alpha=16,
        )

        def download_model(self) -> None:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=self.model_name)


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    ms-swift wants a JSONL file of chat messages
    (`{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`).
    The `train[:4]` slice keeps this run cheap — four examples is enough
    to prove the custom-model seam wires up end-to-end.
    """


@code
def _define_dataset():
    class TinyGSM8KDataset(DatasetConfig):
        def __init__(
            self,
            hf_cache_root,
            data_folder="gsm8k_tiny",
            hf_dataset="openai/gsm8k",
            split="train[:4]",
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

    The custom `SmolLM2_135M` plugs into `MsSwiftConfig` exactly like a
    built-in — the launcher only reads `config.model.model_name`,
    which the subclass provides. Everything is single-parallel (1
    node, 1 GPU, all parallelism axes = 1) since the model is tiny.

    **Note on `image=`.** The default ms-swift image pins Python 3.11,
    but `modal-training-gym` requires the local and remote Python to
    match for serialized functions (`serialized=True`). We pin the
    NGC PyTorch 25.01 image instead — it ships py312, which matches
    this repo's `.python-version`.
    """


@code
def _define_config():
    swift_framework_config = MsSwiftFrameworkConfig(
        image="nvcr.io/nvidia/pytorch:25.01-py3",
        gpu="H100",
        n_nodes=1,
        gpus_per_node=1,
        num_train_epochs=1,
        global_batch_size=1,
        max_length=512,
        save_interval=10,
        eval_iters=0,
    )

    my_training_run = MsSwiftConfig(
        dataset=TinyGSM8KDataset(HF_CACHE_PATH),
        model=SmolLM2_135M(),
        wandb=WandbConfig(project="custom-hf-smoke"),
        framework_config=swift_framework_config,
    )


@markdown
def _build_section():
    """
    ## Build and run

    See [`quickstart`](../../intro/quickstart/quickstart.ipynb) for the
    pattern.
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
    uv run modal run tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.download_model
    uv run modal run tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.prepare_dataset
    uv run modal run --detach tutorials/sft/ms_swift_custom_hf/ms_swift_custom_hf.py::app.train
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
