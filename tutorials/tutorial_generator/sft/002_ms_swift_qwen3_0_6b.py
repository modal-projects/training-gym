"""Tutorial source for `002_ms_swift_qwen3_0_6b` — parsed by generate_tutorial.py.

Single-node, single-GPU LoRA SFT of Qwen3-0.6B on GSM8K using ms-swift.
Targets an A10 (24 GB) to demonstrate that training-gym works on smaller,
cheaper GPUs — not just H100s.
"""

TUTORIAL_METADATA = {
    'framework': '`ms_swift`',
    'cluster_shape': '1 × 1×A10',
    'summary': 'Qwen3-0.6B LoRA SFT on GSM8K — single A10 GPU',
    'difficulty': 'Beginner',
    'order': 15,
    'gpu_type': 'A10',
    'n_gpus': 1,
    'api_classes': [
        'MsSwiftConfig', 'MsSwiftFrameworkConfig',
        'Qwen3_0_6B', 'DatasetConfig', 'WandbConfig',
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Qwen3-0.6B LoRA SFT on GSM8K with ms-swift on a single A10

    **What this tutorial does.** LoRA SFT of
    [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on
    [GSM8K](https://huggingface.co/datasets/openai/gsm8k) using a single
    NVIDIA A10 GPU (24 GB VRAM). This is the cheapest way to run a
    real fine-tuning job with `modal-training-gym` — the A10 costs
    ~$1.10/hr on Modal, so a 5-minute smoke run is under $0.10.

    **Why A10?** Qwen3-0.6B is small enough (≈1.2 GB in bf16) that LoRA
    SFT fits comfortably in 24 GB. No tensor or pipeline parallelism
    needed — everything runs on one GPU.

    **What you'll need.**
    - A Modal account (no multi-node access required).
    - A `wandb` Modal secret holding your W&B API key.

    **What to watch.** W&B project `qwen3-0.6b-sft-a10`.
    `train/loss` should drop within the first few steps.

    **Estimated cost.** ~$1.10/hr on a single A10. A smoke run
    (`num_train_epochs=1`, 4 examples) completes in <5 min ≈ **~$0.09**.
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
    from modal_training_gym.frameworks.ms_swift import (
        MsSwiftConfig,
        MsSwiftFrameworkConfig,
    )
    from modal_training_gym.frameworks.ms_swift.config import HF_CACHE_PATH


@markdown
def _explain_model():
    """
    ## Retarget the model to A10

    `Qwen3_0_6B` ships with `training.gpu_type="H100"` by default.
    Subclass it and override `training` to target the A10 instead.
    The architecture and download logic stay the same — only the
    infrastructure hint changes.
    """


@code
def _define_model():
    class Qwen3_0_6B_A10(Qwen3_0_6B):
        """Qwen3-0.6B retargeted for a single A10 GPU."""

        training = ModelTrainingConfig(
            gpu_type="A10",
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            lora_rank=8,
            lora_alpha=16,
        )


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    ms-swift reads JSONL chat messages. `HuggingFaceDataset` handles the
    download + format conversion automatically — just point it at a HF
    repo.  We slice to 4 examples for the smoke run; remove `hf_split`
    for a real training run.
    """


@code
def _define_dataset():
    class GSM8KDataset(HuggingFaceDataset):
        hf_repo = "openai/gsm8k"
        hf_config = "main"
        hf_split = "train[:4]"
        output_format = "jsonl"
        input_column = "question"
        output_column = "answer"


@markdown
def _explain_config():
    """
    ## Define the experiment

    Single node, single GPU, no parallelism. The NGC PyTorch image is
    used instead of the default ModelScope image because it ships
    Python 3.12, matching this repo's `.python-version` pin (required
    for `serialized=True` Modal functions).
    """


@code
def _define_config():
    swift_framework_config = MsSwiftFrameworkConfig(
        image="nvcr.io/nvidia/pytorch:25.01-py3",
        n_nodes=1,
        gpus_per_node=1,
        num_train_epochs=1,
        global_batch_size=1,
        max_length=512,
        save_interval=10,
        eval_iters=0,
    )

    my_training_run = MsSwiftConfig(
        dataset=GSM8KDataset(HF_CACHE_PATH),
        model=Qwen3_0_6B_A10(),
        wandb=WandbConfig(project="qwen3-0.6b-sft-a10"),
        framework_config=swift_framework_config,
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

    gpu_type = "A10"
    n_gpus = 1
    estimated_minutes = 5  # smoke run; increase for full training

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

    ```
    uv run modal run tutorials/sft/002_ms_swift_qwen3_0_6b/002_ms_swift_qwen3_0_6b.py::app.download_model
    uv run modal run tutorials/sft/002_ms_swift_qwen3_0_6b/002_ms_swift_qwen3_0_6b.py::app.prepare_dataset
    uv run modal run --detach tutorials/sft/002_ms_swift_qwen3_0_6b/002_ms_swift_qwen3_0_6b.py::app.train
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
