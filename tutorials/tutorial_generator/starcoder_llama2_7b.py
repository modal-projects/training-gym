"""Tutorial source for `starcoder_llama2_7b` — parsed by generate_tutorial.py.

README content (from the reference starcoder example) is pulled into markdown
cells so it shows up in both the .ipynb and (as `#` comments) in the .py.
"""

TUTORIAL_METADATA = {
    'framework': '`torchrun` + `hf_accelerate`',
    'cluster_shape': '2 × 8×H100',
    'summary': 'Llama-2-7B SFT on Go + Rust (FSDP)',
    'order': 50,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _title():
    """
    # StarCoder: Fine-tuning Llama 2 for Go and Rust Generation

    This example demonstrates how to fine-tune Meta's Llama-2-7B model to
    generate high-quality Go and Rust code. While Llama 2 is a powerful
    general-purpose language model, it can struggle with generating
    syntactically correct and idiomatic code in specific programming
    languages. This training setup aims to enhance its coding capabilities
    specifically for Go and Rust.
    """


@markdown
def _overview():
    """
    ## Overview

    The training pipeline consists of three main stages:

    1. Dataset ingestion from the StarCoder dataset
    2. Multi-node distributed training using PyTorch FSDP
    3. Model evaluation on coding tasks

    Unlike the other framework examples, `torchrun` and `hf_accelerate`
    aren't tied to a specific training framework — they just spawn a
    user-provided Python script under `torchrun` / `accelerate launch`
    across the Modal cluster. We demo both launchers below with the same
    training script so you can compare.
    """


@markdown
def _prereqs():
    """
    ## Prerequisites

    - Modal account with access to H100 GPUs
    - Hugging Face account with access to Llama 2 model family
    - Weights & Biases account (optional, for experiment tracking)
    """


@notebook_only
@shell("! pip install -q git+https://github.com/modal-projects/training-gym.git@joy/initial-setup")
def _install():
    pass


@code
def _imports():
    import modal

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.models import Llama2_7B
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.hf_accelerate import (
        AccelerateConfig,
        AccelerateFrameworkConfig,
    )
    from modal_training_gym.frameworks.torchrun import (
        DATASET_MOUNT_PATH,
        TorchrunConfig,
        TorchrunFrameworkConfig,
    )


@markdown
def _dataset_section():
    """
    ## 1. Dataset Preparation

    Download and prepare the StarCoder dataset (Go + Rust arrow files).
    The `prepare()` method on `StarcoderGoRustDataset` below does what the
    reference `download_dataset.py::ingest_dataset` does: list files from
    `bigcode/starcoderdata`, filter out incompatible shards, and
    `snapshot_download` them into the data volume.

    - Downloads code samples from the StarCoder dataset
    - Processes and validates the data
    - Stores it in a Modal volume for training
    """


@code
def _define_dataset():
    class StarcoderGoRustDataset(DatasetConfig):
        """StarCoder: Go + Rust shards from `bigcode/starcoderdata`."""

        def __init__(self, data_path):
            self._data_path = str(data_path)
            # The train script globs `go/**/*.arrow` + `rust/**/*.arrow`
            # under `--data_dir`, so we point prompt_data at the root.
            self.prompt_data = self._data_path

        def prepare(self):
            import os
            from datasets import load_dataset
            from huggingface_hub import HfApi

            api = HfApi()
            all_paths = api.list_repo_files(
                repo_id="bigcode/starcoderdata", repo_type="dataset"
            )
            # Only Go and Rust parquet shards.
            excluded = (
                "jupyter-scripts-dedup-filtered",
                "jupyter-structured-clean-dedup",
                "github-issues-filtered-structured",
                "git-commits-cleaned",
            )
            file_paths = [
                p
                for p in all_paths
                if ".parquet" in p
                and any(p.startswith(f"{lang}/") for lang in ("go", "rust"))
                and not any(x in p for x in excluded)
            ]
            print(f"Downloading {len(file_paths)} shards…")
            for fp in file_paths:
                save_path = f"{self._data_path}/{fp}"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                ds = load_dataset("bigcode/starcoderdata", data_files=fp)
                ds.save_to_disk(save_path)
            print(f"Saved {len(file_paths)} shards to {self._data_path}")


@markdown
def _script_section():
    """
    ## 2. Training Script

    The training script is the same for both launchers — supervised
    fine-tuning of Llama-2-7B on packed Go + Rust sequences via TRL's
    `SFTTrainer` and PyTorch FSDP (`full_shard auto_wrap`, activation
    checkpointing on). We define it as a single Python string; the
    framework's `upload_script` writes it to a scripts volume that the
    train function then mounts.
    """


@code
def _define_script():
    import textwrap

    # Indented to match the function body so `textwrap.dedent` can strip
    # the common leading whitespace uniformly.
    TRAIN_SCRIPT = textwrap.dedent(r'''
        import argparse
        import os
        from pathlib import Path

        import datasets
        import torch
        import torch.distributed as dist
        from datasets import load_dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
        )
        from trl import SFTConfig, SFTTrainer

        MODEL_NAME = "meta-llama/Llama-2-7b-hf"


        def download_llama2(cache_dir):
            token = os.environ["HF_TOKEN"]
            tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=token, cache_dir=cache_dir)
            tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype="auto", token=token, cache_dir=cache_dir,
            )
            model.config.use_cache = False  # KV cache breaks activation checkpointing
            return model, tok


        def build_packed_ds(data_dir, tokenizer, buffer_size, block):
            eos_id = tokenizer.eos_token_id

            def gen():
                files = []
                for lang_dir in ["go", "rust"]:
                    files.extend(str(f) for f in (data_dir / lang_dir).glob("**/*.arrow"))
                print(f"Found {len(files)} arrow files")
                ds = load_dataset("arrow", data_files=files, split="train", streaming=True)
                ds = ds.shuffle(buffer_size=buffer_size, seed=44)

                buf = []
                for rec in ds:
                    buf.extend(
                        tokenizer(rec["content"], add_special_tokens=False)["input_ids"]
                        + [eos_id]
                    )
                    while len(buf) >= block:
                        yield {"input_ids": buf[:block], "attention_mask": [1] * block}
                        del buf[:block]

            return datasets.IterableDataset.from_generator(gen)


        def parse_args():
            p = argparse.ArgumentParser()
            p.add_argument("--data_dir", required=True)
            p.add_argument("--output_dir", required=True)
            p.add_argument("--model_cache_dir", default=None)
            p.add_argument("--epochs", type=int, default=2)
            p.add_argument("--batch_per_device", type=int, default=16)
            p.add_argument("--grad_accum", type=int, default=2)
            p.add_argument("--block_size", type=int, default=4096)
            return p.parse_args()


        def main():
            args = parse_args()

            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl",
                    device_id=torch.device(f"cuda:{os.environ['LOCAL_RANK']}"),
                )

            model, tokenizer = download_llama2(args.model_cache_dir)
            train_ds = build_packed_ds(
                Path(args.data_dir), tokenizer, buffer_size=20_000, block=args.block_size,
            )
            collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)

            cfg = SFTConfig(
                output_dir=args.output_dir,
                seed=1234,
                per_device_train_batch_size=args.batch_per_device,
                gradient_accumulation_steps=args.grad_accum,
                learning_rate=8e-5,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                bf16=True,
                max_seq_length=args.block_size,
                save_steps=125,
                logging_steps=1,
                report_to="wandb" if os.environ.get("WANDB_PROJECT") else "none",
                run_name=os.environ.get("WANDB_RUN_NAME"),
                fsdp="full_shard auto_wrap",
                fsdp_config={
                    "activation_checkpointing": True,
                    "forward_prefetch": False,
                    "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
                },
                num_train_epochs=args.epochs,
                max_steps=1,
            )

            SFTTrainer(
                model=model,
                train_dataset=train_ds,
                args=cfg,
                data_collator=collator,
            ).train()

            if dist.is_initialized():
                dist.destroy_process_group()


        if __name__ == "__main__":
            main()
    ''').lstrip("\n")


@markdown
def _training_config_section():
    """
    ## 3. Training Config

    Shared settings for both launchers — matches the reference:

    | Parameter                | Value                   |
    |--------------------------|-------------------------|
    | Global batch size        | 2048                    |
    | Per-device batch size    | 16                      |
    | Gradient accumulation    | 2                       |
    | Learning rate            | 8e-5 (cosine decay)     |
    | Context length           | 4096 tokens             |
    | Epochs                   | 2                       |
    | Nodes × GPUs             | 2 × 8 H100              |

    This command:

    - Launches a cluster of 2 nodes with 8 H100 GPUs each
    - Uses PyTorch FSDP (Fully Sharded Data Parallel) for efficient
      distributed training
    - Automatically configures RDMA for high-speed inter-node communication
    - Saves checkpoints periodically to a Modal volume
    """


@code
def _shared_settings():
    _MODEL = Llama2_7B()
    _DATASET = StarcoderGoRustDataset(DATASET_MOUNT_PATH)
    _WANDB = WandbConfig(project="bigcode-starcoderdata-training")

    _SHARED_SCRIPT_ARGS = [
        "--epochs", "1",
        "--batch_per_device", "1",
        "--grad_accum", "1",
        "--block_size", "512",
        "--model_cache_dir", "/model/model_cache",
    ]


@markdown
def _launcher_a_section():
    """
    ## 4a. torchrun launcher

    Launches the training script via `torchrun` — the stock PyTorch
    distributed launcher. Directly invokes
    `torch.distributed.run.run(...)` on every node.
    """


@code
def _torchrun_app():
    torchrun_framework_config = TorchrunFrameworkConfig(
        gpu="H100",
        n_nodes=2,
        gpus_per_node=8,
        train_script_source=TRAIN_SCRIPT,
        script_args=_SHARED_SCRIPT_ARGS,
    )

    torchrun_run = TorchrunConfig(
        dataset=_DATASET,
        model=_MODEL,
        wandb=_WANDB,
        framework_config=torchrun_framework_config,
    )

    torchrun_app = torchrun_run.build_app()


@markdown
def _launcher_b_section():
    """
    ## 4b. HuggingFace Accelerate launcher

    Same script, launched via `accelerate launch` — HF Accelerate adds
    mixed-precision setup (`--mixed_precision bf16`) and some other
    niceties on top of torchrun's spawn semantics.
    """


@code
def _accelerate_app():
    accelerate_framework_config = AccelerateFrameworkConfig(
        gpu="H100",
        n_nodes=2,
        gpus_per_node=8,
        train_script_source=TRAIN_SCRIPT,
        script_args=_SHARED_SCRIPT_ARGS,
        mixed_precision="bf16",
    )

    accelerate_run = AccelerateConfig(
        dataset=_DATASET,
        model=_MODEL,
        wandb=_WANDB,
        framework_config=accelerate_framework_config,
    )

    accelerate_app = accelerate_run.build_app()


@markdown
def _run_section():
    """
    ## 5. Run it

    Each app has the same three functions:

    - `download_dataset` — runs `StarcoderGoRustDataset.prepare()`.
    - `upload_script`    — writes `TRAIN_SCRIPT` into the scripts volume.
    - `train`            — clustered multi-node, launches the uploaded script.

    Both apps use their own volumes (`<app_name>-data`, `-model`,
    `-scripts`), so running one doesn't affect the other.

    Key training parameters (configurable on the framework config objects
    above):

    - Global batch size: 2048
    - Per-device batch size: 16
    - Gradient accumulation steps: 2
    - Learning rate: 8e-5 with cosine decay
    - Context length: 4096 tokens
    """


@py_only
@markdown
def _run_cli():
    """
    From the CLI (pick one launcher, then run the three stages in order):

    ```bash
    # ── torchrun ─────────────────────────────────────────────────────────
    uv run modal run tutorials/starcoder_llama2_7b/starcoder_llama2_7b.py::torchrun_app.download_dataset
    uv run modal run tutorials/starcoder_llama2_7b/starcoder_llama2_7b.py::torchrun_app.upload_script
    uv run modal run --detach tutorials/starcoder_llama2_7b/starcoder_llama2_7b.py::torchrun_app.train

    # ── accelerate ────────────────────────────────────────────────────────
    uv run modal run tutorials/starcoder_llama2_7b/starcoder_llama2_7b.py::accelerate_app.download_dataset
    uv run modal run tutorials/starcoder_llama2_7b/starcoder_llama2_7b.py::accelerate_app.upload_script
    uv run modal run --detach tutorials/starcoder_llama2_7b/starcoder_llama2_7b.py::accelerate_app.train
    ```
    """


@notebook_only
@markdown
def _run_torchrun_section():
    """
    ### Interactive — torchrun
    """


@notebook_only
@code
def _invoke_torchrun_download_dataset():
    with torchrun_app.run():
        torchrun_app.download_dataset.remote()


@notebook_only
@code
def _invoke_torchrun_upload_script():
    with torchrun_app.run():
        torchrun_app.upload_script.remote()


@notebook_only
@code
def _invoke_torchrun_train():
    with torchrun_app.run():
        torchrun_app.train.remote()


@notebook_only
@markdown
def _run_accelerate_section():
    """
    ### Interactive — accelerate
    """


@notebook_only
@code
def _invoke_accelerate_download_dataset():
    with accelerate_app.run():
        accelerate_app.download_dataset.remote()


@notebook_only
@code
def _invoke_accelerate_upload_script():
    with accelerate_app.run():
        accelerate_app.upload_script.remote()


@notebook_only
@code
def _invoke_accelerate_train():
    with accelerate_app.run():
        accelerate_app.train.remote()


@markdown
def _monitoring_section():
    """
    ## Performance Monitoring

    If you've configured Weights & Biases:

    - Training metrics are logged in real-time
    - You can monitor loss, learning rate, and GPU utilization
    - Compare different training runs and hyperparameters
    """


@markdown
def _scaling_section():
    """
    ## Scaling

    Sample consumption scales with the number of nodes and GPUs. Scaling
    is sublinear but can be improved by increasing the global batch size
    and tuning FSDP configurations.

    | Nodes | GPUs | Samples per minute | Tokens per minute |
    |-------|------|--------------------|-------------------|
    | 2     | 8    | 2,898              | 6,151,645         |
    | 4     | 8    | 4,981              | 10,625,570        |
    | 8     | 8    | 7,675              | 16,357,785        |
    """


@markdown
def _customization_section():
    """
    ## Customization

    - Adjust the number of nodes and GPUs on
      `torchrun_framework_config` / `accelerate_framework_config`
      (`n_nodes`, `gpus_per_node`).
    - Change training hyperparameters inside `TRAIN_SCRIPT`.
    - Add new evaluation prompts by writing a new framework function.
    - Configure data preprocessing in `StarcoderGoRustDataset.prepare()`.
    """
