"""Tutorial source for `megatron_glm_4_7_longmit128k` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). README content is pulled into markdown cells so it shows
up in both the .ipynb and (as `#` comments) in the .py.
"""

from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _title():
    """
    # Megatron: LoRA Fine-tuning GLM-4.7 (358B MoE)

    This example demonstrates how to fine-tune the GLM-4.7 358B Mixture-of-Experts
    model using NVIDIA NeMo's Megatron framework with LoRA (Low-Rank Adaptation)
    on Modal's multi-node infrastructure.
    """


@markdown
def _overview():
    """
    ## Overview

    GLM-4.7 is a 358B parameter MoE model with 92 layers and 160 experts.
    Fine-tuning a model this large requires:

    - Multi-node distributed training across 32 GPUs
    - Advanced parallelism strategies (Tensor, Pipeline, Expert, Data parallelism)
    - Memory-efficient techniques like LoRA and activation recomputation

    The training pipeline consists of three main stages:

    1. Model download and conversion to Megatron format
    2. Dataset preparation (LongMIT-128K)
    3. Multi-node distributed LoRA training
    """


@markdown
def _prereqs():
    """
    ## Prerequisites

    - Modal account with access to B200 (or H100/H200) GPUs
    - Hugging Face account with access to GLM-4.7
    - Weights & Biases account for experiment tracking

    Configure Modal secrets:

    ```bash
    modal secret create huggingface-secret HF_TOKEN=<your-token>
    modal secret create wandb-secret WANDB_API_KEY=<your-key>
    ```
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
    from modal_training_gym.frameworks.megatron import (
        DATA_DIR,
        MegatronConfig,
        MegatronFrameworkConfig,
    )


@markdown
def _explain_dataset():
    """
    ## Define the dataset

    The reference example uses **LongMIT-128K** (`donmaclean/LongMIT-128K`),
    a long-context SFT dataset. The `prepare()` method:

    - Downloads from HuggingFace
    - Builds a long-context SFT prompt from passages + question
    - Filters to `<= 128K` tokens and writes JSONL for Megatron SFT
    - Builds the `.idx.npy` / `.idx.info` index files Megatron needs for
      efficient data loading

    Megatron's `FinetuningDatasetConfig` points at a directory containing a
    `training.jsonl` file — the launcher derives the directory from
    `dataset.prompt_data`.
    """


@code
def _define_dataset():
    class LongMIT128KDataset(DatasetConfig):
        """LongMIT-128K SFT dataset prepped for Megatron's FinetuningDatasetConfig."""

        def __init__(
            self,
            data_root,
            subdir="longmit-128k",
            hf_dataset="donmaclean/LongMIT-128K",
            hf_model="zai-org/GLM-4.7",
            max_tokens=131_072,
            prep_cpu=32,
        ):
            self._data_root = str(data_root)
            self._subdir = subdir
            self._hf_dataset = hf_dataset
            self._hf_model = hf_model
            self._max_tokens = max_tokens
            self._prep_cpu = prep_cpu
            self.prompt_data = f"{self._data_root}/{subdir}/training.jsonl"

        def prepare(self):
            import glob
            import json
            import os

            from datasets import load_dataset
            from megatron.bridge.data.datasets.utils import build_index_files
            from transformers import AutoTokenizer

            out_dir = f"{self._data_root}/{self._subdir}"
            os.makedirs(out_dir, exist_ok=True)

            print(f"Downloading dataset: {self._hf_dataset}")
            ds = load_dataset(self._hf_dataset, split="train", trust_remote_code=True)
            print(f"Dataset loaded: {len(ds)} examples")

            tokenizer = AutoTokenizer.from_pretrained(
                self._hf_model, use_fast=True, trust_remote_code=True
            )

            def _format(example):
                passages = "\n".join(
                    f"Passage {i + 1}:\n{doc['content']}"
                    for i, doc in enumerate(example["all_docs"])
                )
                prompt = (
                    "Answer the question based on the given passages.\n\n"
                    f"{passages}\n\n"
                    f"Question: {example['question']}\nAnswer:"
                )
                answer = example["answer"]
                return {
                    "input": prompt,
                    "output": answer,
                    "n_tokens": len(tokenizer(prompt + answer).input_ids),
                }

            ds = ds.map(_format, remove_columns=ds.column_names, num_proc=self._prep_cpu)
            ds = ds.filter(
                lambda ex: ex["n_tokens"] <= self._max_tokens, num_proc=self._prep_cpu
            )
            print(f"Filtered to {len(ds)} examples (<= {self._max_tokens} tokens)")

            out_jsonl = f"{out_dir}/training.jsonl"
            with open(out_jsonl, "w") as f:
                for ex in ds:
                    json.dump({"input": ex["input"], "output": ex["output"]}, f)
                    f.write("\n")
            print(f"Wrote {out_jsonl}")

            # Rebuild index files (not rebuilt automatically on content change).
            for stale in glob.glob(f"{out_jsonl}.idx*"):
                os.remove(stale)
            build_index_files(
                dataset_paths=[out_jsonl],
                newline_int=10,
                workers=self._prep_cpu,
                index_mapping_dir=None,
            )


@markdown
def _explain_config():
    """
    ## Define the experiment

    Create a `MegatronFrameworkConfig(...)` and pass it into
    `MegatronConfig(dataset=..., model=..., framework_config=...)`. The
    launcher serializes this resolved config to JSON that `train_script.py`
    reads inside each container.

    ### Parallelism Strategy

    The 358B MoE model uses a combination of parallelism strategies across 32 GPUs:

    | Parallelism | Value | Description |
    |-------------|-------|-------------|
    | Tensor (TP) | 2 | Splits attention/FFN across GPUs |
    | Pipeline (PP) | 4 | Distributes layers across pipeline stages |
    | Expert (EP) | 4 | Distributes MoE experts |
    | Context (CP) | 4 | Splits long sequences |

    **Data parallelism** is derived: `DP = n_nodes * gpus_per_node / (TP * PP) = 4 * 8 / (2 * 4) = 4`.

    ### LoRA Configuration

    ```python
    LoRA(dim=128, alpha=32, dropout=0.05)
    ```

    ### Training Parameters

    | Parameter          | Value   |
    |--------------------|---------|
    | Global batch size  | 32      |
    | Micro batch size   | 1       |
    | Sequence length    | 131,072 |
    | Learning rate      | 1e-4 (cosine decay) |
    | Warmup iterations  | 50      |
    | Training iterations| 650     |
    | Checkpoint interval| 130     |

    ### Memory Optimizations

    - **Activation recomputation:** full recomputation, uniform method (tune with
      `recompute_num_layers` — 1 = checkpoint every layer, higher = less overhead).
    - **Mixed precision:** BF16 mixed precision training.
    - **Grouped GEMM:** optimized MoE computation.
    - **Sequence parallel:** enabled for memory efficiency.
    - **Operator fusion:** masked softmax, bias activation, bias dropout, RoPE fusion all on.
    - **TransformerBlock patch:** custom gradient flow fix for LoRA + recompute compatibility.
    """


@code
def _define_config():
    megatron_framework_config = MegatronFrameworkConfig(
        gpu="B200",
        n_nodes=4,
        gpus_per_node=8,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        expert_model_parallel_size=4,
        context_parallel_size=4,
        micro_batch_size=1,
        global_batch_size=32,
        seq_length=131_072,
        train_iters=650,
        lr=1e-4,
        lr_warmup_iters=50,
        save_interval=130,
        lora_dim=128,
        lora_alpha=32,
        lora_dropout=0.05,
        recompute_num_layers=1,
    )

    my_training_run = MegatronConfig(
        dataset=LongMIT128KDataset(DATA_DIR),
        model=GLM_4_7(),
        wandb=WandbConfig(project="glm47-lora"),
        framework_config=megatron_framework_config,
    )


@markdown
def _build_section():
    """
    ## Build the Modal app
    """


@code
def _build_app():
    app = my_training_run.build_app()


@markdown
def _run_section():
    """
    ## Quick Start

    Three stages, in order:

    1. **Download + convert** — pulls the 358B HF weights (~700GB) on CPU,
       then converts to Megatron `torch_dist` on a B200 (~1TB memory).
       Takes several hours; use `--detach`.
    2. **Prepare dataset** — downloads LongMIT-128K, formats it for SFT,
       builds Megatron index files.
    3. **Train** — spins up 4 nodes × 8 B200 with RDMA, runs torchrun
       training, saves periodic checkpoints, logs to WandB.
    """


@py_only
@markdown
def _run_cli():
    """
    From the CLI:

    ```bash
    # Step 1a: Download only
    uv run modal run tutorials/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.download_model

    # Step 1b: Convert only (run after download_model)
    uv run modal run tutorials/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.convert_to_megatron

    # Step 1: Download + convert (hours; use --detach)
    uv run modal run --detach tutorials/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.download_and_convert

    # Step 2: Prepare dataset
    uv run modal run tutorials/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.prepare_dataset

    # Step 3: Train (detach recommended)
    uv run modal run --detach tutorials/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.train_lora
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — open an ephemeral app and run one stage per cell.
    Use either `download_and_convert` or the explicit `download_model` +
    `convert_to_megatron` pair:
    """


@notebook_only
@code
def _invoke_download_model():
    with modal.enable_output():
        with app.run():
            app.download_model.remote()


@notebook_only
@code
def _invoke_convert_to_megatron():
    with modal.enable_output():
        with app.run():
            app.convert_to_megatron.remote()


@notebook_only
@code
def _invoke_download_and_convert():
    with modal.enable_output():
        with app.run():
            app.download_and_convert.remote()


@notebook_only
@code
def _invoke_prepare_dataset():
    with modal.enable_output():
        with app.run():
            app.prepare_dataset.remote()


@notebook_only
@code
def _invoke_train_lora():
    with modal.enable_output():
        with app.run():
            app.train_lora.remote()


@markdown
def _monitoring():
    """
    ## Monitoring

    Training metrics are logged to Weights & Biases:

    - Loss curves (LM loss, MTP loss, load balancing loss)
    - Learning rate schedule
    - Gradient norms
    - Samples processed

    You can also monitor GPU utilization via Modal's dashboard or by exec-ing
    into a running container:

    ```bash
    modal container list
    modal shell <container-id>
    nvidia-smi
    ```
    """


@markdown
def _customization():
    """
    ## Customization

    - Adjust parallelism — set `tensor_model_parallel_size` /
      `pipeline_model_parallel_size` / `expert_model_parallel_size` /
      `context_parallel_size` on `megatron_framework_config`. The product must
      divide `n_nodes * gpus_per_node` and leave a non-zero `DP`.
    - Modify LoRA hyperparameters — `lora_dim`, `lora_alpha`, `lora_dropout`.
    - Change training parameters — `global_batch_size`, `micro_batch_size`,
      `lr`, `train_iters`, `save_interval`.
    - Use a different dataset — write a new `DatasetConfig` subclass whose
      `prepare()` materializes `training.jsonl` + Megatron index files.
    - Tune memory — `recompute_num_layers` (1 = full, higher = less
      recompute overhead) or set `no_recompute = True` (likely OOMs for
      long context).

    ## Files

    | File | Description |
    |------|-------------|
    | `modal_training_gym/frameworks/megatron/config.py`       | `MegatronConfig` + `MegatronFrameworkConfig` |
    | `modal_training_gym/frameworks/megatron/config.py`       | `build_app()` on `MegatronConfig` |
    | `modal_training_gym/frameworks/megatron/train_script.py` | The torchrun target (uses `megatron.bridge`) |
    """
