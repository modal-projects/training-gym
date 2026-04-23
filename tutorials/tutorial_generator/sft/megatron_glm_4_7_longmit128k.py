"""Tutorial source for `megatron_glm_4_7_longmit128k` — parsed by generate_tutorial.py.

Each `@markdown`-decorated function contributes a markdown cell (its
docstring); each `@code`-decorated function contributes a code cell (its
body, dedented). README content is pulled into markdown cells so it shows
up in both the .ipynb and (as `#` comments) in the .py.
"""

TUTORIAL_METADATA = {
    'framework': '`megatron`',
    'cluster_shape': '4 × 8×B200',
    'summary': 'GLM-4.7 LoRA on LongMIT-128K (NeMo bridge)',
    'difficulty': 'Advanced',
    'order': 30,
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # GLM-4.7 long-context LoRA with Megatron on Modal

    **What the `megatron` framework is.** A thin Modal launcher over
    NVIDIA's [NeMo `megatron.bridge`](https://github.com/NVIDIA/NeMo)
    stack — the training script runs inside `nemo/25.11` containers and
    uses `megatron.bridge` as the HF-interop layer. The launcher
    serializes your `MegatronConfig` to JSON, mounts it inside each
    container, and a `train_script.py` inside the framework package
    consumes it under `torchrun`.

    **What makes this tutorial interesting.** GLM-4.7 is a **358B-param
    MoE** with 92 layers and 160 experts. Two things you don't see in
    smaller runs:

    - **4-D parallelism.** TP=2, PP=4, EP=4, CP=4 — all four axes on, with
      data parallelism derived as the leftover (32 GPUs / (2·4·4·4)
      doesn't work cleanly; see the table below for the actual math).
      Context parallelism matters here because the sequences are 128K.
    - **128K context.** LongMIT-128K is a long-context SFT set (passages
      + question + answer). The `prepare()` step tokenizes with the
      GLM-4.7 tokenizer and filters to examples ≤ 131,072 tokens.

    For the shared primitives (`DatasetConfig`, `Model`, 3-stage
    pipeline) see [`quickstart`](../../intro/quickstart/quickstart.ipynb).

    **What you'll need.**
    - Access to Modal's multi-node training preview (4 × 8×B200).
    - `huggingface-secret` Modal secret (GLM-4.7 is a gated repo).
    - `wandb-secret` Modal secret.
    - Several hours — the model download alone is ~700GB, and the
      HF → Megatron `torch_dist` conversion takes significant B200 time.

    **What to watch.** W&B project `glm47-lora`. LM loss, load-balancing
    loss (MoE-specific — should stay near the uniform target), and
    gradient norms. Modal dashboard for per-node GPU utilization.
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

    `LongMIT-128K` is a long-context SFT set: each example is a question
    answered from a list of passages. `prepare()`:

    1. Downloads from HF Hub.
    2. Builds a prompt of the form "Answer the question based on the
       given passages. {passages} Question: … Answer: …"
    3. Tokenizes with the **GLM-4.7 tokenizer** (so the length filter
       matches the training tokenizer exactly) and filters to
       `n_tokens ≤ 131,072`.
    4. Writes `training.jsonl` **and** rebuilds the Megatron
       `.idx.npy` / `.idx.info` files via `megatron.bridge`'s
       `build_index_files`. Megatron's data loader streams from these
       indexes — they have to be rebuilt whenever the `.jsonl` content
       changes.
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

    ### Parallelism (32 GPUs = 4 nodes × 8 B200)

    | Axis | Setting | Why |
    |---|---|---|
    | Tensor (TP)   | 2 | Shard attention / FFN matrices across pairs of GPUs |
    | Pipeline (PP) | 4 | 4-stage pipeline over the 92 transformer blocks |
    | Expert (EP)   | 4 | Distribute the 160 MoE experts |
    | Context (CP)  | 4 | Split the 128K-token sequences along the time axis |
    | Data (implicit) | 32 / (2·4·4·4) = derived, leftover DP | |

    **Context parallelism** is what makes 128K possible on 80GB GPUs.
    Without CP you'd OOM on activations long before you got to the MoE
    routing. It does add cross-GPU attention comms, hence the pairing
    with `sequence_parallel=True` to amortize.

    ### LoRA

    ```python
    LoRA(dim=128, alpha=32, dropout=0.05)
    ```

    Rank-128 is larger than most LoRA rules of thumb (typical: 8–32);
    GLM-4.7 is large enough that the extra adapter params are a
    rounding error on memory but measurably help on a hard SFT task.

    ### Memory optimizations

    - Activation recomputation — full, uniform method (`recompute_num_layers=1`
      means checkpoint every layer). Drop to `recompute_num_layers=2` or
      higher for less overhead, or set `no_recompute=True` if you have
      the memory (likely OOMs at 128K).
    - BF16 mixed precision.
    - Grouped GEMM for the MoE.
    - Fused ops: masked softmax, bias+activation, bias+dropout, RoPE.
    - TransformerBlock patch in the framework code fixes a
      LoRA + recompute gradient-flow bug.

    ### Training schedule

    | Parameter | Value |
    |---|---|
    | Global batch size | 32 |
    | Micro batch size | 1 |
    | Sequence length | 131,072 |
    | Learning rate | 1e-4 (cosine decay) |
    | Training iters | 1 (smoke run — bump for real training) |
    | Save interval | 100 |
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
        train_iters=1,
        lr=1e-4,
        lr_warmup_iters=0,
        save_interval=100,
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
    ## Build and run

    Unlike the other framework tutorials, this one has **two extra
    stages** beyond download/prepare/train:

    - `download_model` pulls the HF weights (~700GB, CPU-only).
    - `convert_to_megatron` runs on a B200 and converts HF format to
      Megatron `torch_dist` (the on-disk format the trainer reads).
    - `download_and_convert` is a convenience wrapper that does both
      in one detached run.
    - `prepare_dataset` runs the dataset pipeline above.
    - `train_lora` is the actual 4-node LoRA run.

    Always `--detach` the long ones.
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
    # Step 1a: Download only
    uv run modal run tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.download_model

    # Step 1b: Convert only (after download_model)
    uv run modal run tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.convert_to_megatron

    # Or both in one detached call:
    uv run modal run --detach tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.download_and_convert

    # Step 2: Prepare dataset
    uv run modal run tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.prepare_dataset

    # Step 3: Train
    uv run modal run --detach tutorials/sft/megatron_glm_4_7_longmit128k/megatron_glm_4_7_longmit128k.py::app.train_lora
    ```
    """


@notebook_only
@markdown
def _run_interactive():
    """
    Interactive — one stage per cell. Use either `download_and_convert`
    or the explicit pair:
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
