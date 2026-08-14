# pyright: reportUndefinedVariable=false
"""Tutorial source for `004_qwen35b` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "Train Qwen3.6-35B-A3B on DAPO-math with GRPO",
    "difficulty": "Advanced",
    "order": 25,
    "api_classes": [
        "HuggingFaceDataset",
        "Endpoint",
        "Qwen3_6_35B",
        "Qwen3_6_35b_Recipe",
        "TrainConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Training Qwen3.6-35B-A3B on DAPO-math

    This tutorial trains **Qwen3.6-35B-A3B** (a 35B-parameter MoE model
    with ~3B active) on grade-school math problems from
    [DAPO-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k).

    The loop:
    1. Load math problems from HuggingFace via `HuggingFaceDataset`.
    2. Score model outputs using slime's built-in `deepscaler` reward
       model, which extracts the final numerical answer and compares
       it to the ground truth.
    3. Feed that score back as a GRPO reward through SLIME.
    4. Compare base vs. trained accuracy.

    Qwen3.6-35B-A3B uses slime's mbridge conversion path:
    the HuggingFace checkpoint is pre-converted to torch_dist format
    before training, enabling fast batched weight sync during training steps.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run modal run -d tutorials/singlenode/000_qwen35b/000_qwen35b.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main"
)
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        CheckpointType,
        Endpoint,
        HuggingFaceDataset,
        Qwen3_6_35B,
        TrainConfig,
        convert_checkpoint_to_hf,
        list_checkpoints,
    )
    from modal_training_gym.train_recipes.slime_recipe import Qwen3_6_35b_Recipe


@markdown
def _dataset_intro():
    """
    ## Load DAPO-math from HuggingFace

    [DAPO-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k)
    contains ~17k math problems with ground-truth answers. We use a
    small subset for this tutorial — 100 training samples and 20 for eval.
    """


@code
def _dataset():
    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_column = "prompt"
        output_column = "label"
        output_format = "jsonl"
        apply_chat_template = True

    dataset = MathDataset(n_rows=120)


@notebook_only
@markdown
def _dataset_preview():
    """
    Let's take a quick look at the dataset.
    """


@notebook_only
@code
def _dataset_preview_code():
    rows = dataset.load()
    for row in rows.select(range(2)):
        prompt = row["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]["content"] if prompt else ""
        print(prompt[:200])
        print(f"  label: {row['label']}")
        print()


@markdown
def _train_intro():
    """
    ## Train with SLIME

    This MoE model runs on 1 × 8×H100 with TP2, PP2, CP1, EP4,
    and optimizer CPU offload, matching the native Slime parallelism
    that works for Qwen3.6-35B-A3B.

    Key points:
    - **`rm_type="deepscaler"`** — slime's built-in math reward that
      extracts and compares numerical answers. No custom reward function
      or sandbox needed.
    - The HF checkpoint is pre-converted to torch_dist format; slime's
      implicit mbridge mode handles fast weight sync during training steps.
    - Built-in slime model args come from
      `scripts/models/qwen3.5-35B-A3B.sh`; the tutorial does not patch slime.
    """


@code
def _train():
    model = Qwen3_6_35B()
    training_run = TrainConfig(
        model=model,
        dataset=dataset,
        recipe=Qwen3_6_35b_Recipe(
            rm_type="deepscaler",
            num_rollout=10,
        ),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _convert_intro():
    """
    ## Convert the checkpoint to HuggingFace format

    Slime writes Megatron-format checkpoints. We can convert them to HuggingFace
    format using `convert_checkpoint_to_hf`, which will run the conversion on a
    GPU function and write the result back to the volume.
    """


@code
def _convert_checkpoint():
    megatron_checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    hf_checkpoint = convert_checkpoint_to_hf(megatron_checkpoint, model)
    print(f"Serving checkpoint: {hf_checkpoint.path}")


@markdown
def _serve_intro():
    """
    ## Serve the trained model

    `Endpoint.launch` provisions a Modal endpoint that mounts the
    checkpoint volume and serves the weights behind an OpenAI-compatible
    API. The endpoint name is derived from the model and checkpoint.

    `launch` returns as soon as the endpoint has a URL; loading a 35B MoE
    checkpoint off the volume takes considerably longer than that, which
    is what `wait_until_ready` waits for.
    """


@code
def _serve_trained():
    endpoint = Endpoint.launch(model, hf_checkpoint, unauthenticated=True)
    endpoint.wait_until_ready(timeout=45 * 60)
    print(f"Trained model URL: {endpoint.url}")
