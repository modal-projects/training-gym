"""Tutorial source for `003_qwen27b` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "4 x 8xH100",
    "summary": "Qwen3.6-27B full-weight GRPO training on 32 GPUs with DAPO-Math-17k",
    "difficulty": "Advanced",
    "order": 31,
    "api_classes": [
        "Qwen3_6_27B",
        "Qwen3_6_27b_Recipe",
        "TrainConfig",
        "TrainingRun",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Multi-node Qwen3.6-27B full-weight training

    This tutorial runs full-weight GRPO on
    [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), a
    27B-parameter hybrid language model from Qwen, using
    [slime](https://github.com/THUDM/slime) across **4 nodes
    (32 H100 GPUs)**.

    The `Qwen3_6_27b_Recipe` preset configures colocated training and
    rollout workers, EAGLE speculative decoding, CPU-offloaded Adam,
    and DeepScaler math reward verification.
    """


@py_only
@markdown
def _run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run python tutorials/multinode/003_qwen27b/003_qwen27b.py
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
        HuggingFaceDataset,
        Qwen3_6_27B,
        Qwen3_6_27b_Recipe,
        TrainConfig,
    )


@markdown
def _dataset_intro():
    """
    ## Dataset

    We use [DAPO-Math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k),
    a collection of math competition problems with verifiable answers.
    The `deepscaler` reward model checks whether the model's response
    matches the reference answer.
    """


@code
def _define_dataset():
    class MathDataset(HuggingFaceDataset):
        hf_repo = "zhuzilin/dapo-math-17k"
        input_column = ""
        output_column = ""
        input_key = "prompt"
        label_key = "label"
        output_format = "jsonl"
        apply_chat_template = True
        always_prepare = True


@markdown
def _train_intro():
    """
    ## Launch training

    `TrainConfig.launch()` starts the Modal app **detached** and returns a
    `TrainingRun` handle as soon as training is spawned. Detached means the run
    survives this process: closing the notebook, dropping your connection, or
    hitting Ctrl-C leaves the 4 nodes training on Modal.

    The handle records the Modal app id and the `train` function-call id, so the
    run can be waited on — or cancelled — from anywhere later.
    """


@code
def _build_and_run():
    def build_training_config() -> TrainConfig:
        return TrainConfig(
            model=Qwen3_6_27B(),
            dataset=MathDataset(n_rows=10),
            recipe=Qwen3_6_27b_Recipe(
                apply_chat_template_kwargs='{"enable_thinking": true}',
            ),
        )

    training_run = build_training_config()
    run = training_run.launch()

    print(f"Training run:  {run.training_run_id}")
    print(f"Modal app:     {run.modal_app_url}")
    print(f"Function call: {run.function_call_id}")


@markdown
def _reattach():
    """
    ## Reattach later

    A launched run is addressable by id from any process — the handle is
    reconstructed from the persisted `function_call_id`, so you don't need to
    keep this session open:

    ```python
    from modal_training_gym import TrainingRun

    run = TrainingRun.from_id("<training_run_id>")
    train_result = run.result()  # block for the TrainResult

    # Or stop it early:
    run.function_call.cancel(terminate_containers=True)
    ```
    """


@markdown
def _wait_intro():
    """
    ## Wait for the result (optional)

    `run.result()` blocks until training finishes and returns the
    `TrainResult`. Interrupting the wait does **not** stop the run: a launched
    run always lives in a detached Modal app, so Ctrl-C here only stops
    waiting. To actually stop training, cancel the call:
    `run.function_call.cancel(terminate_containers=True)`.

    Skip this cell entirely if you just want to launch and walk away.
    """


@code
def _wait_for_result():
    train_result = run.result()
    print(f"Training complete: {train_result.training_run_id}")
