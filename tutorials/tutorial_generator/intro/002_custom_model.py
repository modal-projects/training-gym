"""Tutorial source for `002_custom_model` — parsed by generate_tutorial.py.

Demonstrates plugging a *custom* HuggingFace model into the eval pipeline
by subclassing `ModelConfig` — no entry in the built-in model catalog, no
registry mutation, just a subclass defined inline in the tutorial.
"""

TUTORIAL_METADATA = {
    'framework': '— (eval only)',
    'cluster_shape': '—',
    'summary': 'Custom HuggingFace model — inline `ModelConfig` subclass, serve and evaluate without training',
    'difficulty': 'Beginner',
    'order': 25,
    'api_classes': [
        'ModelConfig', 'HFModelConfiguration', 'EvalConfig',
        'EvalRowResult', 'DatasetConfig',
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Custom HuggingFace model: serve and evaluate on Modal

    **What this tutorial teaches.** How to use a model that isn't in
    the built-in catalog. The built-in classes (`Qwen3_4B`, `Qwen3_0_6B`,
    etc.) are concrete `HFModelConfiguration` subclasses, but nothing
    stops you from writing your own inline — no registry, no library
    fork, just a subclass in your tutorial file.

    **What it actually runs.** We serve
    [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
    (a 135M-param Llama-family model) via vLLM and run a small
    GSM8K math eval against it. No training — the goal is to prove
    the custom-model seam works end-to-end with serve + eval,
    cheaply.

    **Prerequisites.** A Modal account with a `huggingface-secret`
    containing your `HF_TOKEN`.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    from typing import Any

    from modal_training_gym.common.dataset import HuggingFaceDataset
    from modal_training_gym.common.eval import EvalConfig, EvalRowResult
    from modal_training_gym.common.models import HFModelConfiguration


@markdown
def _explain_model():
    """
    ## Define the custom model

    Subclass `HFModelConfiguration` with just `model_name` — the
    base class already implements `download()` via
    `huggingface_hub.snapshot_download`. That's the whole seam.

    For models that need post-download conversion or special
    tokenizer setup, override `download()` in your subclass.
    """


@code
def _define_model():
    class SmolLM2_135M(HFModelConfiguration):
        model_name = "HuggingFaceTB/SmolLM2-135M"


@markdown
def _explain_serve():
    """
    ## Serve the model

    `DeploymentConfig.serve()` builds a vLLM app on Modal, deploys it,
    and returns a `ModelDeployment` with the live endpoint URL.
    The model weights are downloaded automatically on first deploy.
    """


@code
def _serve_model():
    from modal_training_gym.common.deployment import DeploymentConfig

    model = SmolLM2_135M()
    deployment = DeploymentConfig(
        model=model,
        app_name="smollm2-135m-serve",
        served_model_name="smollm2-135m",
    ).serve()
    print(deployment.url)


@markdown
def _explain_generate():
    """
    With the model deployed, we can send it a quick prompt to
    verify it's alive.
    """


@code
def _generate():
    response = deployment.generate("What is 2 + 2?")
    print(response)


@markdown
def _explain_eval():
    """
    ## Define the eval

    An `EvalConfig` owns the model-calling loop. You provide:

    1. A **dataset** — must expose `load()` returning iterable examples.
    2. A **scoring function** — takes `(example, model_response)` and
       returns an `EvalRowResult` with a `score` and `passed` flag.
    3. A **prompt builder** — override `build_prompt()` to format each
       example into the prompt the model sees.

    We'll use a tiny slice of GSM8K (grade-school math) and check
    whether the model's answer contains the correct number.
    """


@code
def _define_dataset():
    class TinyGSM8K(HuggingFaceDataset):
        hf_repo = "openai/gsm8k"
        hf_config = "main"
        hf_split = "test[:10]"
        input_column = "question"
        output_column = "answer"


@code
def _define_scoring():
    import re

    def extract_final_number(text: str) -> str | None:
        matches = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
        return matches[-1] if matches else None

    def score_gsm8k(example: dict[str, Any], response: str) -> EvalRowResult:
        gold_answer = example["answer"]
        gold_number = extract_final_number(gold_answer.split("####")[-1])
        pred_number = extract_final_number(response)

        passed = gold_number is not None and pred_number == gold_number
        return EvalRowResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            response=response,
            metadata={"gold": gold_number, "predicted": pred_number},
        )


@code
def _define_eval_config():
    class GSM8KEval(EvalConfig):
        eval_fn = score_gsm8k

        def build_prompt(self, example: dict[str, Any]) -> str:
            return (
                f"{example['question']}\n\n"
                "Solve step by step. Put your final numeric answer "
                "after ####."
            )


@markdown
def _explain_run():
    """
    ## Run the eval

    `evaluate()` iterates over the dataset, calls the model for
    each example, scores the response, and returns an `EvalResult`
    with per-row detail.
    """


@code
def _run_eval():
    eval_dataset = TinyGSM8K()

    result = GSM8KEval(
        deployment=deployment,
        dataset=eval_dataset,
    ).evaluate(debug=True)

    print(f"\nAccuracy: {result.accuracy:.1%}")
    print(f"Passed: {result.correct}/{result.total}")

    for i, row in enumerate(result.rows):
        status = "PASS" if row.passed else "FAIL"
        print(f"  [{status}] #{i+1}: gold={row.metadata['gold']} pred={row.metadata['predicted']}")


@markdown
def _whats_next():
    """
    ## What's next

    - **Train the model** — plug the same `SmolLM2_135M` into a
      `TrainConfig` with a `SlimeRecipe` (with a `ModelArchitecture`) to GRPO-train it,
      then re-run this eval to measure improvement. See
      [`000_rl_basics`](../../rl/000_rl_basics/000_rl_basics.ipynb).
    - **Custom reward** — swap `score_gsm8k` for any function that
      returns `EvalRowResult` — the same signature works as a
      verifiable reward for RL training.
    """
