"""Tutorial source for `002_simple_eval` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Build a math eval with a scoring function, run it against a base model, and iterate on the prompt until scores improve.",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
        "Qwen3_4B",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Writing an eval

    Before training, you need a baseline: how well does the untrained model already perform on your task?
    An eval answers that question and doubles as a sanity check for your reward function — if the base model scores 0% or 100%, something is off.
    """

@py_only
@markdown
def run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run python tutorials/intro/002_simple_eval/002_simple_eval.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass

@code
def _imports():
    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        HuggingFaceDataset,
        Qwen3_4B,
    )


@markdown
def _serve_base_intro():
    """
    ## Serve the base model

    First, deploy the model just like in the previous tutorial.
    """


@code
def _serve_base_model():
    base_model = Qwen3_4B()
    base_model_deployment = DeploymentConfig(
        model=base_model,
    ).serve()
    print(f"Base model deployed to {base_model_deployment.url}")

@markdown
def _scoring_intro():
    """
    ## Define the eval dataset

    We'll use [`Onlydrinkwater/int_division`](https://huggingface.co/datasets/Onlydrinkwater/int_division) — a HuggingFace dataset of integer division problems. Each row has a `question` (the problem) and a `label` (the correct answer). Subclass `HuggingFaceDataset` to point at it.
    """
@code
def _define_dataset_code():
    class IntDivisionDataset(HuggingFaceDataset):
        hf_repo = "Onlydrinkwater/int_division"
        input_column = "question"
        output_column = "label"
        output_format = "jsonl"
        apply_chat_template = True
        system_prompt = (
            "You are a math problem solver. Solve the given math problem."
        )
    eval_dataset = IntDivisionDataset(n_rows=20)

@notebook_only
@markdown
def _eval_dataset_head():
    """
    Let's take a look at the eval set.
    """

@notebook_only
@code
def _eval_dataset_head_code():
    df = eval_dataset.to_pandas()
    print(f"Number of rows in eval set: {len(df)}")
    df.head(5)


@markdown
def _grade_haiku_into_eval():
    """
    ## Define a scoring function

    The scoring function receives each dataset row and the model's response, then returns an `EvalRowResult` with a score. Here we do a simple exact-match: score 1.0 if the response equals the label, 0.0 otherwise.
    """

@code
def _eval_base_model():
    def eval_response_fn(_example: dict, response: str) -> EvalRowResult:
        if _example["label"] == response:
            return EvalRowResult(score=1.0, response=response)
        else:
            return EvalRowResult(score=0.0, response=response)

    eval_config = EvalConfig(
        dataset=eval_dataset,
        eval_response_fn=eval_response_fn,
    )

@code
def _eval_base_model_code():
    print("——— Running base model evaluation... ———")
    base_eval = eval_config.evaluate(base_model_deployment, debug=True)
    print(f"Average score: {base_eval.mean:.1f}")
    print("——— Base model evaluation complete ———")

@markdown
def _eval_base_model_results():
    """
    ## Debugging a zero score

    A score of 0.0 means none of the responses matched the labels exactly. Let's inspect a response to understand why.
    """

@code
def _eval_base_model_results_code():
    print(base_eval.rows[0])


@markdown
def _debug_eval_results():
    """
    ## Fixing the eval

    Two problems:
    1. **Thinking mode is on** — the model emits internal reasoning before the answer, so the response is never just a number.
    2. **No format instruction** — even without thinking, the model wraps the answer in natural language.

    Training on a reward function that always returns 0 teaches the model nothing. Let's fix all three:
    - Disable thinking via `generate_kwargs` and cap `max_tokens` to keep responses short.
    - Embed the format instruction in `prompt_template` so it reaches the model at inference time. (Note: `system_prompt` only affects training data preparation — it isn't sent during `generate()` calls.)
    - Extract the first integer from the response with a regex instead of requiring an exact string match.
    """

@code
def _fixed_eval_response_fn():
    import re

    class FixedIntDivisionDataset(HuggingFaceDataset):
        hf_repo = "Onlydrinkwater/int_division"
        input_column = "question"
        output_column = "label"
        output_format = "jsonl"
        apply_chat_template = True
        prompt_template = (
            "Solve this math problem. Reply with ONLY the integer answer, nothing else.\n\n{input}"
        )
    
    def new_eval_response_fn(_example: dict, response: str) -> EvalRowResult:
        match = re.search(r"-?\d+", response)
        extracted = match.group() if match else ""
        if _example["label"] == extracted:
            return EvalRowResult(score=1.0, response=response)
        else:
            return EvalRowResult(score=0.0, response=response)

    new_eval_config = EvalConfig(
        dataset=FixedIntDivisionDataset(),
        eval_response_fn=new_eval_response_fn,
        generate_kwargs={
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 20,
        },
    )

@code
def _fixed_eval_response_fn_code():
    print("——— Running fixed eval... ———")
    fixed_eval = new_eval_config.evaluate(base_model_deployment, debug=True)
    print(f"Average fixed eval score: {fixed_eval.mean:.1f}")
    print("——— Fixed eval complete ———")