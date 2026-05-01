"""Tutorial source for `000_rl_basics` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`ms_swift`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
        "Qwen3_4B",
        "EvalConfig",
        "EvalRowResult",
        "MsSwiftConfig",
        "MsSwiftFrameworkConfig",
        "WandbConfig",
        "TrainResult",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # RL basics: verifiable rewards, haiku edition

    This tutorial uses Qwen3-4B and haiku poems to introduce the
    **verifiable reward** pattern that underpins RL post-training:

    1. Serve the base model.
    2. Define a scoring function with a verifiable reward (syllable structure).
    3. Evaluate the base model against that scorer.
    4. LoRA SFT the model on reference haiku data with ms-swift.
    5. Serve the trained checkpoint.
    6. Evaluate it with the same scorer and compare.

    Why haikus? A haiku has two attributes you can score
    automatically — whether it follows the 5-7-5 syllable format
    (deterministic, cheap) and whether the poem is actually good
    (subjective, needs an LLM judge). That split between
    *verifiable* and *subjective* rewards is exactly the landscape
    RL post-training operates in. This tutorial covers the
    verifiable half; see
    [`003_slime_with_llm_as_judge`](../003_slime_with_llm_as_judge/)
    for the LLM-judge extension.

    The training config is intentionally tiny: 1×H100 and one
    training iteration. It is a smoke run for the workflow, not a
    quality run.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass


@code
def _imports():
    import re
    from typing import Any

    import modal

    from modal_training_gym.common.dataset import DatasetConfig
    from modal_training_gym.common.eval import EvalConfig, EvalRowResult
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.train_result import TrainResult
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.frameworks.ms_swift import (
        MsSwiftConfig,
        MsSwiftFrameworkConfig,
    )
    from modal_training_gym.frameworks.ms_swift.config import HF_CACHE_PATH


@markdown
def _serve_base_intro():
    """
    ## Serve the base model

    So, how does Qwen3-4B currently fare at writing haikus? We can
    serve the base model and find out.

    `ModelConfig.serve()` builds and deploys a vLLM app, then
    returns a `ModelDeployment` with the concrete endpoint URL.
    """


@code
def _serve_base_model():
    base_model = Qwen3_4B()
    base_deployment = base_model.serve(
        app_name="qwen3-4b-base-serve",
        served_model_name="qwen3-4b-base",
    )
    print(base_deployment.url)


@markdown
def _dataset_and_eval_intro():
    """
    ## Define the haiku dataset and scorer

    The dataset loads
    [`statworx/haiku`](https://huggingface.co/datasets/statworx/haiku)
    from HuggingFace — ~7k haiku poems tagged with a `keywords`
    topic. Each row becomes a chat-format training example: a system
    prompt asking for a haiku, plus a user prompt like
    *"Write a haiku about cat."*, paired with the reference poem as
    the assistant response.

    `prepare()` writes chat-format JSONL for ms-swift:
    `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`.

    `load()` returns raw `{keywords, text}` dictionaries for local
    inspection and eval.

    The scoring function is the **verifiable reward**: it checks
    whether the response has three lines and whether each line hits
    the 5-7-5 syllable target. No network call, no GPU, no
    subjectivity — just syllable counting via a vowel-group
    heuristic. This is the kind of reward that makes RL
    post-training tractable: fast, deterministic, and always
    available inside the training loop.

    `EvalConfig` owns the model-calling loop. The task-specific part
    is a scoring function passed to `.evaluate(...)`, which must
    return `EvalRowResult`.
    """


@code
def _define_dataset_and_eval():
    SYSTEM_PROMPT = (
        "You are a haiku poet. Write a haiku about the given topic. "
        "Use the 5-7-5 syllable format across three lines."
    )


    class HaikuDataset(DatasetConfig):
        input_column = "keywords"

        def __init__(
            self,
            data_root=HF_CACHE_PATH,
            *,
            split: str = "train",
            n_train: int = 500,
            n_eval: int = 50,
        ):
            self.split = split
            self.n_train = n_train
            self.n_eval = n_eval
            self.prompt_data = f"{data_root}/haiku/{split}.jsonl"

        def load(self) -> list[dict[str, Any]]:
            from datasets import load_dataset

            ds = load_dataset("statworx/haiku", split="train")
            if self.split == "train":
                rows = ds.select(range(min(self.n_train, len(ds))))
            else:
                rows = ds.select(range(len(ds) - self.n_eval, len(ds)))
            return [dict(row) for row in rows]

        def to_pandas(self):
            import pandas as pd

            return pd.DataFrame(self.load())

        def prepare(self) -> None:
            import json
            import os

            os.makedirs(os.path.dirname(self.prompt_data), exist_ok=True)
            with open(self.prompt_data, "w") as f:
                for record in self.load():
                    f.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {
                                        "role": "user",
                                        "content": f"Write a haiku about {record['keywords'].lower()}.",
                                    },
                                    {"role": "assistant", "content": record["text"]},
                                ]
                            }
                        )
                        + "\n"
                    )


    def _count_syllables(text: str) -> int:
        words = re.findall(r"[a-zA-Z]+", text)
        total = 0
        for word in words:
            count = len(re.findall(r"[aeiouy]+", word.lower()))
            if word.lower().endswith("e") and count > 1:
                count -= 1
            total += max(count, 1)
        return total


    def score_haiku(example: dict[str, Any], response: str) -> EvalRowResult:
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        has_three_lines = len(lines) == 3

        syllable_score = 0.0
        syllable_counts = []
        if has_three_lines:
            for line, target in zip(lines, [5, 7, 5]):
                count = _count_syllables(line)
                syllable_counts.append(count)
                diff = abs(count - target)
                if diff == 0:
                    syllable_score += 1.0
                elif diff == 1:
                    syllable_score += 0.5
            syllable_score /= 3.0

        score = (0.25 if has_three_lines else 0.0) + 0.75 * syllable_score
        return EvalRowResult(
            score=score,
            passed=score >= 0.75,
            response=response,
            metadata={
                "lines": len(lines),
                "syllable_counts": syllable_counts,
            },
        )


    class HaikuPrompts(EvalConfig):
        def build_prompt(self, example: dict[str, Any]) -> str:
            return (
                f"Write a haiku about {example['keywords'].lower()}.\n\n"
                "Output only the three lines of the haiku, nothing else."
            )


    train_dataset = HaikuDataset(split="train", n_train=500)
    eval_dataset = HaikuDataset(split="eval", n_eval=50)


@markdown
def _eval_dataset_head():
    """
    Let's take a look at the eval set. Each row has a `keywords`
    topic and a reference `text` haiku.
    """


@code
def _eval_dataset_head_code():
    df = eval_dataset.to_pandas()
    print(len(df))
    df.head(5)


@markdown
def _eval_base_intro():
    """
    ## Evaluate the base model
    """


@code
def _eval_base_model():
    base_eval = HaikuPrompts(
        deployment=base_deployment,
        dataset=eval_dataset,
    ).evaluate(score_haiku)
    print(f"Base haiku score: {base_eval.accuracy:.1%}")
    print(f"Passed (score >= 0.75): {base_eval.correct}/{base_eval.total}")


@markdown
def _train_intro():
    """
    ## Train with ms-swift

    This run uses a single H100 and one iteration — enough to
    verify the pipeline end-to-end. Increase `train_iters` and
    `n_train` for a real experiment.
    """


@code
def _define_training_run():
    swift_framework_config = MsSwiftFrameworkConfig(
        n_nodes=1,
        gpus_per_node=1,
        global_batch_size=1,
        max_length=2048,
        train_iters=1,
        num_train_epochs=1,
        save_interval=1,
        eval_iters=0,
    )

    my_training_run = MsSwiftConfig(
        name="qwen3-4b-haiku-sft",
        dataset=train_dataset,
        model=base_model,
        wandb=WandbConfig(project="qwen3-4b-haiku-sft"),
        framework_config=swift_framework_config,
    )


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

    ```bash
    uv run modal run tutorials/rl/000_rl_basics/000_rl_basics.py::app.download
    uv run modal run tutorials/rl/000_rl_basics/000_rl_basics.py::app.prepare_dataset
    uv run modal run --detach tutorials/rl/000_rl_basics/000_rl_basics.py::app.train
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
            app.download.remote()


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


@markdown
def _trained_eval_intro():
    """
    ## Serve and evaluate the trained checkpoint

    After training completes, `TrainResult.load(...)` reconstructs the
    trained model with its checkpoint path and volume metadata attached.
    Calling `.serve(...)` deploys that checkpoint behind vLLM.
    """


@code
def _serve_and_eval_trained():
    result = TrainResult.load("qwen3-4b-haiku-sft")
    print(result.latest_checkpoint_path())

    deployment = result.model.serve(
        app_name="qwen3-4b-haiku-sft-serve",
        served_model_name="qwen3-4b-haiku-sft",
    )
    print(deployment.url)

    trained_eval = HaikuPrompts(
        deployment=deployment,
        dataset=eval_dataset,
    ).evaluate(score_haiku)
    print(f"Trained haiku score: {trained_eval.accuracy:.1%}")
    print(f"Passed (score >= 0.75): {trained_eval.correct}/{trained_eval.total}")
    print(f"Delta: {trained_eval.accuracy - base_eval.accuracy:+.1%}")


@markdown
def _whats_next():
    """
    ## What's next: RL with GRPO

    SFT teaches the model to imitate reference haikus. RL goes
    further — it lets the model explore and discover poems that
    score well on the reward function, even ones that look nothing
    like the training set.

    The [`001_slime_intro`](../001_slime_intro/) tutorial runs
    GRPO (Group Relative Policy Optimization) with the
    [slime](https://github.com/THUDM/slime) framework on Modal.
    [`003_slime_with_llm_as_judge`](../003_slime_with_llm_as_judge/)
    extends the same haiku task with an LLM-as-judge style reward
    on top of the structure score defined here.
    """
