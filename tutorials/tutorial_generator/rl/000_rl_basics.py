"""Tutorial source for `000_rl_basics` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
        "Qwen3_4B",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "TrainConfig",
        "SlimeRecipe",
        "WandbConfig",
        "TrainResult",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell

@py_only
@markdown
def _todo_impl():
    """
    TODO: Make the python implementation of the tutorial actually run and good. Omit some of the teaching sections to make code < 100 LOC.
    """
    pass

@notebook_only
@markdown
def _intro():
    """
    # RL basics: verifiable rewards, haiku edition

    This tutorial uses Qwen3-4B and haiku poems to introduce the
    **verifiable reward** pattern that underpins RL post-training:

    1. Serve the base model.
    2. Define a scoring function with a verifiable reward (syllable structure).
    3. Evaluate the base model against that scorer.
    4. GRPO-train the model with [slime](https://github.com/THUDM/slime) using the reward function.
    5. Serve the trained checkpoint.
    6. Evaluate it with the same scorer and compare.

    **Why haikus?** A haiku has two attributes you can score
    automatically — whether it follows the 5-7-5 syllable format
    (deterministic, cheap) and whether the poem is actually good. That split between
    *verifiable* and *subjective* rewards is exactly the landscape
    RL post-training operates in. This tutorial covers the
    verifiable half. In later tutorials, we will cover the subjective half.

    The training config is intentionally tiny: 1xH100 and one
    training iteration. It is a smoke run for the workflow, not a
    quality run.
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main")
def _install():
    pass

@notebook_only
@code
def _imports():
    import re
    from typing import Any

    from modal_training_gym.common.dataset import HuggingFaceDataset
    from modal_training_gym.common.deployment import DeploymentConfig
    from modal_training_gym.common.eval import EvalConfig, EvalRowResult
    from modal_training_gym.common.models import Qwen3_4B
    from modal_training_gym.common.train import TrainConfig
    from modal_training_gym.common.train_result import TrainResult
    from modal_training_gym.common.wandb import WandbConfig
    from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe


@notebook_only
@markdown
def _serve_base_intro():
    """
    ## Serve the base model

    So, how does Qwen3-4B currently fare at writing haikus? We can
    serve the base model and find out.

    `DeploymentConfig.serve()` builds and deploys a vLLM app, then
    returns a `ModelDeployment` with the concrete endpoint URL.
    """


@notebook_only
@code
def _serve_base_model():
    base_model = Qwen3_4B()
    base_deployment = DeploymentConfig(
        model=base_model,
    ).serve()
    print(base_deployment.url)

@notebook_only
@markdown
def _qualitative_eval_of_base_model():
    """
    Now that the model has come alive, we can request it to write a haiku about a topic.
    """

@notebook_only
@code
def _qualitative_eval_of_base_model_code():
    response = base_deployment.generate(
        "Write a haiku about cat.",
        chat_template_kwargs={"enable_thinking": False},
    )
    print(response)


@notebook_only
@markdown
def _quantiative_grading_of_base_model():
    """
    Okay, how do we evaluate if that was a good haiku or not?
    In addition to the poetic quality, haikus must follow the 5-7-5 syllable format.
    We can use a simple heuristic to count the syllables in the response.
    """

@notebook_only
@code
def _count_syllables_in_haiku_code():
    def _count_syllables(text: str) -> int:
        words = re.findall(r"[a-zA-Z]+", text)
        total = 0
        for word in words:
            count = len(re.findall(r"[aeiouy]+", word.lower()))
            if word.lower().endswith("e") and count > 1:
                count -= 1
            total += max(count, 1)
        return total

@code
@notebook_only
def _grade_haiku():
    def score_haiku(response: str) -> int:
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        has_three_lines = len(lines) == 3

        syllable_counts = []
        syllable_count_diffs = []
        if has_three_lines:
            for i, (line, target) in enumerate(zip(lines, [5, 7, 5])):
                count = _count_syllables(line)
                syllable_counts.append(count)
                diff = abs(count - target)
                if diff != 0:
                    print(f"Line {i+1} has {count} syllables, expected {target}. Difference: {diff}")
                    syllable_count_diffs.append(diff)
        
        return syllable_count_diffs
    
    response = base_deployment.generate(
        "Write a haiku about cat.",
        chat_template_kwargs={"enable_thinking": False},
    )
    print(response)
    print(f"Syllable count diffs: {score_haiku(response)}")

@markdown
@notebook_only
def _grade_haiku_into_eval():
    """
    Seems straightforward enough, right? How do we run an eval on our base model?
    We can transform our scoring function above into an Eval Configuration.

    First, to explain, an Eval Configuration is a class that owns the model-calling loop.
    The task-specific part is a scoring function passed to `.evaluate(...)`, which must
    return `EvalRowResult`.
    """


@markdown
@notebook_only
def _define_dataset():
    """
    Let's also define a Haiku dataset.
    Here, we use the statworx/haiku dataset from HuggingFace.
    Each row has a `keywords` topic and a reference `text` haiku.
    We can use this dataset to train our model.
    """

@notebook_only
@code
def _define_dataset_code():
    class HaikuDataset(HuggingFaceDataset):
        hf_repo = "statworx/haiku"
        input_column = "keywords"
        output_column = "text"
        output_format = "jsonl"
        apply_chat_template = True
        system_prompt = (
            "You are a haiku poet. Write a haiku about the given topic. "
            "Use the 5-7-5 syllable format across three lines."
        )
        prompt_template = "Write a haiku about {input}."

    train_dataset = HaikuDataset(n_rows=50, include_eval=True)
    eval_dataset = HaikuDataset(n_rows=10)


@notebook_only
@markdown
def _eval_dataset_head():
    """
    Let's take a look at the eval set. Each row has a `keywords`
    topic and a reference `text` haiku.
    """

@notebook_only
@code
def _eval_dataset_head_code():
    df = eval_dataset.to_pandas()
    print(len(df))
    df.head(5)

@notebook_only
@markdown
def _eval_base_intro():
    """
    ## Evaluate the base model
    """

@notebook_only
@code
def _eval_base_model():
    def haiku_prompt(example: dict) -> str:
        return (
            f"Write a haiku about {example['keywords'].lower()}.\n\n"
            "Output only the three lines of the haiku, nothing else."
        )
    
    class HaikuEvalConfig(EvalConfig):
        def build_prompt(self, example: dict) -> str:
            return (
                f"Write a haiku about {example['keywords'].lower()}.\n\n"
                "Output only the three lines of the haiku, nothing else."
            )
        
        def eval_fn(self, _df_row: dict, response: str) -> EvalRowResult:
            syllable_count_diffs = score_haiku(response)
            return EvalRowResult(
                score=sum(syllable_count_diffs),
                response=response,
                metadata={
                    "lines": len(lines),
                    "syllable_counts": syllable_counts,
                },
            )

    base_eval = HaikuEvalConfig(
        deployment=base_deployment,
        dataset=eval_dataset,
        generate_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    ).evaluate()
    print(f"Base haiku score: {base_eval.accuracy:.1%}")
    print(f"Passed (score >= 0.75): {base_eval.correct}/{base_eval.total}")

@notebook_only
@markdown
def _train_intro():
    """
    ## Train with slime

    Now, let's actually train the model to write good haikus.
    Here, we use the slime framework (https://github.com/THUDM/slime) on Modal.
    """

@notebook_only
@code
def _define_training_run():
    def haiku_rm(args, sample, **kwargs) -> float:
        syllable_count_diffs = score_haiku(sample.response)
        return sum(syllable_count_diffs)

    my_training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            wandb=WandbConfig(project="slime-grpo", group="qwen3-4b-haiku"),

            custom_rm_function=haiku_rm,

            num_rollout=10,
            save_interval=5,
            apply_chat_template_kwargs='{"enable_thinking": false}',

            image_run_commands=lambda image: image.run_commands(
                "uv pip install --system aiohttp nltk>=3.8.0",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )


@notebook_only
@markdown
def _train_section():
    """
    ## Train

    `TrainConfig.train()` builds the Modal app, runs training, and
    returns a `TrainResult` with the run ID and checkpoint path.
    """


@notebook_only
@code
def _invoke_train():
    result = my_training_run.train()


@notebook_only
@markdown
def _trained_eval_intro():
    """
    ## Serve and evaluate the trained checkpoint

    The returned `TrainResult` has the checkpoint path and volume
    metadata attached. Wrapping `result.model` in a `DeploymentConfig`
    and calling `.serve()` deploys that checkpoint behind SGLang.
    """


@notebook_only
@code
def _serve_and_eval_trained():
    print(result.latest_checkpoint_path())

    deployment = DeploymentConfig(
        model=result.model,
        app_name="qwen3-4b-haiku-serve",
        served_model_name="qwen3-4b-haiku",
    ).serve()
    print(deployment.url)

    def haiku_prompt(example: dict[str, Any]) -> str:
        return (
            f"Write a haiku about {example['keywords'].lower()}.\n\n"
            "Output only the three lines of the haiku, nothing else."
        )

    trained_eval = EvalConfig(
        prompt_fn=haiku_prompt,
        deployment=deployment,
        dataset=eval_dataset,
        eval_fn=(lambda golden_df_row, response: score_haiku(response)),
        generate_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    ).evaluate(debug=True)
    print(f"Trained haiku score: {trained_eval.accuracy:.1%}")

