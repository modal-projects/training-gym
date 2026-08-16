"""Tutorial source for `000_rl_basics` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "Haiku evaluation with verifiable rewards",
    "difficulty": "Beginner",
    "order": 10,
    "api_classes": [
        "Qwen3_5_4B",
        "Endpoint",
        "TrainConfig",
        "SlimeRecipe",
        "TrainResult",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # RL basics: verifiable rewards, haiku edition

    This tutorial uses Qwen3.5-4B and haiku poems to introduce the
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
    verifiable half. In a later tutorial, we will cover the subjective half.
    """

@py_only
@markdown
def run_instructions():
    """
    To run the tutorial, run the following command:
    ```
    uv run tutorials/rl/000_rl_basics/000_rl_basics.py
    ```
    """


@notebook_only
@shell(
    "import importlib.util\n"
    "\n"
    "# Skip if modal_training_gym is already importable (e.g. a local editable\n"
    "# checkout) so your edits keep taking effect and the env stays synced.\n"
    "if importlib.util.find_spec('modal_training_gym') is None:\n"
    "    %uv pip install -q git+https://github.com/modal-projects/training-gym.git@main\n"
    "if importlib.util.find_spec('nltk') is None:\n"
    "    %uv pip install -q nltk"
)
def _install():
    pass

@py_only
@code
def _ensure_nltk():
    import importlib.util

    if importlib.util.find_spec("nltk") is None:
        raise RuntimeError(
            "This tutorial requires the 'nltk' package. "
            "Install it before running: uv pip install -q nltk"
        )


@code
def _imports():
    import re

    from modal_training_gym import (
        Endpoint,
        HuggingFaceDataset,
        Qwen3_5_4B,
        SlimeRecipe,
        TrainConfig,
        convert_checkpoint_to_hf,
        list_checkpoints,
    )


@markdown
def _serve_base_intro():
    """
    ## Serve the base model

    So, how does Qwen3.5-4B currently fare at writing haikus? We can
    serve the base model and find out.

    The training gym has several config classes so you can define deployment, training, and evaluation configurations,
    and reuse them across different runs for parameter sweeps.

    `Endpoint.launch` provisions a Modal endpoint that serves Qwen3.5-4B
    behind an OpenAI-compatible Chat Completions API. Pass
    `unauthenticated=True` so the endpoint is reachable without Modal
    proxy-auth tokens. `launch` returns once the endpoint has a URL;
    `wait_until_ready` waits until it can serve.
    """


@code
def _serve_base_model():
    base_model = Qwen3_5_4B()
    base_model_deployment = Endpoint.launch(base_model, unauthenticated=True)
    base_model_deployment.wait_until_ready(timeout=15 * 60)
    print(f"Base model deployed to {base_model_deployment.url}")

@notebook_only
@markdown
def _qualitative_eval_of_base_model():
    """
    The model will take a moment to download and spin up, but once it's ready, we can request it to write a haiku about a topic.
    """

@notebook_only
@code
def _qualitative_eval_of_base_model_code():
    msg = base_model_deployment.chat(
        [{"role": "user", "content": "Write a haiku about cat."}],
        chat_template_kwargs={"enable_thinking": True},
    )
    response = msg.get("content") or msg.get("reasoning_content") or ""
    print(response)


@markdown
def _scoring_intro():
    """
    Let's now cover the evaluation part of the tutorial.

    A good eval takes a particular outcome and assigns a score to it. It can be binary (pass/fail) or continuous (0-100),
    deterministic or subjective, and cheap or expensive to compute.

    In our case, we want our model to be good at writing haiku poems, so how do we evaluate if an llm response was a good haiku or not?
    
    Well, a haiku must follow the 5-7-5 syllable format, so we can count syllables using NLTK's CMU Pronouncing Dictionary
    (with a regex fallback for words not in the dictionary)
    and score how close each line is to its target syllable count.

    We can give it score 0 if it doesn't follow the 5-7-5 syllable format, and 1 if it does. But that's not very informative.
    Instead, we can score it based on how close it is to the target syllable count for each line.
    """
@code
def _score_haiku():
    _cmudict_cache = {}

    def _get_cmudict() -> dict:
        if not _cmudict_cache:
            import nltk
            from nltk.corpus import cmudict
            nltk.download("cmudict", quiet=True)
            _cmudict_cache.update(cmudict.dict())
        return _cmudict_cache

    def _count_syllables(text: str) -> int:
        cmu = _get_cmudict()
        total = 0
        for word in re.findall(r"[a-zA-Z]+", text):
            phones = cmu.get(word.lower())
            if phones:
                total += sum(p[-1].isdigit() for p in phones[0])
            else:
                count = len(re.findall(r"[aeiouy]+", word.lower()))
                if word.lower().endswith("e") and count > 1:
                    count -= 1
                total += max(count, 1)
        return total

    def score_haiku(response: str) -> float:
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        if len(lines) != 3:
            return -10
        total_diff = sum(
            abs(_count_syllables(line) - target)
            for line, target in zip(lines, [5, 7, 5])
        )
        return -float(total_diff)

@notebook_only
@code
def _score_haiku_demo():
    msg = base_model_deployment.chat(
        [{"role": "user", "content": "Write a haiku about cat."}],
        chat_template_kwargs={"enable_thinking": True},
    )
    response = msg.get("content") or msg.get("reasoning_content") or ""
    print(response)
    print(f"Score: {score_haiku(response)}")

@markdown
def _define_dataset():
    """
    Let's also define a Haiku dataset.
    Here, we use the statworx/haiku dataset from HuggingFace.
    Each row has a `keywords` topic and a reference `text` haiku.
    We can use this dataset to train our model.

    Datasets for training models can take many form factors, and huggingface dataset is just one of them.
    If you're curious about other options, check out the [DatasetConfig](https://gym.modal.dev/reference/core/datasetconfig/) documentation.
    """

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
        always_prepare = True # For the purpose of this tutorial, we want to prepare the dataset every time we run it, in case there is stale data from a previous run.

    train_dataset = HaikuDataset(n_rows=10)
    eval_dataset = HaikuDataset(n_rows=5)

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

@markdown
def _grade_haiku_into_eval():
    """
    Seems straightforward enough, right? How do we run an eval on our base model with this dataset?
    Well, we already have a way to determine if a response is a good haiku or not: `score_haiku`!
    So all we need to do is loop the eval dataset, call the deployment, score each response, and aggregate.
    """


@notebook_only
@markdown
def _eval_base_intro():
    """
    ## Evaluate the base model

    """

@code
def _eval_base_model():
    def run_eval(deployment, *, max_concurrency: int = 2) -> float:
        from concurrent.futures import ThreadPoolExecutor

        deployment.wait_until_ready(timeout=15 * 60)

        def _score_one(example):
            topic = str(example[eval_dataset.input_column])
            prompt = eval_dataset.prompt_template.format(input=topic)
            msg = deployment.chat(
                [{"role": "user", "content": prompt}],
                chat_template_kwargs={"enable_thinking": True},
            )
            return score_haiku(msg.get("content") or msg.get("reasoning_content") or "")

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            scores = list(executor.map(_score_one, eval_dataset.load()))
        return sum(scores) / len(scores) if scores else float("nan")

    print("——— Running base model evaluation... ———")
    base_mean = run_eval(base_model_deployment)
    print(f"Average haiku score: {base_mean:.1f}")
    print("——— Base model evaluation complete ———")

@markdown
def _train_intro():
    """
    ## Train with slime

    Now, let's actually train the model to write good haikus.
    Here, we use the slime framework (https://github.com/THUDM/slime) on Modal.

    All flags that are native to slime can be passed to the `TrainConfig` object.
    You can also add patches to slime using the `image_overlay` argument.
    """

@code
def _define_training_run():
    async def haiku_rm(args, sample, **kwargs) -> float:
        response = base_model.parse_response(sample.response)
        return score_haiku(response.content)
    
    training_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            custom_rm_function=haiku_rm,

            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,

            num_rollout=10,
            rollout_batch_size=16,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,

            save_interval=5,
            apply_chat_template_kwargs='{"enable_thinking": true}',

            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp 'nltk>=3.8.0'",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )


@markdown
def _train_section():
    """
    ## Train

    `TrainConfig.train()` builds the Modal app, runs training, and
    returns a `TrainResult` with the run ID and checkpoint path.
    """


@code
def _invoke_train():
    print("——— Running training... ———")
    train_result = training_run.train()
    print("——— Training complete ———")


@markdown
def _trained_eval_intro():
    """
    ## Serve and evaluate the trained checkpoint

    The returned `TrainResult` has the checkpoint path and volume
    metadata attached. Endpoints require Hugging Face weights, so convert
    the Megatron checkpoint first, then pass it to `Endpoint.launch`.
    """


@code
def _serve_and_eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    hf_checkpoint = convert_checkpoint_to_hf(checkpoint, Qwen3_5_4B())
    print(hf_checkpoint.path)

    trained_model_deployment = Endpoint.launch(
        Qwen3_5_4B(), hf_checkpoint, unauthenticated=True
    )
    trained_model_deployment.wait_until_ready(timeout=15 * 60)
    print(f"Trained model deployed to {trained_model_deployment.url}")


@markdown
def _trained_eval_section():
    """
    ## Evaluate the first checkpoint

    Now let's run the same eval on the trained model and compare.
    """


@code
def _eval_trained():
    print("——— Running trained model evaluation... ———")
    trained_mean = run_eval(trained_model_deployment)
    print(f"Trained haiku score: {trained_mean:.1f}")
    print("——— Trained model evaluation complete ———")

@markdown
def _continue_to_train_off_of_a_checkpoint():
    """
    ## Train off of a checkpoint
    Hmm, looks like the trained model is not doing very well.
    Maybe it's because it only trained for 10 iterations.

    What happens if we train it for more?
    We want to train it off of the latest checkpoint, not from scratch.
    """

@code
def _continue_to_train_off_of_a_checkpoint_code():
    new_training_run = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=train_dataset,
        checkpoint=checkpoint,
        recipe=SlimeRecipe(
            custom_rm_function=haiku_rm,

            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,

            num_rollout=20,
            rollout_batch_size=16,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,

            save_interval=10,
            apply_chat_template_kwargs='{"enable_thinking": true}',

            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp 'nltk>=3.8.0'",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )
    print("——— Running new training... ———")
    new_train_result = new_training_run.train()
    print("——— New training complete ———")

@markdown
def _trained_eval_off_of_a_checkpoint():
    """
    ## Evaluate the continued checkpoint

    Now let's run the same eval on the newly trained model and compare.
    """

@code
def _trained_eval_off_of_a_checkpoint_code():
    new_checkpoint = list_checkpoints(new_train_result.training_run_id)[-1]
    new_hf_checkpoint = convert_checkpoint_to_hf(new_checkpoint, Qwen3_5_4B())
    print(new_hf_checkpoint.path)

    new_model_deployment = Endpoint.launch(
        Qwen3_5_4B(), new_hf_checkpoint, unauthenticated=True
    )
    new_model_deployment.wait_until_ready(timeout=15 * 60)
    print(f"Newly trained model deployed to {new_model_deployment.url}")

@markdown
def _trained_eval_off_of_a_checkpoint_results():
    """
    ## Compare second-run results

    Now let's compare the results of the newly trained model and the base model.
    """

@code
def _trained_eval_off_of_a_checkpoint_results_code():
    print("——— Running trained model evaluation... ———")
    new_mean = run_eval(new_model_deployment)
    print(f"Trained model (new) haiku score: {new_mean:.1f}")
    print("——— Trained model (new) evaluation complete ———")

@markdown
def _compare_results():
    """
    ## Compare all runs

    Now let's compare the results across all three checkpoints.
    """

@code
def _compare_results_code():
    print(f"Base model haiku score: {base_mean:.1f}")
    print(f"Trained model haiku score: {trained_mean:.1f}")
    print(f"Trained model (new) haiku score: {new_mean:.1f}")
