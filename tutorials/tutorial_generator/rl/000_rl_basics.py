"""Tutorial source for `000_rl_basics` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 8×H100",
    "summary": "Writing correct haikus",
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
    # Getting started with RL

    This tutorial introduces some core features of the Training Gym by walking through 
    a simple example of Reinforcement Learning with Verifiable Rewards (RLVR), a
    foundational method of RL post-training. Here, we teach
    [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
    how to write correct haikus. Step by step, we'll show the foundations of running
    training jobs on the Gym.
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
        list_checkpoints,
    )


@markdown
def _serve_base_intro():
    """
    ## Running the base model

    As with all training tasks, we need a baseline to decide how much training we need.
    To do that, we need a way to run inference on the base model so that we can try it out.

    Luckily, [Endpoints](https://modal.com/docs/guide/endpoints) allows us to easily deploy a
    production-ready LLM inference endpoint on Modal's managed infrastructure. It supports both open
    model weights in addition to custom fine tunes, sourced from either a Hugging Face repo or a
    [Modal Volume](https://modal.com/docs/guide/volumes).

    It will take a moment to download the model weights onto a Modal Volume and boot containers past the
    [cold-start](https://modal.com/docs/guide/cold-start#what-is-a-cold-start).
    Once you see the URL has been printed, you're ready to move on!
    """


@code
def _serve_base_model():
    base_model = Qwen3_5_4B()
    base_model_deployment = Endpoint.launch(
        base_model, unauthenticated=True, recreate_if_existing=True
    )
    base_model_deployment.wait_until_ready(timeout=15 * 60)
    print(f"base model deployed to {base_model_deployment.url}")


@markdown
def _scoring_intro():
    """
    ## Defining a scoring function

    To evaluate the base model, we need a function that takes as input a haiku and outputs a score
    (a.k.a. reward when we're training) to represent whether it follows the 5-7-5 syllable format.
    We can do that using NLTK's
    [CMU Pronouncing Dictionary](https://github.com/prosegrinder/python-cmudict),

    How should we define our scoring function? We could give it a score of 0 if it doesn't follow
    the format and 1 if it does, but that's not very informative for both the models being trained, and,
    more importantly, the human training the models! Instead, we want the score to provide sufficient
    granularity such that it's immediately obvious what the failure mode is (if any). Below, we implement
    the following function:

    - Return `-10` if the model was so incompetent that failed to return three lines.
    - Otherwise, return the negative sum of absolute differences between the predicted and target
    syllable count for each line.

    What does this mean? That the model will receive increasingly negative scores the further off
    its haiku is, with a maximum score of `0`. Let's now see how it does.
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
        chat_template_kwargs={"enable_thinking": False},
    )
    response = msg.get("content") or msg.get("reasoning_content") or ""
    print(f"haiku: {response}")
    score = score_haiku(response)
    print(f"score: {score}")


@markdown
def _define_dataset():
    """
    ## Creating a dataset for training and validation

    Note that we've only qualitatively assessed its performance. Now, we should get concrete
    numbers. How do we do that? First, we'll have to curate a dataset. Luckily,
    [statworx/haiku](https://huggingface.co/datasets/statworx/haiku) from Huggingface
    already exists, so we don't have to create one ourselves.

    Note that for more complex tasks, it is almost certainly the case that you will be 
    creating your own dataset. Why? Because the task you're trying to get your model
    to do is either too expensive or simply too hard for a bigger model. In either case,
    this is because the task is sufficiently out-of-distribution, and no existing dataset 
    will serve your needs.
    
    See the
    [multi-turn example](https://gym.modal.dev/tutorials/rl/002_multiturn/) for a basic
    example of creating your own dataset, or the
    [DatasetConfig](https://gym.modal.dev/reference/core/datasetconfig/) documentation
    for a deeper dive.
    """


@code
def _define_dataset_code():
    class HaikuDataset(HuggingFaceDataset):
        hf_repo = "statworx/haiku"
        input_column = "keywords"
        output_column = "text"
        output_format = "jsonl"
        apply_chat_template = True
        prompt_template = "Write a haiku about {input}."
        always_prepare = True

    train_dataset = HaikuDataset(n_rows=10)
    eval_dataset = HaikuDataset(n_rows=5)


@notebook_only
@markdown
def _eval_dataset_head():
    """
    Let's take a quick peek at the eval set:
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
    ## Evaluating the base model

    All we need to do now is, for each sample in our eval dataset,
    call the Endpoint, score each response, and calculate the mean.
    By default, Endpoints can process multiple inputs
    [concurrently](https://modal.com/docs/guide/servers#concurrency-and-autoscaling),
    so we loop over samples in parallel to speed up eval.
    """


@code
def _eval_base_model():
    def run_eval(deployment, max_concurrency: int = 2) -> float:
        from concurrent.futures import ThreadPoolExecutor

        deployment.wait_until_ready(timeout=15 * 60)

        def _score_one(example):
            topic = str(example[eval_dataset.input_column])
            prompt = eval_dataset.prompt_template.format(input=topic)
            msg = deployment.chat(
                [{"role": "user", "content": prompt}],
                chat_template_kwargs={"enable_thinking": False},
            )
            return score_haiku(msg.get("content") or msg.get("reasoning_content") or "")

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            scores = list(executor.map(_score_one, eval_dataset.load()))
        return sum(scores) / len(scores) if scores else float("nan")

    print("running base model evaluation...")
    base_mean = run_eval(base_model_deployment)
    print(f"average score: {base_mean:.1f}")


@markdown
def _train_intro():
    """
    ## Training the model

    Finally, onto the training. The Gym supports both the
    [Slime](https://github.com/THUDM/slime) and
    [Miles](https://github.com/radixark/miles) frameworks.
    Here, we use Slime for demonstration purposes.

    Training is simple: pass in the model you intend to train,
    the dataset you wish to train on, and a recipe for how you want
    training to occur. The recipe wraps all
    [framework-native flags](https://thudm.github.io/slime/get_started/usage.html)
    in addition to providing Modal-specific ones.

    An explanation of some of the knobs we set below:

    - `colocate` shares the same GPUs between rollout and training, alternating between the two.
    This is simply for demonstration purposes: set to `False` to give rollouts dedicated GPUs and 
    go even faster.
    - `num_rollout` sets the total rollout/train iterations to run. Each iteration samples a batch, 
    scores it, and applies one policy update.
    - `rollout_batch_size` determines the number of prompts sampled per rollout iteration.
    - `custom_rm_function` allows us to use our scoring function we defined above as a reward function
    during training.

    Once we run the code below, training kicks off and we'll immediately get a run ID, which we may
    use to watch the run's progress in the dashboard.
    """


@code
def _define_training_run():
    async def haiku_rm(args, sample, **kwargs) -> float:
        response = base_model.parse_response(sample.response)
        return score_haiku(response.content)

    train_run = TrainConfig(
        model=base_model,
        dataset=train_dataset,
        recipe=SlimeRecipe(
            gpu_type="H100",
            tensor_model_parallel_size=1,
            rollout_num_gpus_per_engine=1,
            sequence_parallel=False,
            colocate=True,
            num_rollout=10,
            rollout_batch_size=16,
            rollout_max_response_len=4096,
            rollout_temperature=1.0,
            save_interval=5,
            apply_chat_template_kwargs='{"enable_thinking": false}',
            custom_rm_function=haiku_rm,
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp 'nltk>=3.8.0'",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )
    
    train_result = train_run.train()
    print(f"run id: {train_result.training_run_id}")


@markdown
def _trained_eval_intro():
    """
    ## Serve and evaluate the trained checkpoint

    We'll get the latest checkpoint and create a new Endpoint so we may evaluate it.
    """


@code
def _serve_and_eval_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    print(checkpoint.path)

    trained_model_deployment = Endpoint.launch(
        Qwen3_5_4B(), checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    trained_model_deployment.wait_until_ready(timeout=15 * 60)
    print(f"checkpoint deployed to {trained_model_deployment.url}")


@markdown
def _trained_eval_section():
    """
    Now, let's run the same eval as before.
    """


@code
def _eval_trained():
    print("running checkpoint evaluation...")
    trained_mean = run_eval(trained_model_deployment)
    print(f"average score: {trained_mean:.1f}")


@markdown
def _continue_to_train_off_of_a_checkpoint():
    """
    ## Continuing training off the checkpoint
    Hmm, it looks like the trained model is still not doing very well.
    A likely cause is that it only trained for 10 iterations.
    Let's continue training, starting from the last checkpoint.
    """


@code
def _continue_to_train_off_of_a_checkpoint_code():
    new_train_run = TrainConfig(
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
            apply_chat_template_kwargs='{"enable_thinking": false}',
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system aiohttp 'nltk>=3.8.0'",
                "python -c \"import nltk; nltk.download('cmudict', quiet=True)\"",
            ),
        ),
    )

    new_train_result = new_train_run.train()
    print(f"run id: {new_train_result.training_run_id}")


@markdown
def _trained_eval_off_of_a_checkpoint():
    """
    ## Evals Evals Evals

    Once again, we'll create a new Endpoint for the new checkpoint and run evals on it.
    """


@code
def _trained_eval_off_of_a_checkpoint_code():
    new_checkpoint = list_checkpoints(new_train_result.training_run_id)[-1]
    print(new_checkpoint.path)

    new_model_deployment = Endpoint.launch(
        Qwen3_5_4B(), new_checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    new_model_deployment.wait_until_ready(timeout=15 * 60)
    print(f"new checkpoint deployed to {new_model_deployment.url}")

    print("running new checkpoint evaluation...")
    new_mean = run_eval(new_model_deployment)
    print(f"average score: {new_mean:.1f}")
