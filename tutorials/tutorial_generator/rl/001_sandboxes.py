# pyright: reportUndefinedVariable=false
"""Tutorial source for `001_sandboxes` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Code RL with Harbor sandboxed evals",
    "difficulty": "Intermediate",
    "order": 20,
    "api_classes": [
        "HarborDataset",
        "CustomDeployment",
        "Qwen3_4B",
        "SlimeRecipe",
        "TrainConfig",
        "score_in_sandbox",
        "extract_code",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Code RL with Harbor hello-world + Modal sandboxes

    What if you have a task where you want to score model outputs by running them in an environment?

    This tutorial trains a model on the
    [hello-world](https://hub.harborframework.com/tasks/harbor/hello-world/latest)
    task from Harbor Hub, scoring solutions by spawning and executing them in Modal sandboxes.

    Workflow:
    1. Pull the hello-world task from Harbor Hub via `HarborDataset`.
    2. Score model outputs by running them in a Modal sandbox.
    3. Reuse the same `score_in_sandbox` helper as a SLIME `custom_rm_function`.
    4. Train and compare base vs. trained behavior.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run tutorials/rl/001_sandboxes/001_sandboxes.py
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
    "if importlib.util.find_spec('harbor') is None:\n"
    "    %uv pip install -q harbor"
)
def _install():
    pass


@code
def _imports():
    from modal_training_gym import (
        CustomDeployment,
        HarborDataset,
        Qwen3_4B,
        SlimeRecipe,
        TrainConfig,
        extract_code,
        list_checkpoints,
        score_in_sandbox,
    )


@markdown
def _dataset_intro():
    """
    ## Load hello-world from Harbor Hub

    `HarborDataset` accepts a `dataset_name` to pull tasks from
    [Harbor Hub](https://hub.harborframework.com). Each task has:
    - `instruction.md` — the problem statement (prompt)
    - `task.toml` — metadata (difficulty, category)
    - `tests/` — verification tests (format varies by task)

    The hello-world task uses pytest-based verification rather than
    `*.in`/`*.out` file pairs, so we define stdin/stdout test cases
    inline and pass them to `score_in_sandbox` via the `test_cases` field.

    A single dataset instance handles both training and eval —
    `prepare()` writes train and eval splits to the volume,
    while `load()` returns all tasks for offline evaluation.
    """


@code
def _dataset():
    HELLO_WORLD_TESTS = [{"input": "", "expected_output": "Hello, world!\n"}]

    dataset = HarborDataset(
        dataset_name="harbor/hello-world",
        label_metadata_path="task.toml",
        train_repeats=20,
        always_prepare=True, # For the purpose of this tutorial, we want to prepare the dataset every time we run it, in case there is stale data from a previous run.
        system_prompt=(
            "You are an expert Python programmer. "
            "Solve the given problem by writing a complete Python program. "
            "Your program must print the answer to stdout using print(). "
            "Do not create or write any files. "
            "Put your solution in a ```python code fence."
        ),
    )


@notebook_only
@markdown
def _dataset_preview():
    """
    Let's take a quick look at part of the dataset as a pandas DataFrame.
    Each row includes the task prompt plus the parsed Harbor label metadata.
    """


@notebook_only
@code
def _dataset_preview_code():
    df = dataset.to_pandas()
    print(len(df))
    df.head(5)


@markdown
def _harbor_eval_intro():
    """
    ## Evaluate with sandboxed scoring

    The sandbox scoring loop:
    1. Sends each task's prompt to the deployed model.
    2. Extracts Python code from the response (stripping thinking tags,
       chat-template artifacts, and code fences via `extract_code`).
    3. Runs the extracted code in a Modal sandbox against the test cases.
    4. Returns a score = fraction of test cases passed.

    Since hello-world doesn't ship `*.in`/`*.out` file pairs, we pass
    `test_cases` directly to `score_in_sandbox`.

    Passing `model=Qwen3_4B()` into `extract_code` enables model-aware
    response parsing.
    """


@code
def _serve_eval_base():
    base_model = Qwen3_4B()
    base_deployment = CustomDeployment.launch(
        base_model,
        unauthenticated=True,
    )
    print(f"Base model URL: {base_deployment.url}")

    def run_eval(deployment, *, max_concurrency: int = 2) -> float:
        from concurrent.futures import ThreadPoolExecutor

        deployment.wait_until_ready(timeout=3000)

        def _score_one(example):
            prompt = example["instruction"]
            messages = [
                {"role": "system", "content": dataset.system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = deployment.generate(
                prompt,
                ensure_ready=False,
                messages=messages,
            )
            code = extract_code(response, model=base_model)
            reward, _meta = score_in_sandbox(code, test_cases=HELLO_WORLD_TESTS)
            return float(reward)

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            rewards = list(executor.map(_score_one, dataset.load()))
        return sum(rewards) / len(rewards) if rewards else float("nan")

    print("Running base eval...")
    base_mean = run_eval(base_deployment)
    print(f"Base mean reward: {base_mean:.4f}")


@markdown
def _train_intro():
    """
    ## Train with SLIME and sandbox reward

    For training, we reuse the same `score_in_sandbox` and `extract_code`
    helpers — wrapped in an async reward function for SLIME's
    `custom_rm_function`.

    `score_in_sandbox` enforces `sandbox_cpu`/`sandbox_memory` with a
    `"limit"` policy by default: rather than reserving that capacity up
    front, the values become burst ceilings, so Modal bills each sandbox
    by actual CPU-/RAM-second usage instead of the (usually idle)
    reservation. Pass `cpu_policy="ignore"` to let rollouts burst above
    the configured values, or `"reserve"` for the legacy fixed-reservation
    behavior.
    """


@code
def _train():
    async def sandbox_rm(args, sample, **kwargs) -> float:
        import asyncio

        code = extract_code(sample.response, model=base_model)
        reward, meta = await asyncio.to_thread(
            score_in_sandbox, code, test_cases=HELLO_WORLD_TESTS,
        )
        sample.metadata = {**(getattr(sample, "metadata", None) or {}), "sandbox": meta}
        return float(reward)

    training_run = TrainConfig(
        model=Qwen3_4B(),
        dataset=dataset,
        recipe=SlimeRecipe(
            custom_rm_function=sandbox_rm,

            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,

            num_rollout=10,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            rollout_max_response_len=2048,
            rollout_temperature=0.9,

            global_batch_size=8,
            eval_max_response_len=2048,
            n_samples_per_eval_prompt=8,
            max_tokens_per_gpu=4096,
            save_interval=10,
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system modal>=1.2.0",
            ),
        ),
    )
    print("Starting training...")
    train_result = training_run.train()
    print(f"Training run id: {train_result.training_run_id}")


@markdown
def _serve_trained_intro():
    """
    ## Evaluate the trained checkpoint
    """


@code
def _serve_trained():
    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    trained_deployment = CustomDeployment.launch(
        Qwen3_4B(),
        checkpoint=checkpoint,
        app_name="qwen3-4b-hello-world-serve",
        served_model_name="qwen3-4b-hello-world",
        unauthenticated=True,
    )
    print(f"Trained model URL: {trained_deployment.url}")

    trained_mean = run_eval(trained_deployment)
    print(f"Trained mean reward: {trained_mean:.4f}")
    print(f"Base mean reward:    {base_mean:.4f}")
