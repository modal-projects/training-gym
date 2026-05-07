# pyright: reportUndefinedVariable=false
"""Tutorial source for `001_code_golf_sandboxes` — parsed by generate_tutorial.py."""

TUTORIAL_METADATA = {
    "framework": "`slime`",
    "cluster_shape": "1 × 1×H100",
    "summary": "Competitive programming RL with Harbor USACO and sandboxed verification",
    "difficulty": "Intermediate",
    "order": 20,
    "api_classes": [
        "HarborDataset",
        "DeploymentConfig",
        "EvalConfig",
        "EvalRowResult",
        "ModelDeployment",
        "Qwen3_4B",
        "SlimeRecipe",
        "TrainConfig",
        "WandbConfig",
    ],
}


from tutorial_generator import code, markdown, notebook_only, py_only, shell


@markdown
def _intro():
    """
    # Competitive programming RL with Harbor USACO + Modal sandboxes

    This tutorial trains a model on USACO competitive programming problems
    from the [Harbor Hub](https://hub.harborframework.com/datasets/usaco/usaco/latest),
    scoring solutions by executing them in Modal sandboxes.

    Workflow:
    1. Pull the USACO dataset from Harbor Hub via `HarborDataset`.
    2. Score model outputs by executing code in Modal sandboxes,
       piping test inputs via stdin and comparing stdout.
    3. Use that scorer for both offline eval and as SLIME `custom_rm_function`.
    4. Train and compare base vs. trained behavior.

    The key pattern: **correctness drives reward**.
    Reward = fraction of test cases whose output matches expected.
    """


@py_only
@markdown
def _run_instructions():
    """
    Run with:
    ```
    uv run python tutorials/rl/001_code_golf_sandboxes/001_code_golf_sandboxes.py
    ```
    """


@notebook_only
@shell("%uv pip install -q git+https://github.com/modal-projects/training-gym.git@main harbor")
def _install():
    pass


@code
def _imports():
    import json
    import re
    from functools import lru_cache

    import modal

    from modal_training_gym import (
        DeploymentConfig,
        EvalConfig,
        EvalRowResult,
        HarborDataset,
        ModelDeployment,
        Qwen3_4B,
        SlimeRecipe,
        TrainConfig,
        WandbConfig,
    )
    from modal_training_gym.common.checkpoint import list_checkpoints


@markdown
def _dataset_intro():
    """
    ## Load USACO from Harbor Hub

    `HarborDataset` accepts a `dataset_name` to pull datasets from
    [Harbor Hub](https://hub.harborframework.com). Each USACO task has:
    - `instruction.md` — the problem statement (prompt)
    - `task.toml` — metadata (difficulty, category)
    - `tests/` — input/output test pairs for verification

    Setting `test_data_dir="tests"` reads `*.in`/`*.out` file pairs and
    embeds them in each sample's label so the reward function can verify
    solutions without filesystem access.

    A single dataset instance handles both training and eval —
    `prepare()` writes train and eval splits to the volume,
    while `load()` returns all tasks for offline evaluation.
    """


@code
def _dataset():
    dataset = HarborDataset(
        dataset_name="usaco/usaco",
        label_metadata_path="task.toml",
        test_data_dir="tests",
        train_size=5,
        eval_size=5,
        train_repeats=2,
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
def _sandbox_reward_intro():
    """
    ## Sandbox-backed scorer

    We execute candidate code in a Modal sandbox with test inputs piped via
    `sys.stdin`, then compare stdout against expected output. All test cases
    run in a single sandbox per sample for efficiency.
    """


@code
def _sandbox_reward():
    _CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
    _SANDBOX_IMAGE = modal.Image.debian_slim(python_version="3.11")

    def extract_python_code(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if "<|im_start|>assistant" in normalized:
            normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
        if "</think>" in normalized:
            normalized = normalized.split("</think>", 1)[-1]
        normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
        if match := _CODE_FENCE_RE.search(normalized):
            return match.group(1).strip()
        return normalized

    @lru_cache(maxsize=1)
    def _sandbox_app() -> modal.App:
        return modal.App.lookup("training-gym-usaco-rm", create_if_missing=True)

    def score_usaco_with_sandbox(
        response: str,
        *,
        test_cases: list[dict],
        timeout_sec: int = 60,
    ) -> tuple[float, dict]:
        code = extract_python_code(response)
        if not test_cases:
            return 0.0, {"passed": 0, "total": 0}

        script = "\n".join([
            "import sys, io, json",
            f"candidate = {json.dumps(code)}",
            f"tests = {json.dumps(test_cases)}",
            "results = []",
            "for tc in tests:",
            "    sys.stdin = io.StringIO(tc['input'])",
            "    buf = io.StringIO()",
            "    old = sys.stdout",
            "    sys.stdout = buf",
            "    try:",
            "        exec(candidate, {}, {})",
            "        sys.stdout = old",
            "        results.append({'output': buf.getvalue(), 'ok': True})",
            "    except Exception as e:",
            "        sys.stdout = old",
            "        results.append({'output': '', 'ok': False})",
            "print(json.dumps(results))",
        ]) + "\n"

        try:
            sandbox = modal.Sandbox.create(
                "python", "-c", script,
                app=_sandbox_app(),
                image=_SANDBOX_IMAGE,
                timeout=timeout_sec,
                cpu=1.0,
                memory=1024,
            )
            stdout = sandbox.stdout.read()
            sandbox.stderr.read()
            sandbox.wait()
        except Exception:
            return 0.0, {"passed": 0, "total": len(test_cases)}

        try:
            results = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return 0.0, {"passed": 0, "total": len(test_cases)}

        passed = sum(
            1 for r, tc in zip(results, test_cases)
            if r.get("ok") and r.get("output", "").strip() == tc["expected_output"].strip()
        )
        return passed / len(test_cases), {"passed": passed, "total": len(test_cases)}

    async def usaco_rm(args, sample, **kwargs) -> float:
        import asyncio
        import json

        raw = getattr(sample, "label", None)
        label = json.loads(raw) if isinstance(raw, str) else (raw or {})
        test_cases = label.get("test_cases", [])
        if not test_cases:
            return 0.0

        reward, meta = await asyncio.to_thread(
            score_usaco_with_sandbox, sample.response, test_cases=test_cases,
        )
        sample.metadata = {**(getattr(sample, "metadata", None) or {}), "usaco": meta}
        return float(reward)


@markdown
def _serve_eval_base_intro():
    """
    ## Serve and evaluate the base model
    """


@code
def _serve_eval_base():
    base_model = Qwen3_4B()
    base_deployment: ModelDeployment = DeploymentConfig(model=base_model).serve()
    print(f"Base model URL: {base_deployment.url}")

    def eval_response_fn(example: dict, response: str) -> EvalRowResult:
        test_cases = example.get("label", {}).get("test_cases", [])
        score, metadata = score_usaco_with_sandbox(response, test_cases=test_cases)
        return EvalRowResult(score=score, response=response, metadata=metadata)

    eval_config = EvalConfig(
        dataset=dataset,
        eval_response_fn=eval_response_fn,
        generate_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("Running base eval...")
    base_eval = eval_config.evaluate(base_deployment, debug=True)
    print(f"Base mean reward: {base_eval.mean:.4f}")


@markdown
def _train_intro():
    """
    ## Train with SLIME and sandbox reward
    """


@code
def _train():
    training_run = TrainConfig(
        model=Qwen3_4B(),
        dataset=dataset,
        recipe=SlimeRecipe(
            wandb=WandbConfig(project="gym-tutorial", group="qwen3-4b-usaco"),
            custom_rm_function=usaco_rm,
            num_rollout=10,
            rollout_batch_size=32,
            n_samples_per_prompt=8,
            global_batch_size=256,
            rollout_max_response_len=2048,
            eval_max_response_len=2048,
            n_samples_per_eval_prompt=8,
            rollout_temperature=0.9,
            max_tokens_per_gpu=4096,
            save_interval=10,
            apply_chat_template_kwargs='{"enable_thinking": false}',
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
    trained_deployment = DeploymentConfig(
        model=Qwen3_4B(),
        checkpoint=checkpoint,
        app_name="qwen3-4b-usaco-serve",
        served_model_name="qwen3-4b-usaco",
    ).serve()
    print(f"Trained model URL: {trained_deployment.url}")

    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    print(f"Trained mean reward: {trained_eval.mean:.4f}")
    print(f"Base mean reward:    {base_eval.mean:.4f}")
