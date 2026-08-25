# ---
# order: 1
# ---
#
# # Code RL with Harbor hello-world + Modal sandboxes
#
# What if you have a task where you want to score model outputs by running them in an environment?
#
# This tutorial trains a model on the
# [hello-world](https://hub.harborframework.com/datasets/harbor/hello-world/latest)
# task from Harbor Hub, scoring solutions by spawning and executing them in Modal sandboxes.
#
# Workflow:
# 1. Pull the hello-world task from Harbor Hub via `HarborDataset`.
# 2. Score model outputs by running them in a Modal sandbox.
# 3. Reuse the same scorer as a SLIME `custom_rm_function`.
# 4. Train and compare base vs. trained behavior.

import modal

from modal_training_gym import (
    Endpoint,
    HarborDataset,
    Qwen3_5_4B,
    SlimeRecipe,
    TrainConfig,
    extract_code,
    list_checkpoints,
)

# ## Load hello-world from Harbor Hub
#
# `HarborDataset` accepts a `dataset_name` to pull tasks from
# [Harbor Hub](https://hub.harborframework.com). Each task has:
# - `instruction.md` — the problem statement (prompt)
# - `task.toml` — metadata (difficulty, category)
# - `tests/` — verification tests (format varies by task)
#
# The hello-world task asks the agent to create `hello.txt` with
# `Hello, world!` as its content. We check this file in our eval
# and reward function, matching the task's verifier.
#
# A single dataset instance handles both training and eval —
# `prepare()` writes train and eval splits to the volume,
# while `load()` returns all tasks for offline evaluation.

EXPECTED_HELLO = "Hello, world!"

dataset = HarborDataset(
    dataset_name="harbor/hello-world",
    label_metadata_path="task.toml",
    train_repeats=20,
    always_prepare=True,  # For the purpose of this tutorial, we want to prepare the dataset every time we run it, in case there is stale data from a previous run.
    system_prompt=(
        "You are an expert Python programmer. "
        "Solve the given problem by writing a complete Python program. "
        "Your program may create or modify files as needed. "
        "Put your solution in a ```python code fence."
    ),
)

# ## Evaluate with a file-based sandbox check
#
# The custom eval sends the Harbor instruction to the model, extracts its
# Python program with `extract_code`, and executes that program with `/app`
# as the working directory in a Modal Sandbox. It then reads `/app/hello.txt`
# directly with `sandbox.filesystem.read_text` and awards a point only when
# the content matches `Hello, world!`. We keep that sandbox logic in one
# local helper so the eval and training reward use exactly the same check.
#
# Passing `model=Qwen3_5_4B()` into `extract_code` enables model-aware
# response parsing.

def score_hello_file(code):
    sandbox_app = modal.App.lookup(
        "training-gym-hello-world",
        create_if_missing=True,
    )
    sandbox_image = modal.Image.debian_slim(python_version="3.12").run_commands(
        "mkdir -p /app",
    )
    sandbox = modal.Sandbox._experimental_create(
        "sleep",
        "infinity",
        app=sandbox_app,
        image=sandbox_image,
        workdir="/app",
        timeout=10,
        cpu=0.125,
        memory=128,
    )

    stderr = None

    try:
        process = sandbox.exec("python", "-c", code, timeout=3)
        process.wait()
        stderr = process.stderr.read()
        content = sandbox.filesystem.read_text("/app/hello.txt")
        score = float(content.strip() == EXPECTED_HELLO)
        metadata = {"hello_txt": content, "stderr": stderr}
    except modal.exception.SandboxFilesystemError:
        score = 0.0
        metadata = {"error": "hello.txt was not created or not readable", "stderr": stderr}
    except modal.exception.SandboxTerminatedError:
        score = 0.0
        metadata = {"error": "Sandbox was terminated during execution", "stderr": stderr}
    finally:
        sandbox.terminate()
        sandbox.detach()
    return score, metadata

model = Qwen3_5_4B()

base_deployment = Endpoint.launch(
    model, unauthenticated=True, recreate_if_existing=True
)
print(f"Base model URL: {base_deployment.url}")

def run_eval(deployment, *, max_concurrency: int = 2) -> float:
    from concurrent.futures import ThreadPoolExecutor

    deployment.wait_until_ready(timeout=15 * 60)

    def _score_one(example):
        prompt = example["instruction"]
        msg = deployment.chat(
            [
                {"role": "system", "content": dataset.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        response = msg.get("content") or msg.get("reasoning_content") or ""
        code = extract_code(response, model=model)
        score, _metadata = score_hello_file(code)
        return score

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        rewards = list(executor.map(_score_one, dataset.load()))
    return sum(rewards) / len(rewards) if rewards else float("nan")

print("Running base eval...")
base_mean = run_eval(base_deployment)
print(f"Base mean reward: {base_mean:.4f}")

# ## Train with SLIME and sandbox reward
#
# For training, we reuse the same `extract_code` and `score_hello_file`
# helpers — wrapped in an async reward function for SLIME's `custom_rm_function`.

async def sandbox_rm(args, sample, **kwargs) -> float:
    import asyncio

    code = extract_code(sample.response, model=model)
    reward, meta = await asyncio.to_thread(score_hello_file, code)
    sample.metadata = {**(getattr(sample, "metadata", None) or {}), "sandbox": meta}
    return reward

config = TrainConfig(
    model=model,
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
            "uv pip install --system 'modal>=1.5.2'",
        ),
    ),
)
print("Starting training...")
run = config.launch()
print(f"run id: {run.training_run_id}")

# ## Evaluate the trained checkpoint

result = run.result()
checkpoint = list_checkpoints(result.training_run_id)[-1]
trained_deployment = Endpoint.launch(
    model, checkpoint, unauthenticated=True, recreate_if_existing=True
)
print(f"Trained model URL: {trained_deployment.url}")

trained_mean = run_eval(trained_deployment)
print(f"Trained mean reward: {trained_mean:.4f}")
print(f"Base mean reward:    {base_mean:.4f}")
