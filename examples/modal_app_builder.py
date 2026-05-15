"""Train Qwen3-8B to write deployable Modal apps via GRPO.

Synthetically generates app-building prompts from a fixed idea list and
trains the model with a sandbox-backed reward function that runs
``modal deploy`` on the generated code.

Setup:
    export MODAL_TOKEN_ID=<your-token-id>
    export MODAL_TOKEN_SECRET=<your-token-secret>
    MODAL_ENVIRONMENT=joy-agent-dev uv run python examples/modal_app_builder.py

References:
    - 001_sandboxes tutorial  (sandbox reward pattern)
    - Modal SDK skill         https://github.com/modal-projects/skills
    - Modal API docs          https://modal.com/docs
    - AGENTS.md               training-gym conventions
"""

import json
import os
import re
import uuid

from modal_training_gym import (
    DatasetConfig,
    DeploymentConfig,
    EvalConfig,
    EvalRowResult,
    ModelDeployment,
    Qwen3_8B,
    SlimeRecipe,
    TrainConfig,
    list_checkpoints,
)

# ── Credentials (captured at import time, cloudpickled to remote workers) ─

_MODAL_TOKEN_ID = os.environ.get("MODAL_TOKEN_ID", "")
_MODAL_TOKEN_SECRET = os.environ.get("MODAL_TOKEN_SECRET", "")
_MODAL_ENVIRONMENT = os.environ.get("MODAL_ENVIRONMENT", "joy-agent-dev")

# ── Fixed list of Modal app ideas ─────────────────────────────────────────

MODAL_APP_IDEAS = [
    "a web endpoint that returns 'Hello, World!' as JSON",
    "a function that computes the nth Fibonacci number",
    "a scheduled cron job that prints the current UTC time every hour",
    "an API endpoint that converts Celsius to Fahrenheit",
    "a function that generates a random UUID and returns it",
    "a web endpoint that accepts a POST with a name and returns a greeting",
    "a function that counts the number of words in a given string",
    "a web endpoint that returns the current date and time as ISO 8601",
    "a function that checks if a given number is prime",
    "a function that reverses a string and returns it",
    "a web endpoint that accepts a list of numbers and returns their sum",
    "a function that computes the factorial of a given integer",
    "a scheduled job that logs 'heartbeat' every 5 minutes",
    "a web endpoint that returns server uptime information",
    "a function that generates a random password of a given length",
    "a web endpoint that echoes back the request headers as JSON",
    "a function that sorts a list of strings alphabetically",
    "a web endpoint that returns a random motivational quote",
    "a function that validates whether a string is a valid email address",
    "a web endpoint that computes the SHA-256 hash of a given input string",
]

SYSTEM_PROMPT = (
    "You are an expert Modal developer. Your task is to write a complete, "
    "deployable Modal app in a single Python file.\n\n"
    "Modal is a serverless cloud platform for running Python code. "
    "Full API reference: https://modal.com/docs/reference\n"
    "Guide: https://modal.com/docs/guide\n\n"
    "Key Modal API patterns:\n"
    '- Create an app: `app = modal.App("my-app")`\n'
    "- Decorate functions: `@app.function()` to run on Modal\n"
    "- Web endpoints: `@modal.fastapi_endpoint()` on top of "
    "`@app.function(image=modal.Image.debian_slim().pip_install("
    '"fastapi[standard]"))`\n'
    '- Scheduled jobs: `@app.function(schedule=modal.Cron("0 * * * *"))` '
    "or `modal.Period(hours=1)`\n"
    '- Custom images: `modal.Image.debian_slim(python_version="3.12").'
    'pip_install("package")`\n'
    '- GPU functions: `@app.function(gpu="A10G")` or `gpu="H100"`\n'
    '- Volumes: `vol = modal.Volume.from_name("my-vol", '
    'create_if_missing=True)` with `volumes={"/data": vol}`\n\n'
    "Example of a working Modal app:\n"
    "```python\n"
    "import modal\n\n"
    'app = modal.App("example-app")\n\n'
    'image = modal.Image.debian_slim(python_version="3.12").pip_install('
    '"fastapi[standard]")\n\n'
    "@app.function(image=image)\n"
    "@modal.fastapi_endpoint()\n"
    "def hello():\n"
    '    return {"message": "Hello, World!"}\n'
    "```\n\n"
    "Write ONLY a single Python file inside a ```python code fence. "
    "The code must be complete and deployable with `modal deploy`. "
    "Do NOT include any text, markdown, or explanations outside the "
    "code fence."
)

# ── Dataset: synthetic app ideas ──────────────────────────────────────────


class ModalAppDataset(DatasetConfig):
    """Synthetic dataset of Modal app-building prompts."""

    label_key = "label"
    input_key = "messages"
    output_format = "jsonl"
    apply_chat_template = True
    n_repeats: int = 5

    def __init__(self, *, n_repeats: int = 5, **kwargs):
        self.n_repeats = n_repeats
        if "dataset_id" not in kwargs:
            kwargs["dataset_id"] = f"modal-app-ideas-{uuid.uuid4()}"
        super().__init__(**kwargs)

    def _generate_rows(self) -> list[dict]:
        rows = []
        for idea in MODAL_APP_IDEAS:
            for _ in range(self.n_repeats):
                prompt = f"Build a Modal app that does the following: {idea}."
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                rows.append(
                    {
                        "messages": messages,
                        "prompt": prompt,
                        "label": json.dumps({"task": idea}),
                    }
                )
        return rows

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        rows = self._generate_rows()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        if eval_paths:
            eval_rows = self._generate_rows()[: len(MODAL_APP_IDEAS)]
            for eval_path in eval_paths.values():
                os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                with open(eval_path, "w") as f:
                    for row in eval_rows:
                        f.write(json.dumps(row) + "\n")

    def load(self):
        return self._generate_rows()


# ── Reward function: sandbox-backed modal deploy ──────────────────────────

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_python_code(text: str) -> str:
    """Extract Python code from a model response."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "<|im_start|>assistant" in normalized:
        normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
    if "</think>" in normalized:
        normalized = normalized.split("</think>", 1)[-1]
    normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
    if match := _CODE_FENCE_RE.search(normalized):
        return match.group(1).strip()
    return normalized


def score_modal_deploy(
    response: str,
    *,
    timeout_sec: int = 120,
) -> tuple[float, dict]:
    """Score by attempting ``modal deploy`` inside a Modal sandbox.

    MODAL_TOKEN_ID / MODAL_TOKEN_SECRET are injected as sandbox secrets so
    the CLI inside the sandbox can authenticate with Modal.
    """
    import modal as _modal

    code = extract_python_code(response)
    if not code or len(code) < 20:
        return 0.0, {"error": "no_code_extracted", "code_length": len(code)}

    try:
        compile(code, "<modal-app>", "exec")
    except SyntaxError as e:
        return 0.0, {"error": "syntax_error", "detail": str(e)}

    if "modal" not in code.lower():
        return 0.0, {"error": "no_modal_import"}

    deploy_script = (
        "\n".join(
            [
                "import subprocess, sys, os, json, tempfile, re",
                f"code = {json.dumps(code)}",
                "with tempfile.NamedTemporaryFile("
                "    mode='w', suffix='.py', delete=False, dir='/tmp'"
                ") as f:",
                "    f.write(code)",
                "    app_path = f.name",
                "result = subprocess.run(",
                "    ['modal', 'deploy', app_path],",
                "    capture_output=True, text=True, timeout=90,",
                "    env={**os.environ},",
                ")",
                "# Clean up: stop the deployed app so it doesn't linger",
                "app_name = None",
                "m = re.search(r'modal\\.App\\([\"\\']([^\"\\']*)[\"\\'\\)]', code)",
                "if m:",
                "    app_name = m.group(1)",
                "if app_name and result.returncode == 0:",
                "    subprocess.run(",
                "        ['modal', 'app', 'stop', app_name],",
                "        capture_output=True, text=True, timeout=30,",
                "    )",
                "output = {",
                "    'returncode': result.returncode,",
                "    'stdout': result.stdout[-500:] if result.stdout else '',",
                "    'stderr': result.stderr[-500:] if result.stderr else '',",
                "}",
                "print(json.dumps(output))",
            ]
        )
        + "\n"
    )

    if not _MODAL_TOKEN_ID or not _MODAL_TOKEN_SECRET:
        return 0.0, {"error": "missing_modal_credentials"}

    try:
        app = _modal.App.lookup("training-gym-modal-deploy-rm", create_if_missing=True)
        sandbox = _modal.Sandbox.create(
            "python",
            "-c",
            deploy_script,
            app=app,
            image=_modal.Image.debian_slim(python_version="3.12").pip_install("modal"),
            timeout=timeout_sec,
            cpu=1.0,
            memory=1024,
            secrets=[
                _modal.Secret.from_dict(
                    {
                        "MODAL_TOKEN_ID": _MODAL_TOKEN_ID,
                        "MODAL_TOKEN_SECRET": _MODAL_TOKEN_SECRET,
                        "MODAL_ENVIRONMENT": _MODAL_ENVIRONMENT,
                    }
                ),
            ],
        )
        stdout = sandbox.stdout.read()
        sandbox.stderr.read()
        sandbox.wait()
    except Exception as e:
        return 0.0, {"error": "sandbox_error", "detail": str(e)}

    try:
        result = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return 0.0, {"error": "json_decode_error", "stdout": stdout[:200]}

    if result.get("returncode") == 0:
        return 1.0, {"deployed": True, **result}

    stderr = result.get("stderr", "")
    if "Error" in stderr and "SyntaxError" not in stderr:
        return 0.3, {"deployed": False, "partial_credit": True, **result}
    return 0.0, {"deployed": False, **result}


async def modal_deploy_rm(args, sample, **kwargs) -> float:
    """Reward model for slime: score by ``modal deploy`` success."""
    import asyncio

    reward, meta = await asyncio.to_thread(
        score_modal_deploy,
        sample.response,
    )
    sample.metadata = {**(getattr(sample, "metadata", None) or {}), "deploy": meta}
    return float(reward)


# ── Training configuration ────────────────────────────────────────────────


def eval_response_fn(example: dict, response: str) -> EvalRowResult:
    score, metadata = score_modal_deploy(response)
    return EvalRowResult(score=score, response=response, metadata=metadata)


def modal_eval_fn(
    deployment: ModelDeployment,
    example: dict,
) -> EvalRowResult:
    """Eval function that includes the system prompt in the messages.

    Training uses the full ``messages`` field (system + user prompt) via
    chat-template application, but ``EvalConfig.build_prompt`` only sends
    the bare ``prompt`` column as a user message.  This custom eval_fn
    recreates the same message structure the model saw during training so
    that the system prompt context is present at eval time.
    """
    prompt = example.get("prompt", "")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = deployment.generate(
        prompt,
        ensure_ready=False,
        messages=messages,
        chat_template_kwargs={"enable_thinking": False},
    )
    score, metadata = score_modal_deploy(text)
    return EvalRowResult(score=score, response=text, metadata=metadata)


if __name__ == "__main__":
    dataset = ModalAppDataset(n_repeats=5, always_prepare=True)

    training_run = TrainConfig(
        model=Qwen3_8B(),
        dataset=dataset,
        recipe=SlimeRecipe(
            custom_rm_function=modal_deploy_rm,
            gpu_type="H100",
            colocate=True,
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            num_rollout=50,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            rollout_max_response_len=2048,
            rollout_temperature=0.9,
            global_batch_size=8,
            eval_max_response_len=2048,
            n_samples_per_eval_prompt=8,
            max_tokens_per_gpu=2048,
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

    # ── Evaluate trained checkpoint ───────────────────────────────────────

    checkpoint = list_checkpoints(train_result.training_run_id)[-1]
    trained_deployment = DeploymentConfig(
        model=Qwen3_8B(),
        checkpoint=checkpoint,
        app_name="qwen3-8b-modal-builder-serve",
        served_model_name="qwen3-8b-modal-builder",
    ).serve()
    print(f"Trained model URL: {trained_deployment.url}")

    eval_config = EvalConfig(
        dataset=dataset,
        eval_fn=modal_eval_fn,
    )
    trained_eval = eval_config.evaluate(trained_deployment, debug=True)
    print(f"Trained mean reward: {trained_eval.mean:.4f}")
