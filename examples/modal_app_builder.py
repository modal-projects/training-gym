"""Train Qwen3-8B to write deployable Modal apps via GRPO.

Synthetically generates app-building prompts from a fixed idea list and
trains the model with a two-stage reward function:

1. **Deploy check** — runs ``modal deploy`` in a sandbox to verify the
   code is syntactically valid and creates a working Modal app.
2. **LLM judge** — a separate model evaluates whether the deployed app
   actually implements the requested functionality (not just a thin stub).

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

# Judge model URL — set by deploy_judge() before training starts.
# The reward function calls this endpoint to grade code quality.
_JUDGE_URL: str = ""

# ── Fixed list of Modal app ideas ─────────────────────────────────────────

MODAL_APP_IDEAS = [
    # ── Volume-backed persistence ──
    (
        "a Volume-backed key-value store with PUT and GET web endpoints. "
        "Use modal.Volume to persist data as JSON files under /data. "
        "PUT /kv?key=foo&value=bar writes the value, GET /kv?key=foo reads it"
    ),
    (
        "a URL shortener that stores mappings in a modal.Volume. "
        "POST /shorten accepts a JSON body with 'url' and returns a short code. "
        "GET /r/{code} returns a redirect response to the original URL"
    ),
    (
        "a paste-bin service using a modal.Volume to persist text snippets. "
        "POST /paste accepts JSON with 'content', stores it with a random ID, "
        "and returns the ID. GET /paste/{id} retrieves the content"
    ),
    (
        "a visitor counter web endpoint backed by a modal.Volume. "
        "Each request to GET /count increments a counter stored in a JSON file "
        "on the volume and returns the updated count"
    ),
    # ── Custom images with pip packages ──
    (
        "a web endpoint that accepts a CSV string, parses it with the pandas "
        "library, and returns summary statistics (mean, median, std) as JSON. "
        "Use modal.Image.debian_slim().pip_install('pandas', 'fastapi[standard]')"
    ),
    (
        "a Markdown-to-HTML converter web endpoint. Accept a POST with "
        "markdown text and return rendered HTML. Use pip_install('markdown', "
        "'fastapi[standard]') on the image"
    ),
    (
        "a web endpoint that accepts a JSON payload with 'expression' and "
        "evaluates it as a safe math expression using the 'simpleeval' library. "
        "Install simpleeval and fastapi[standard] on the image"
    ),
    (
        "a YAML-to-JSON converter web endpoint. Accept a POST with raw YAML "
        "text and return the equivalent JSON. Use pip_install('pyyaml', "
        "'fastapi[standard]') on the image"
    ),
    # ── Scheduled jobs with real logic ──
    (
        "a scheduled cron job (every hour) that fetches the current Bitcoin "
        "price from a public API (e.g. coingecko) and appends the timestamp "
        "and price to a JSON log file stored on a modal.Volume"
    ),
    (
        "a scheduled job using modal.Period(minutes=30) that writes system "
        "health metrics (timestamp, random simulated cpu/memory percentages) "
        "as a JSON line to a log file on a modal.Volume"
    ),
    # ── Multiple functions / fan-out ──
    (
        "an app with two functions: a @app.function 'process_item' that "
        "uppercases a string, and a web endpoint 'batch_process' that "
        "accepts a JSON list of strings, calls process_item.remote() for "
        "each one using map, and returns the results"
    ),
    (
        "an app with a web endpoint that accepts a list of URLs, spawns "
        "a separate @app.function for each URL that returns the HTTP status "
        "code (using the requests library), collects results via .map(), "
        "and returns a JSON mapping of url -> status_code"
    ),
    # ── @app.cls (class-based) ──
    (
        "a class-based Modal app using @app.cls() with an @modal.enter() "
        "method that loads a list of facts into memory, and a "
        "@modal.fastapi_endpoint() method that returns a random fact"
    ),
    (
        "a class-based Modal app using @app.cls() with a web endpoint that "
        "maintains an in-memory counter via @modal.enter(), increments it on "
        "each request, and returns the count"
    ),
    # ── Multi-endpoint apps ──
    (
        "an app with three web endpoints: POST /items to add an item (stored "
        "in a modal.Volume as JSON), GET /items to list all items, and "
        "DELETE /items/{id} to remove an item by ID"
    ),
    (
        "a unit conversion API with three web endpoints: "
        "GET /convert/temperature?value=100&from=C&to=F, "
        "GET /convert/length?value=10&from=km&to=miles, and "
        "GET /convert/weight?value=5&from=kg&to=lbs. Each endpoint "
        "performs the conversion and returns JSON"
    ),
    # ── Image builds with system packages ──
    (
        "a web endpoint that accepts an image URL, downloads it using "
        "the requests library, generates a thumbnail using Pillow "
        "(PIL), and returns the thumbnail dimensions. "
        "Install pillow, requests, and fastapi[standard] on the image"
    ),
    (
        "a web endpoint that accepts text and returns a word frequency "
        "analysis: the top 10 most common words with their counts, using "
        "Python's collections.Counter. Return as sorted JSON"
    ),
    # ── Error handling / validation ──
    (
        "a JSON schema validator web endpoint. Accept a POST with "
        "'schema' and 'data' fields. Use the jsonschema library to validate "
        "the data against the schema and return whether it's valid plus any "
        "error messages. Install jsonschema and fastapi[standard]"
    ),
    (
        "a base64 encoding/decoding service with two endpoints: "
        "POST /encode accepts text and returns base64, "
        "POST /decode accepts base64 and returns text. "
        "Handle invalid base64 input gracefully with error responses"
    ),
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
    'pip_install("pkg1", "pkg2")`\n'
    '- GPU functions: `@app.function(gpu="A10G")` or `gpu="H100"`\n'
    '- Volumes: `vol = modal.Volume.from_name("my-vol", '
    'create_if_missing=True)` with `volumes={"/data": vol}`\n'
    "- Fan-out: call `fn.remote(arg)` or `list(fn.map(args))` from "
    "another Modal function\n"
    "- Classes: `@app.cls()` with `@modal.enter()` for setup and "
    "`@modal.fastapi_endpoint()` for serving\n"
    "- Multiple endpoints: define several `@app.function` / "
    "`@modal.fastapi_endpoint` in one app\n\n"
    "Example — a Volume-backed counter web endpoint:\n"
    "```python\n"
    "import json, modal\n\n"
    'app = modal.App("counter-app")\n'
    'vol = modal.Volume.from_name("counter-vol", create_if_missing=True)\n'
    'image = modal.Image.debian_slim(python_version="3.12").pip_install('
    '"fastapi[standard]")\n\n'
    '@app.function(image=image, volumes={"/data": vol})\n'
    "@modal.fastapi_endpoint()\n"
    "def count():\n"
    '    path = "/data/count.json"\n'
    "    try:\n"
    "        vol.reload()\n"
    '        n = json.loads(open(path).read())["n"]\n'
    "    except Exception:\n"
    "        n = 0\n"
    "    n += 1\n"
    '    open(path, "w").write(json.dumps({"n": n}))\n'
    "    vol.commit()\n"
    '    return {"count": n}\n'
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


# ── LLM judge for code quality ────────────────────────────────────────────

JUDGE_PROMPT = (
    "You are evaluating a Modal app that was generated for this task:\n"
    "{task}\n\n"
    "Here is the generated code:\n"
    "```python\n{code}\n```\n\n"
    "Score the code from 0 to 10 based on whether it ACTUALLY implements "
    "the requested functionality:\n"
    "- 0-2: Trivial stub that ignores the task (e.g. returns hardcoded "
    "'Hello World' for a Fibonacci task, or is a near-copy of the example).\n"
    "- 3-5: Partially implements the task but with major gaps or wrong logic.\n"
    "- 6-8: Implements the core functionality with minor issues.\n"
    "- 9-10: Fully and correctly implements the requested functionality.\n\n"
    'Respond with ONLY a JSON object: {{"score": <int>, "reason": "<brief>"}}'  # noqa: E501
)


def judge_code_quality(code: str, task: str) -> tuple[float, dict]:
    """Call the LLM judge to evaluate code quality.

    Returns ``(normalized_score, metadata)`` where score is 0.0-1.0.
    Falls back to 1.0 (no penalty) if the judge is unavailable.
    """
    import requests

    if not _JUDGE_URL:
        return 1.0, {"judge": "skipped", "reason": "no judge URL configured"}

    prompt = JUDGE_PROMPT.format(task=task, code=code)
    body = {
        "model": "judge",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    try:
        resp = requests.post(
            f"{_JUDGE_URL}/v1/chat/completions",
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Strip thinking tags if present
        if "</think>" in content:
            content = content.split("</think>", 1)[-1]
        content = content.strip()
        parsed = json.loads(content)
        raw_score = int(parsed["score"])
        reason = parsed.get("reason", "")
        normalized = max(0.0, min(1.0, raw_score / 10.0))
        return normalized, {"judge_score": raw_score, "judge_reason": reason}
    except Exception as e:
        return 1.0, {"judge": "error", "detail": str(e)[:200]}


# ── Reward function: sandbox-backed modal deploy + LLM judge ──────────────

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
    task: str = "",
    timeout_sec: int = 120,
) -> tuple[float, dict]:
    """Score by deploying in a sandbox then grading with an LLM judge.

    Stage 1 — deploy: ``modal deploy`` runs inside a Modal sandbox.
    Stage 2 — judge: an LLM evaluates whether the code actually
    implements the requested functionality (not just a thin wrapper).

    Final score = deploy_score * judge_score.  If deploy fails the
    judge is skipped and the score is 0.0.
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
        judge_score, judge_meta = judge_code_quality(code, task)
        final_score = round(judge_score, 2)
        return final_score, {"deployed": True, **judge_meta, **result}

    stderr = result.get("stderr", "")
    if "Error" in stderr and "SyntaxError" not in stderr:
        return 0.1, {"deployed": False, "partial_credit": True, **result}
    return 0.0, {"deployed": False, **result}


async def modal_deploy_rm(args, sample, **kwargs) -> float:
    """Reward model for slime: deploy + LLM judge scoring."""
    import asyncio

    # Extract the task description from the label metadata.
    task = ""
    label_raw = getattr(sample, "label", None) or ""
    if isinstance(label_raw, str):
        try:
            task = json.loads(label_raw).get("task", "")
        except (json.JSONDecodeError, ValueError):
            pass

    reward, meta = await asyncio.to_thread(
        score_modal_deploy,
        sample.response,
        task=task,
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
    """Eval function that includes the system prompt and LLM judge.

    Training uses the full ``messages`` field (system + user prompt) via
    chat-template application, but ``EvalConfig.build_prompt`` only sends
    the bare ``prompt`` column as a user message.  This custom eval_fn
    recreates the same message structure the model saw during training so
    that the system prompt context is present at eval time.
    """
    prompt = example.get("prompt", "")
    task = ""
    label_raw = example.get("label", "")
    if isinstance(label_raw, str):
        try:
            task = json.loads(label_raw).get("task", "")
        except (json.JSONDecodeError, ValueError):
            pass
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
    score, metadata = score_modal_deploy(text, task=task)
    return EvalRowResult(score=score, response=text, metadata=metadata)


def deploy_judge(app_name: str = "modal-app-judge") -> str:
    """Deploy the base Qwen3-8B as a judge model and return its URL."""
    global _JUDGE_URL  # noqa: PLW0603

    judge_deployment = DeploymentConfig(
        model=Qwen3_8B(),
        checkpoint=None,
        app_name=app_name,
        served_model_name="judge",
    ).serve()
    _JUDGE_URL = judge_deployment.url
    print(f"Judge model deployed at {_JUDGE_URL}")
    return _JUDGE_URL


if __name__ == "__main__":
    # Deploy the judge model first so the reward function can call it.
    deploy_judge()

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
