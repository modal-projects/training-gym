"""RL setup for training a model to write pytest unit tests.

The task: the model is shown a task description and a working Python
implementation, and must reply with a pytest suite for it.

Reward is **validity-only**: 1.0 when the generated suite runs clean against
the reference implementation in a Modal Sandbox, 0.0 otherwise. "Runs clean"
means pytest exits 0 *and* collected at least one passing test — a file with
no tests exits 5 and scores 0, so an empty response cannot farm reward.

Note that a validity-only reward still has a cheap optimum: one trivially true
assertion passes just as well as a thorough suite. ``sample.metadata`` carries
``n_tests`` / ``n_assertions`` on every rollout so that collapse is visible in
the traces (``training-gym run trace <run-id>``) if it happens. Tightening the
reward means editing :func:`score_test_suite` only — the gate is the one place
that decides a score.

Usage:

    uv run examples/unit_test_rl.py preview   # print formatted prompts + labels
    uv run examples/unit_test_rl.py score     # exercise the reward in a sandbox
    uv run examples/unit_test_rl.py train [N] # run N rollouts, blocking
    uv run examples/unit_test_rl.py launch [N]# same, detached; prints the run id
    uv run examples/unit_test_rl.py eval      # base vs. trained checkpoint
"""

from __future__ import annotations

import ast
import base64
import json
import sys
import warnings
from typing import Any

from modal_training_gym import (
    DatasetConfig,
    Endpoint,
    Qwen3_5_4B,
    SlimeRecipe,
    TrainConfig,
    extract_code,
    list_checkpoints,
)

# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meticulous Python test engineer who writes pytest suites.\n"
    "Rules:\n"
    "- Reply with exactly one ```python code fence containing only test code.\n"
    "- The implementation under test is already importable: the line "
    "`from solution import *` is prepended to your file for you.\n"
    "- Write standalone `def test_*():` functions that use plain `assert`.\n"
    "- Do not redefine the implementation, and do not use mocks, files, "
    "the network, or `input()`.\n"
    "- Every assertion you write must hold for a correct implementation."
)

PROMPT_TEMPLATE = """Write a pytest test suite for the following Python code.

What the code is supposed to do:
{text}

Implementation (already saved as `solution.py`):
```python
{code}
```

Cover the typical cases and the edge cases you can justify from the code."""


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class MbppTestWritingDataset(DatasetConfig):
    """MBPP reframed as a test-writing task.

    MBPP rows carry ``text`` (the task), ``code`` (a working reference
    implementation) and ``test_list`` (ground-truth assertions). The model sees
    ``text`` and ``code``; the label carries the reference implementation the
    reward function runs the generated suite against.

    MBPP's own ``test_list`` is kept in the label but deliberately *not* shown
    to the model and not used for reward — it is there so you can spot-check
    that a reference implementation is actually sound.
    """

    input_key = "messages"
    label_key = "label"
    apply_chat_template = True
    always_prepare = True

    hf_repo = "google-research-datasets/mbpp"
    hf_config = "full"
    hf_split = "train"  # 374 rows; "test" (500) is the bigger pool
    eval_split = "validation"  # 90 rows
    # slime evaluates every eval prompt at once, so this caps the sandbox
    # fan-out at eval_n_rows * n_samples_per_eval_prompt.
    eval_n_rows = 32
    n_rows = 0  # 0 = all rows in the split
    row_offset = 0
    max_reference_chars = 2000

    def _rows_for(self, hf_split: str, *, limit: int = 0) -> list[dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_repo, self.hf_config, split=hf_split)

        rows: list[dict[str, Any]] = []
        for row in ds:
            reference = (row["code"] or "").strip()
            setup = (row["test_setup_code"] or "").strip()
            if not reference or len(reference) > self.max_reference_chars:
                continue
            # A reference we cannot parse locally can never be a valid target.
            # MBPP sources are full of unescaped regex backslashes, so the
            # SyntaxWarnings they raise here are noise, not signal.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse(f"{setup}\n\n{reference}" if setup else reference)
            except SyntaxError:
                continue

            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(
                                text=(row["text"] or "").strip(),
                                code=reference,
                            ),
                        },
                    ],
                    "label": json.dumps(
                        {
                            "task_id": row["task_id"],
                            "reference": reference,
                            "setup": setup,
                            "mbpp_tests": list(row["test_list"] or []),
                        }
                    ),
                }
            )
            if limit and len(rows) >= limit:
                break
        return rows

    def load(self, split: str = "all") -> list[dict[str, Any]]:
        if split == "eval":
            return self._rows_for(self.eval_split, limit=self.eval_n_rows)
        rows = self._rows_for(self.hf_split)
        start = min(self.row_offset, len(rows))
        stop = len(rows) if not self.n_rows else min(start + self.n_rows, len(rows))
        return rows[start:stop]

    def prepare(self, path: str, eval_paths: dict[str, str] | None = None) -> None:
        import os

        from datasets import Dataset

        os.makedirs(os.path.dirname(path), exist_ok=True)
        Dataset.from_list(self.load("train")).to_parquet(path)

        if eval_paths:
            eval_rows = self.load("eval")
            for eval_path in eval_paths.values():
                os.makedirs(os.path.dirname(eval_path), exist_ok=True)
                Dataset.from_list(eval_rows).to_parquet(eval_path)


# --------------------------------------------------------------------------
# Reward: run the generated suite against the reference implementation
# --------------------------------------------------------------------------

SANDBOX_APP_NAME = "training-gym-unit-test-rl"

# Runs inside the sandbox. Writes solution.py + test_generated.py, runs pytest
# with the built-in JUnit XML reporter (no plugins needed), and prints one
# sentinel-prefixed JSON line for the caller to parse.
_RUNNER = r"""
import base64, json, os, subprocess, sys, xml.etree.ElementTree as ET

payload = json.loads(base64.b64decode(sys.argv[1]).decode())
os.makedirs("/app", exist_ok=True)
os.chdir("/app")

solution = payload["setup"] + "\n\n" + payload["reference"] if payload["setup"] else payload["reference"]
with open("solution.py", "w") as f:
    f.write(solution + "\n")
with open("test_generated.py", "w") as f:
    f.write("from solution import *  # noqa: F401,F403\n\n" + payload["tests"] + "\n")

result = {"exit_code": -1, "tests": 0, "failures": 0, "errors": 0, "skipped": 0}
try:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_generated.py",
         "-q", "--tb=short", "-p", "no:cacheprovider", "--junitxml=report.xml"],
        capture_output=True, text=True, timeout=payload["pytest_timeout"],
    )
    result["exit_code"] = proc.returncode
    result["stdout"] = proc.stdout[-2000:]
    result["stderr"] = proc.stderr[-1000:]
except subprocess.TimeoutExpired:
    result["error"] = "pytest timed out"

try:
    suite = ET.parse("report.xml").getroot().find("testsuite")
    if suite is not None:
        for key in ("tests", "failures", "errors", "skipped"):
            result[key] = int(suite.get(key, 0))
except Exception as exc:
    result["report_error"] = repr(exc)

print("__RESULT__" + json.dumps(result))
"""


def _static_shape(test_code: str) -> dict[str, Any]:
    """Count test functions and assertions without executing anything."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(test_code)
    except SyntaxError as exc:
        return {"parse_error": f"{exc.msg} (line {exc.lineno})"}

    n_tests = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    )
    n_assertions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Assert))
    return {"n_tests": n_tests, "n_assertions": n_assertions}


def score_test_suite(
    test_code: str,
    reference: str,
    setup: str = "",
    *,
    sandbox_timeout_sec: int = 300,
    pytest_timeout_sec: int = 45,
) -> tuple[float, dict[str, Any]]:
    """Score a generated pytest suite. Returns ``(reward, metadata)``.

    Reward is 1.0 only when pytest exits 0 *and* at least one test ran and
    passed. Everything else — a syntax error, a failing assertion, a collection
    error, an empty file, a hang — is 0.0.
    A reward function must never propagate an exception: slime calls this from
    inside a Ray actor, and anything that escapes kills the whole run rather
    than scoring one rollout 0. Every failure below therefore becomes a 0.0
    with a ``reason``, and :func:`score_test_suite` wraps the lot in a blanket
    guard as a backstop.
    """
    shape = _static_shape(test_code)
    if "parse_error" in shape:
        # Cannot possibly run clean; skip the sandbox round-trip entirely.
        return 0.0, {**shape, "reason": "syntax_error"}

    try:
        return _score_in_sandbox(
            test_code,
            reference,
            setup,
            shape,
            sandbox_timeout_sec=sandbox_timeout_sec,
            pytest_timeout_sec=pytest_timeout_sec,
        )
    except BaseException as exc:  # noqa: BLE001 - never let a reward kill a run
        return 0.0, {**shape, "reason": "reward_error", "error": repr(exc)[:300]}


def _score_in_sandbox(
    test_code: str,
    reference: str,
    setup: str,
    shape: dict[str, Any],
    *,
    sandbox_timeout_sec: int,
    pytest_timeout_sec: int,
) -> tuple[float, dict[str, Any]]:
    """Sandbox half of :func:`score_test_suite`. May raise; the caller guards."""
    import modal

    payload = base64.b64encode(
        json.dumps(
            {
                "tests": test_code,
                "reference": reference,
                "setup": setup,
                "pytest_timeout": pytest_timeout_sec,
            }
        ).encode()
    ).decode()

    app = modal.App.lookup(SANDBOX_APP_NAME, create_if_missing=True)
    image = modal.Image.debian_slim(python_version="3.12").pip_install("pytest==8.3.4")

    sandbox = modal.Sandbox._experimental_create(
        "python",
        "-c",
        _RUNNER,
        payload,
        app=app,
        image=image,
        workdir="/app",
        timeout=sandbox_timeout_sec,
        cpu=(0.125, 1.0),
        memory=(256, 1024),
    )

    try:
        sandbox.wait()
        stdout = sandbox.stdout.read()
        stderr = sandbox.stderr.read()
    except modal.exception.SandboxTimeoutError:
        # The sandbox hit its wall clock. Under a concurrency spike (slime's
        # eval fans out every prompt at once) this is queueing, not a bad test
        # suite -- but validity-only cannot distinguish them, so it scores 0.
        return 0.0, {**shape, "reason": "sandbox_timeout"}
    except modal.exception.SandboxTerminatedError:
        return 0.0, {**shape, "reason": "sandbox_terminated"}
    finally:
        try:
            sandbox.terminate()
        except Exception:  # noqa: BLE001 - cleanup must not mask the result
            pass

    line = next(
        (ln for ln in stdout.splitlines() if ln.startswith("__RESULT__")),
        None,
    )
    if line is None:
        return 0.0, {
            **shape,
            "reason": "no_runner_output",
            "stdout": stdout[-1000:],
            "stderr": stderr[-1000:],
        }

    report = json.loads(line[len("__RESULT__") :])
    passed = report["tests"] - report["failures"] - report["errors"] - report["skipped"]
    clean = report["exit_code"] == 0 and passed >= 1

    metadata = {**shape, **report, "n_passed": passed}
    if not clean:
        metadata["reason"] = (
            "no_tests_collected" if report["exit_code"] == 5 else "tests_did_not_pass"
        )
    return float(clean), metadata


async def unit_test_rm(args, sample, **kwargs) -> float:
    """SLIME ``custom_rm_function``: reward a rollout's generated test suite."""
    import asyncio

    label = json.loads(sample.label)
    test_code = extract_code(sample.response, model=Qwen3_5_4B())

    reward, meta = await asyncio.to_thread(
        score_test_suite,
        test_code,
        label["reference"],
        label.get("setup", ""),
    )
    sample.metadata = {
        **(getattr(sample, "metadata", None) or {}),
        "unit_test": {**meta, "task_id": label.get("task_id")},
    }
    return reward


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def build_config(*, num_rollout: int = 10) -> TrainConfig:
    return TrainConfig(
        model=Qwen3_5_4B(),
        dataset=MbppTestWritingDataset(),
        recipe=SlimeRecipe(
            custom_rm_function=unit_test_rm,
            gpu_type="H100",
            colocate=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            # TP=1 keeps the existing TP=1 torch_dist checkpoint usable; TP=2
            # forces a reconversion that fails on this model.
            tensor_model_parallel_size=1,
            sequence_parallel=False,
            rollout_num_gpus_per_engine=1,
            num_rollout=num_rollout,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            global_batch_size=8,
            # The 10-step smoke run pinned 31% of samples at a 4096 cap, and a
            # truncated response is an unparseable fence -- a guaranteed 0.
            rollout_max_response_len=8192,
            rollout_temperature=0.9,
            eval_max_response_len=8192,
            n_samples_per_eval_prompt=4,
            max_tokens_per_gpu=10240,
            save_interval=10,
            # The reward function spawns Modal Sandboxes from the training
            # container, so `modal` has to be importable there.
            image_overlay=lambda image: image.run_commands(
                "uv pip install --system 'modal>=1.5.2'",
            ),
        ),
    )


# --------------------------------------------------------------------------
# Offline eval: base vs. trained
# --------------------------------------------------------------------------


def run_eval(deployment, rows: list[dict[str, Any]], *, max_concurrency: int = 8):
    from concurrent.futures import ThreadPoolExecutor

    deployment.wait_until_ready(timeout=15 * 60)
    model = Qwen3_5_4B()

    def _score_one(row: dict[str, Any]) -> float:
        msg = deployment.chat(row["messages"])
        response = msg.get("content") or msg.get("reasoning_content") or ""
        label = json.loads(row["label"])
        reward, _meta = score_test_suite(
            extract_code(response, model=model),
            label["reference"],
            label.get("setup", ""),
        )
        return reward

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        rewards = list(pool.map(_score_one, rows))
    return sum(rewards) / len(rewards) if rewards else float("nan")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _cmd_preview() -> None:
    rows = MbppTestWritingDataset(n_rows=3).load("train")
    print(f"{len(rows)} rows")
    for row in rows:
        label = json.loads(row["label"])
        print("=" * 72)
        print(f"task_id: {label['task_id']}")
        print(row["messages"][1]["content"])
        print(f"-- label.reference ({len(label['reference'])} chars) --")
        print(label["reference"])


def _cmd_score() -> None:
    """Exercise the reward on correct, incorrect, malformed and empty suites."""
    row = json.loads(MbppTestWritingDataset(n_rows=1).load("train")[0]["label"])
    reference, setup = row["reference"], row["setup"]
    print(f"reference (task {row['task_id']}):\n{reference}\n")

    cases = {
        "correct": "\n".join(
            f"def test_mbpp_{i}():\n    {t}" for i, t in enumerate(row["mbpp_tests"])
        ),
        "incorrect": "def test_wrong():\n    assert False, 'deliberately failing'",
        "malformed": "def test_broken(:\n    assert True",
        "empty": "",
        "trivial": "def test_trivial():\n    assert True",
    }
    for name, tests in cases.items():
        reward, meta = score_test_suite(tests, reference, setup)
        print(f"[{name:>9}] reward={reward}  {meta}")


def _cmd_train() -> None:
    """Block until training finishes."""
    num_rollout = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    result = build_config(num_rollout=num_rollout).train()
    print(f"training run id: {result.training_run_id}")


def _cmd_launch() -> None:
    """Start a detached run and return immediately with its id."""
    num_rollout = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run = build_config(num_rollout=num_rollout).launch(prepare_inputs=True)
    print(f"training run id: {run.training_run_id}")
    print(f"modal app url:   {run.modal_app_url}")


def _cmd_eval() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: unit_test_rl.py eval <training_run_id>")
    training_run_id = sys.argv[2]

    rows = MbppTestWritingDataset().load("eval")[:32]
    base = Endpoint.launch(
        Qwen3_5_4B(), unauthenticated=True, recreate_if_existing=True
    )
    print(f"base mean reward:    {run_eval(base, rows):.4f}")

    checkpoint = list_checkpoints(training_run_id)[-1]
    trained = Endpoint.launch(
        Qwen3_5_4B(), checkpoint, unauthenticated=True, recreate_if_existing=True
    )
    print(f"trained mean reward: {run_eval(trained, rows):.4f}")


COMMANDS = {
    "preview": _cmd_preview,
    "score": _cmd_score,
    "train": _cmd_train,
    "launch": _cmd_launch,
    "eval": _cmd_eval,
}


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    if command not in COMMANDS:
        raise SystemExit(f"unknown command {command!r}; expected {sorted(COMMANDS)}")
    COMMANDS[command]()
