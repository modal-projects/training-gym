"""Reward functions and scoring utilities for Harbor tasks.

Provides:
- ``HarborTask`` — lightweight dataclass for inline task definitions
- ``harbor_rm`` — async reward model for SlimeRecipe
- ``score_harbor_task`` — standalone sync scorer for eval pipelines
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass
class HarborTask:
    """Lightweight inline definition of a Harbor-compatible task.

    Use this when you want to define tasks directly in Python rather than
    loading from Harbor task directories on disk. Works with both
    ``HarborDataset`` (via ``write_to_dir()``) and ``score_harbor_task``.

    Fields mirror the Harbor task specification:
    - ``name``: unique task identifier
    - ``instruction``: prompt text given to the model
    - ``test_script``: newline-separated assert expressions (or list of them)
    - ``entrypoint``: function name the model must define
    - ``solution``: reference solution (optional, for eval/length bonus)
    - ``timeout_sec``: sandbox timeout
    - ``docker_image``: sandbox base image
    """

    name: str
    instruction: str
    test_script: str | list[str] = ""
    entrypoint: str = "solve"
    solution: str = ""
    timeout_sec: int = 30
    docker_image: str = "python:3.11-slim"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tests(self) -> list[str]:
        """Return test assertions as a list."""
        if isinstance(self.test_script, list):
            return self.test_script
        return [line for line in self.test_script.splitlines() if line.strip()]

    def to_label(self) -> dict[str, Any]:
        """Serialize to the label dict format expected by ``harbor_rm``."""
        label: dict[str, Any] = {
            "harbor_task_name": self.name,
            "entrypoint": self.entrypoint,
            "tests": self.tests,
            "reference_solution": self.solution,
            "timeout_sec": self.timeout_sec,
            "docker_image": self.docker_image,
        }
        label.update(self.metadata)
        return label

    def write_to_dir(self, tasks_root: str) -> str:
        """Write this task as a Harbor directory under *tasks_root*.

        Creates ``instruction.md`` and ``record.json`` files. Returns the
        path to the created task directory.
        """
        import os

        task_dir = os.path.join(tasks_root, self.name)
        os.makedirs(task_dir, exist_ok=True)

        with open(os.path.join(task_dir, "instruction.md"), "w", encoding="utf-8") as f:
            f.write(self.instruction)

        with open(os.path.join(task_dir, "record.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_name": self.name,
                    "entrypoint": self.entrypoint,
                    "tests": self.tests,
                    "reference_solution": self.solution,
                },
                f,
            )

        return task_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_code(text: str) -> str:
    """Extract Python code from a model response, handling fences and
    thinking tokens."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "<|im_start|>assistant" in normalized:
        normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
    if "</think>" in normalized:
        normalized = normalized.split("</think>", 1)[-1]
    normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
    if match := _CODE_FENCE_RE.search(normalized):
        return match.group(1).strip()
    return normalized


def _build_test_runner_script(
    candidate_code: str,
    tests: list[str],
    entrypoint: str,
) -> str:
    """Build a self-contained Python script that runs assert-style tests
    against *candidate_code* and prints a JSON result summary."""
    lines = [
        "import json",
        f"candidate = {json.dumps(candidate_code)}",
        f"tests = {json.dumps(tests)}",
        f"entrypoint = {json.dumps(entrypoint)}",
        "scope = {}",
        "try:",
        "    exec(candidate, scope, scope)",
        "    fn = scope.get(entrypoint)",
        "    if not callable(fn):",
        "        raise RuntimeError(f'Missing callable: {entrypoint}')",
        "    passed = 0",
        "    for expr in tests:",
        "        exec(expr, scope, scope)",
        "        passed += 1",
        "    print(json.dumps({",
        "        'pass_rate': passed / len(tests) if tests else 0.0,",
        "        'passed': passed,",
        "        'total': len(tests),",
        "    }))",
        "except Exception as exc:",
        "    print(json.dumps({",
        "        'pass_rate': 0.0,",
        "        'passed': 0,",
        "        'total': len(tests),",
        "        'error': repr(exc),",
        "    }))",
    ]
    return "\n".join(lines) + "\n"


def _last_json_line(text: str) -> dict:
    """Return the last JSON-object line from *text*, or ``{}``."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def _run_in_sandbox(
    script: str,
    docker_image: str = "python:3.11-slim",
    timeout_sec: int = 30,
) -> tuple[str, str, int]:
    """Execute *script* in a Modal sandbox and return (stdout, stderr, exit_code)."""
    import modal

    app = modal.App.lookup("training-gym-harbor-rm", create_if_missing=True)
    image = modal.Image.from_registry(docker_image)
    sandbox = modal.Sandbox.create(
        "python",
        "-c",
        script,
        app=app,
        image=image,
        timeout=timeout_sec,
        cpu=1.0,
        memory=1024,
    )
    stdout = sandbox.stdout.read()
    stderr = sandbox.stderr.read()
    exit_code = sandbox.wait()
    return stdout, stderr, exit_code


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------


def score_harbor_task(
    response: str,
    *,
    tests: list[str],
    entrypoint: str,
    reference_solution: str = "",
    timeout_sec: int = 30,
    docker_image: str = "python:3.11-slim",
    length_bonus_weight: float = 0.0,
) -> tuple[float, dict]:
    """Score a model *response* against assert-style *tests*.

    Returns ``(reward, metadata)`` where reward is in ``[0, 1]``.
    """
    candidate_code = _extract_code(response)
    script = _build_test_runner_script(candidate_code, tests, entrypoint)

    try:
        stdout, stderr, exit_code = _run_in_sandbox(script, docker_image, timeout_sec)
    except Exception as exc:
        candidate_bytes = len(candidate_code.encode("utf-8"))
        reference_bytes = (
            len(reference_solution.encode("utf-8")) if reference_solution else 0
        )
        return 0.0, {
            "pass_rate": 0.0,
            "candidate_bytes": candidate_bytes,
            "reference_bytes": reference_bytes,
            "ratio": 0.0,
            "sandbox_exit_code": -1,
            "sandbox_stderr": repr(exc),
        }

    payload = _last_json_line(stdout)
    pass_rate = float(payload.get("pass_rate", 0.0))

    candidate_bytes = len(candidate_code.encode("utf-8"))
    reference_bytes = (
        len(reference_solution.encode("utf-8")) if reference_solution else 0
    )
    if reference_bytes and candidate_bytes:
        ratio = min(2.0, reference_bytes / max(candidate_bytes, 1))
    else:
        ratio = 1.0
    size_bonus = max(0.0, ratio - 1.0)
    reward = pass_rate * (1.0 + (length_bonus_weight * size_bonus))

    metadata = {
        "pass_rate": pass_rate,
        "candidate_bytes": candidate_bytes,
        "reference_bytes": reference_bytes,
        "ratio": ratio,
        "sandbox_exit_code": exit_code,
        "sandbox_stderr": stderr.strip(),
    }
    return reward, metadata


# ---------------------------------------------------------------------------
# Async reward model for SlimeRecipe
# ---------------------------------------------------------------------------


async def harbor_rm(args: Any, sample: Any, **kwargs: Any) -> float:
    """Built-in reward model for Harbor tasks.

    Reads ``tests``, ``entrypoint``, and ``reference_solution`` from the
    sample's label JSON, executes the model's response in a Modal sandbox,
    and returns a reward in ``[0, 1]``.

    Drop-in replacement for hand-written ``code_golf_rm`` functions:
    ``SlimeRecipe(custom_rm_function=harbor_rm, ...)``
    """
    import asyncio
    import json
    import re
    from functools import lru_cache

    import modal

    def _inline_extract_code(text: str) -> str:
        code_fence_re = re.compile(
            r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL
        )
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if "<|im_start|>assistant" in normalized:
            normalized = normalized.rsplit("<|im_start|>assistant", 1)[-1]
        if "</think>" in normalized:
            normalized = normalized.split("</think>", 1)[-1]
        normalized = normalized.replace("<think>", "").replace("<|im_end|>", "").strip()
        if match := code_fence_re.search(normalized):
            return match.group(1).strip()
        return normalized

    def _inline_last_json_line(text: str) -> dict:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return {}

    def _inline_score(
        response: str,
        *,
        tests: list[str],
        entrypoint: str,
        reference_solution: str,
        timeout_sec: int = 30,
        docker_image: str = "python:3.11-slim",
        length_bonus_weight: float = 0.2,
    ) -> tuple[float, dict]:
        candidate_code = _inline_extract_code(response)
        lines = [
            "import json",
            f"candidate = {json.dumps(candidate_code)}",
            f"tests = {json.dumps(tests)}",
            f"entrypoint = {json.dumps(entrypoint)}",
            "scope = {}",
            "try:",
            "    exec(candidate, scope, scope)",
            "    fn = scope.get(entrypoint)",
            "    if not callable(fn):",
            "        raise RuntimeError(f'Missing callable: {entrypoint}')",
            "    passed = 0",
            "    for expr in tests:",
            "        exec(expr, scope, scope)",
            "        passed += 1",
            "    print(json.dumps({",
            "        'pass_rate': passed / len(tests) if tests else 0.0,",
            "        'passed': passed,",
            "        'total': len(tests),",
            "    }))",
            "except Exception as exc:",
            "    print(json.dumps({",
            "        'pass_rate': 0.0,",
            "        'passed': 0,",
            "        'total': len(tests),",
            "        'error': repr(exc),",
            "    }))",
        ]
        script = "\n".join(lines) + "\n"

        @lru_cache(maxsize=1)
        def _sandbox_app() -> modal.App:
            return modal.App.lookup("training-gym-harbor-rm", create_if_missing=True)

        sandbox_image = modal.Image.from_registry(docker_image)

        try:
            sandbox = modal.Sandbox.create(
                "python",
                "-c",
                script,
                app=_sandbox_app(),
                image=sandbox_image,
                timeout=timeout_sec,
                cpu=1.0,
                memory=1024,
            )
            stdout = sandbox.stdout.read()
            stderr = sandbox.stderr.read()
            exit_code = sandbox.wait()
        except Exception as exc:
            candidate_bytes = len(candidate_code.encode("utf-8"))
            reference_bytes = (
                len(reference_solution.encode("utf-8")) if reference_solution else 0
            )
            return 0.0, {
                "pass_rate": 0.0,
                "candidate_bytes": candidate_bytes,
                "reference_bytes": reference_bytes,
                "ratio": 0.0,
                "sandbox_exit_code": -1,
                "sandbox_stderr": repr(exc),
            }

        payload = _inline_last_json_line(stdout)
        pass_rate = float(payload.get("pass_rate", 0.0))
        candidate_bytes = len(candidate_code.encode("utf-8"))
        reference_bytes = (
            len(reference_solution.encode("utf-8")) if reference_solution else 0
        )
        if reference_bytes and candidate_bytes:
            ratio = min(2.0, reference_bytes / max(candidate_bytes, 1))
        else:
            ratio = 1.0
        size_bonus = max(0.0, ratio - 1.0)
        reward = pass_rate * (1.0 + (length_bonus_weight * size_bonus))
        return reward, {
            "pass_rate": pass_rate,
            "candidate_bytes": candidate_bytes,
            "reference_bytes": reference_bytes,
            "ratio": ratio,
            "sandbox_exit_code": exit_code,
            "sandbox_stderr": stderr.strip(),
        }

    # --- Parse label from sample ---
    raw_label = getattr(sample, "label", None)
    if isinstance(raw_label, dict):
        label = raw_label
    elif raw_label:
        try:
            label = json.loads(raw_label)
        except json.JSONDecodeError:
            label = {}
    else:
        label = {}

    tests = label.get("tests")
    entrypoint = label.get("entrypoint")
    reference_solution = label.get("reference_solution", "")
    if (
        not isinstance(tests, list)
        or not tests
        or not isinstance(entrypoint, str)
        or not entrypoint
    ):
        sample_metadata = getattr(sample, "metadata", None)
        if not isinstance(sample_metadata, dict):
            sample_metadata = {}
        sample_metadata["harbor"] = {"pass_rate": 0.0, "reason": "missing_label"}
        sample.metadata = sample_metadata
        return 0.0

    reward, metadata = await asyncio.to_thread(
        _inline_score,
        sample.response,
        tests=tests,
        entrypoint=entrypoint,
        reference_solution=reference_solution or "",
    )
    sample_metadata = getattr(sample, "metadata", None)
    if not isinstance(sample_metadata, dict):
        sample_metadata = {}
    sample_metadata["harbor"] = metadata
    sample.metadata = sample_metadata
    return float(reward)
