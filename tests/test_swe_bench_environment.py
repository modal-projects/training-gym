from __future__ import annotations

import io
import json
import sys
from types import ModuleType

import pytest

from tutorials.swe_bench import env, generate, qwen3_model


@pytest.fixture
def fake_miniswe(monkeypatch: pytest.MonkeyPatch):
    minisweagent = ModuleType("minisweagent")
    agents = ModuleType("minisweagent.agents")
    default = ModuleType("minisweagent.agents.default")

    class DefaultAgent:
        result = {"exit_status": "Submitted", "submission": "patch"}

        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, **kwargs):
            return dict(self.result)

    default.DefaultAgent = DefaultAgent
    for name, module in {
        "minisweagent": minisweagent,
        "minisweagent.agents": agents,
        "minisweagent.agents.default": default,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return DefaultAgent


class _Tokenizer:
    eos_token_id = 0

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        return 1

    @staticmethod
    def apply_chat_template(messages, **kwargs) -> str:
        return "prompt"

    @staticmethod
    def encode(text: str, **kwargs) -> list[int]:
        return list(range(len(text)))


class _Environment:
    boot_time = 0.1
    exec_time = 0.2
    exec_timeouts = 0
    deadline = None

    def capture_patch(self) -> str:
        return ""

    def close(self) -> None:
        pass

    def get_template_vars(self) -> dict[str, str]:
        return {"cwd": "/testbed"}


_LIMITS = {
    "query_timeout": 1,
    "max_context_len": 1024,
    "episode_timeout": 1,
    "exec_timeout": 1,
    "max_steps": 1,
    "grade_timeout": 1,
}


def _grading_task(parser: str = "parse_log_pytest") -> dict:
    return {
        "instance_id": "task",
        "image_name": "image",
        "repo": "org/repo",
        "workdir": "/repo",
        "problem_statement": "fix",
        "install_config": {"log_parser": parser, "test_cmd": "pytest -rA"},
        "test_patch": "+++ b/tests/test_fix.py\n",
        "FAIL_TO_PASS": ["tests/test_fix.py::test_fix"],
        "PASS_TO_PASS": [],
    }


class _Grader:
    def __init__(self, output: str = "PASSED tests/test_fix.py::test_fix") -> None:
        self.output = output
        self.commands: list[str] = []

    def write_file(self, path: str, content: str) -> None:
        pass

    def execute_bash(self, command: str, *, timeout: int) -> tuple[int, str]:
        self.commands.append(command)
        return 0, self.output

    def close(self) -> None:
        pass


def _run_episode(monkeypatch: pytest.MonkeyPatch, grade):
    monkeypatch.setattr(
        generate.SweEnvironment, "create", lambda *args, **kwargs: _Environment()
    )
    monkeypatch.setattr(generate, "grade_swe_patch", grade)
    return generate._run_episode(
        {"instance_id": "task", "problem_statement": "fix"},
        _Tokenizer(),
        {},
        "http://router",
        _LIMITS,
        "session",
    )


@pytest.mark.parametrize(
    ("parser", "output", "required"),
    [
        (
            "parse_log_pytest",
            "PASSED tests/test_fix.py::test_fix",
            "tests/test_fix.py::test_fix",
        ),
        (
            "parse_log_pytest_options",
            "PASSED tests/test_fix.py::test_fix[/tmp/case.json]",
            "tests/test_fix.py::test_fix[/case.json]",
        ),
        (
            "parse_log_pytest_v2",
            "tests/test_fix.py::test_fix PASSED",
            "tests/test_fix.py::test_fix",
        ),
    ],
)
def test_grader_dispatches_configured_log_parser(
    monkeypatch: pytest.MonkeyPatch, parser: str, output: str, required: str
) -> None:
    grader = _Grader(output)
    monkeypatch.setattr(env.SweEnvironment, "create", lambda *args, **kwargs: grader)
    task = _grading_task(parser)
    task["FAIL_TO_PASS"] = [required]

    verdict = env.grade_swe_patch(task, "diff --git a/x b/x")

    assert verdict.passed


def test_grader_restores_agent_controlled_pytest_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grader = _Grader()
    monkeypatch.setattr(env.SweEnvironment, "create", lambda *args, **kwargs: grader)
    model_patch = "\n".join(
        [
            "diff --git a/conftest.py b/conftest.py",
            "--- /dev/null",
            "+++ b/conftest.py",
            "diff --git a/pyproject.toml b/pyproject.toml",
            "--- a/pyproject.toml",
            "+++ b/pyproject.toml",
        ]
    )

    verdict = env.grade_swe_patch(_grading_task(), model_patch)

    assert verdict.passed
    script = grader.commands[0]
    assert script.index("git apply") < script.index("git checkout HEAD -- conftest.py")
    assert "git checkout HEAD -- pyproject.toml" in script
    assert script.index("git checkout HEAD") < script.rindex("git apply")


def test_grader_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    # Apply/setup failure: `set -e` aborts the script, no PASSED lines → terminal reward 0.
    grader = _Grader("error: patch failed: x:1")
    monkeypatch.setattr(env.SweEnvironment, "create", lambda *args, **kwargs: grader)
    verdict = env.grade_swe_patch(_grading_task(), "diff --git a/x b/x")
    assert verdict.passed is False
    assert verdict.harness_error is False

    # Grader infrastructure failure is a harness error, not a model failure.
    def boom(*args, **kwargs):
        raise TimeoutError("sandbox boot")

    monkeypatch.setattr(env.SweEnvironment, "create", boom)
    verdict = env.grade_swe_patch(_grading_task(), "diff --git a/x b/x")
    assert verdict.passed is False
    assert verdict.harness_error is True


def test_weight_sync_abort_reissues_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_call = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
    responses = [
        {
            "text": "par",
            "meta_info": {
                "finish_reason": {"type": "abort"},
                "output_token_logprobs": [(-0.5, 9)],
            },
        },
        {
            "text": "",
            "meta_info": {
                "finish_reason": {"type": "abort"},
                "output_token_logprobs": [],
            },
        },
        {
            "text": tool_call,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [(-0.1, 7), (-0.2, 8)],
            },
        },
    ]
    payloads: list[dict] = []

    def urlopen(req, timeout):
        payloads.append(json.loads(req.data))
        return io.BytesIO(json.dumps(responses.pop(0)).encode())

    monkeypatch.setattr(qwen3_model.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(qwen3_model.time, "sleep", lambda s: None)
    exceptions = ModuleType("minisweagent.exceptions")
    exceptions.FormatError = type("FormatError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "minisweagent.exceptions", exceptions)
    model = qwen3_model.Qwen3RecordingModel(
        _Tokenizer(), {}, "http://router", "{output}", "session", max_context_len=8192
    )

    out = model.query([{"role": "user", "content": "fix"}])

    prompt_ids = _Tokenizer.encode("prompt")
    assert [p["input_ids"] for p in payloads] == [prompt_ids] * 3
    assert model.tokens == prompt_ids + [7, 8]
    assert model.logprobs[len(prompt_ids) :] == [-0.1, -0.2]
    assert model.resumed_turns == 2
    assert model.aborted is False
    assert out["extra"]["actions"] == [{"command": "ls"}]


def test_grading_harness_failure_is_retryable(
    fake_miniswe, monkeypatch: pytest.MonkeyPatch
) -> None:
    reward, _, stats = _run_episode(
        monkeypatch,
        lambda *args, **kwargs: env.EvalVerdict(passed=False, harness_error=True),
    )
    assert reward == 0
    assert stats["harness_error"] is True

    def boom(*args, **kwargs):
        raise RuntimeError("grader crashed")

    reward, _, stats = _run_episode(monkeypatch, boom)
    assert reward == 0
    assert stats["harness_error"] is True
