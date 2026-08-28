from __future__ import annotations

import asyncio
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

from modal_training_gym.common.dataset import HarborDataset
from modal_training_gym.common.eval import HarborEval
from modal_training_gym.common.harbor import (
    TrainingGymResponseAgent,
    extract_harbor_candidate,
    harbor_reward,
    score_harbor_response,
)


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _install_fake_harbor(
    monkeypatch,
    *,
    reward: float = 1.0,
    exception_info=None,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    class EnvironmentType(str, Enum):
        MODAL = "modal"

    class ResourceMode(str, Enum):
        GUARANTEE = "guarantee"
        LIMIT = "limit"
        IGNORE = "ignore"

    class Trial:
        @classmethod
        async def create(cls, config):
            captured["config"] = config
            return cls()

        async def run(self):
            return SimpleNamespace(
                task_name="task-a",
                trial_name="trial-a",
                verifier_result=SimpleNamespace(rewards={"reward": reward}),
                agent_result=SimpleNamespace(metadata={"candidate_return_code": 0}),
                exception_info=exception_info,
            )

    modules = {
        "harbor": ModuleType("harbor"),
        "harbor.models": ModuleType("harbor.models"),
        "harbor.models.environment_type": ModuleType("harbor.models.environment_type"),
        "harbor.models.trial": ModuleType("harbor.models.trial"),
        "harbor.models.trial.config": ModuleType("harbor.models.trial.config"),
        "harbor.trial": ModuleType("harbor.trial"),
        "harbor.trial.trial": ModuleType("harbor.trial.trial"),
    }
    modules["harbor.models.environment_type"].EnvironmentType = EnvironmentType
    modules["harbor.models.trial.config"].ResourceMode = ResourceMode
    for name in (
        "AgentConfig",
        "EnvironmentConfig",
        "TaskConfig",
        "TrialConfig",
        "VerifierConfig",
    ):
        setattr(modules["harbor.models.trial.config"], name, _Config)
    modules["harbor.trial.trial"].Trial = Trial
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return captured


def test_score_harbor_response_builds_modal_trial(tmp_path: Path, monkeypatch) -> None:
    captured = _install_fake_harbor(monkeypatch)

    score, metadata = asyncio.run(
        score_harbor_response(
            "print('candidate')",
            task_path=tmp_path,
            candidate_path="/tmp/candidate file.py",
            candidate_command="python {candidate_path}",
            timeout_sec=42,
            sandbox_cpu=2,
            sandbox_memory=2048,
            cpu_policy="reserve",
            memory_policy="limit",
        )
    )

    assert score == 1.0
    assert metadata["harbor_rewards"] == {"reward": 1.0}
    config = captured["config"]
    assert config.trials_dir.name.startswith("training-gym-harbor-trial-")
    assert not config.trials_dir.exists()
    assert config.environment.type.value == "modal"
    assert config.environment.override_cpus == 2
    assert config.environment.override_memory_mb == 2048
    assert config.environment.kwargs == {
        "sandbox_timeout_secs": 300,
        "sandbox_idle_timeout_secs": 120,
    }
    assert config.agent.kwargs["response"] == "print('candidate')"
    assert config.agent.kwargs["candidate_path"] == "/tmp/candidate file.py"
    assert config.verifier.override_timeout_sec == 42
    assert config.agent.import_path == (
        "modal_training_gym.common.harbor:TrainingGymResponseAgent"
    )


def test_score_harbor_response_surfaces_trial_failure(
    tmp_path: Path, monkeypatch
) -> None:
    exception_info = SimpleNamespace(
        exception_type="RuntimeError",
        exception_message="sandbox upload failed",
        exception_traceback="trace",
    )
    _install_fake_harbor(
        monkeypatch,
        reward=0.0,
        exception_info=exception_info,
    )

    score, metadata = asyncio.run(
        score_harbor_response("candidate", task_path=tmp_path)
    )

    assert score == 0.0
    assert metadata["harbor_error"] == {
        "type": "RuntimeError",
        "message": "sandbox upload failed",
    }


def test_training_gym_agent_uploads_before_execution(tmp_path: Path) -> None:
    calls: list[tuple[object, ...]] = []

    class Environment:
        async def upload_file(self, source, target):
            calls.append(("upload", Path(source).read_text(), target))

        async def exec(self, command):
            calls.append(("exec", command))
            return SimpleNamespace(return_code=0, stdout="ok", stderr="")

    context = SimpleNamespace(metadata=None)
    agent = TrainingGymResponseAgent(
        tmp_path,
        response="print(1)",
        candidate_path="/tmp/candidate file.py",
        candidate_command="python {candidate_path}",
    )

    asyncio.run(agent.run("", Environment(), context))

    assert calls == [
        ("upload", "print(1)", "/tmp/candidate file.py"),
        ("exec", "python '/tmp/candidate file.py'"),
    ]
    assert context.metadata["candidate_return_code"] == 0


def test_harbor_eval_uses_trial_scorer(tmp_path: Path, monkeypatch) -> None:
    import modal_training_gym.common.harbor as harbor

    captured: dict[str, object] = {}

    async def fake_score(response, **kwargs):
        captured["response"] = response
        captured.update(kwargs)
        return 0.75, {"harbor_task_name": "task-a"}

    monkeypatch.setattr(harbor, "resolve_harbor_task_path", lambda label: tmp_path)
    monkeypatch.setattr(harbor, "score_harbor_response", fake_score)

    class Deployment:
        def generate(self, prompt, **kwargs):
            return "print('candidate')"

    evaluation = HarborEval(
        dataset=HarborDataset(path=str(tmp_path)),
        extract_code_fn=lambda response: response,
    )
    result = evaluation._harbor_eval_fn(
        Deployment(),
        {
            "prompt": "Write the candidate.",
            "label": {
                "harbor_task_data_rel": "HarborDataset/harbor_tasks/source/task-a",
                "harbor_candidate_path": "/tmp/candidate.py",
                "harbor_candidate_command": "python {candidate_path}",
            },
        },
    )

    assert result.score == 0.75
    assert result.metadata == {"harbor_task_name": "task-a"}
    assert captured["response"] == "print('candidate')"
    assert captured["task_path"] == tmp_path
    assert captured["candidate_path"] == "/tmp/candidate.py"
    assert captured["timeout_sec"] == 60


def test_extract_harbor_candidate_accepts_flexible_fences() -> None:
    assert extract_harbor_candidate("```py\nprint(1)\n```") == "print(1)"
    assert extract_harbor_candidate("```\nprint(2)\n```") == "print(2)"
    assert (
        extract_harbor_candidate("<think>scratch</think>\n```python\nprint(3)\n```")
        == "print(3)"
    )


def test_harbor_eval_default_extracts_py_fence(tmp_path: Path, monkeypatch) -> None:
    import modal_training_gym.common.harbor as harbor

    captured: dict[str, object] = {}

    async def fake_score(response, **kwargs):
        captured["response"] = response
        return 1.0, {}

    monkeypatch.setattr(harbor, "resolve_harbor_task_path", lambda label: tmp_path)
    monkeypatch.setattr(harbor, "score_harbor_response", fake_score)

    class Deployment:
        def generate(self, prompt, **kwargs):
            return "```py\nprint('candidate')\n```"

    result = HarborEval(dataset=HarborDataset(path=str(tmp_path)))._harbor_eval_fn(
        Deployment(),
        {
            "prompt": "Write the candidate.",
            "label": {
                "harbor_task_data_rel": "HarborDataset/harbor_tasks/source/task-a"
            },
        },
    )

    assert result.score == 1.0
    assert captured["response"] == "print('candidate')"


def test_harbor_reward_uses_shared_extract(monkeypatch) -> None:
    import modal_training_gym.common.harbor as harbor

    captured: dict[str, object] = {}

    async def fake_score(response, label, **kwargs):
        captured["response"] = response
        return 1.0, {"harbor_task_name": "task-a"}

    monkeypatch.setattr(harbor, "score_from_label", fake_score)
    sample = SimpleNamespace(response="```py\nprint(1)\n```", label="{}", metadata=None)

    assert asyncio.run(harbor_reward(None, sample)) == 1.0
    assert captured["response"] == "print(1)"
    assert sample.metadata == {"harbor": {"harbor_task_name": "task-a"}}


def test_live_harbor_import_is_real_package() -> None:
    import harbor
    import harbor.models.environment_type
    import harbor.models.trial.config
    import harbor.trial.trial

    assert harbor.models.trial.config.TrialConfig.__module__.startswith("harbor.")
    assert harbor.trial.trial.Trial.__module__.startswith("harbor.")
