from __future__ import annotations

import asyncio
import importlib
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from modal_training_gym.common.dataset import HarborDataset
from modal_training_gym.common.eval import HarborEval
from modal_training_gym.common.harbor import (
    extract_harbor_candidate,
    score_harbor_response,
)
from modal_training_gym.common.models import Qwen3_5_4B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

_SLIME_RECIPE_KWARGS = {
    "gpu_type": "H100",
    "colocate": True,
    "tensor_model_parallel_size": 1,
    "sequence_parallel": False,
    "rollout_num_gpus_per_engine": 1,
    "num_rollout": 1,
    "rollout_batch_size": 16,
    "rollout_max_response_len": 4096,
    "rollout_temperature": 1.0,
    "save_interval": 10,
}


class _Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _install_fake_harbor(
    monkeypatch: pytest.MonkeyPatch,
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


def test_score_harbor_response_builds_modal_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_score_harbor_response_surfaces_trial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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


def test_training_gym_agent_uploads_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BaseAgent:
        def __init__(self, logs_dir, model_name=None, **kwargs):
            self.logs_dir = logs_dir

    harbor_agents = ModuleType("harbor.agents")
    harbor_agents_base = ModuleType("harbor.agents.base")
    harbor_agents_base.BaseAgent = BaseAgent
    monkeypatch.setitem(sys.modules, "harbor.agents", harbor_agents)
    monkeypatch.setitem(sys.modules, "harbor.agents.base", harbor_agents_base)
    monkeypatch.delitem(
        sys.modules,
        "modal_training_gym.common.harbor_agent",
        raising=False,
    )
    module = importlib.import_module("modal_training_gym.common.harbor_agent")

    calls: list[tuple[object, ...]] = []

    class Environment:
        async def upload_file(self, source, target):
            calls.append(("upload", Path(source).read_text(), target))

        async def exec(self, command):
            calls.append(("exec", command))
            return SimpleNamespace(return_code=0, stdout="ok", stderr="")

    context = SimpleNamespace(metadata=None)
    agent = module.TrainingGymResponseAgent(
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


def test_extract_harbor_candidate_strips_chat_markup() -> None:
    response = (
        "<|im_start|>assistant\n<think>reasoning</think>\n"
        "```python\nprint('candidate')\n```\n<|im_end|>"
    )

    assert extract_harbor_candidate(response) == "print('candidate')"


def test_harbor_eval_uses_trial_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from modal_training_gym.common import harbor

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
                "test_cases": [{"input": "", "expected": "legacy"}],
            },
        },
    )

    assert result.score == 0.75
    assert result.metadata == {"harbor_task_name": "task-a"}
    assert captured["response"] == "print('candidate')"
    assert captured["task_path"] == tmp_path
    assert captured["candidate_path"] == "/tmp/candidate.py"


def test_harbor_eval_preserves_inline_test_case_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from modal_training_gym.common import eval as eval_module

    captured: dict[str, object] = {}

    def fake_score(response, **kwargs):
        captured["response"] = response
        captured.update(kwargs)
        return 1.0, {"test_cases": 1}

    monkeypatch.setattr(eval_module, "score_in_sandbox", fake_score)

    class Deployment:
        def generate(self, prompt, **kwargs):
            return "print('candidate')"

    evaluation = HarborEval(
        dataset=HarborDataset(path=str(tmp_path)),
        extract_code_fn=lambda response: response,
    )
    test_cases = [{"input": "", "expected": "candidate"}]
    result = evaluation._harbor_eval_fn(
        Deployment(),
        {
            "prompt": "Write the candidate.",
            "label": {
                "harbor_task_path": str(tmp_path / "task-a"),
                "test_cases": test_cases,
            },
        },
    )

    assert result.score == 1.0
    assert result.metadata == {"test_cases": 1}
    assert captured["response"] == "print('candidate')"
    assert captured["test_cases"] == test_cases


@pytest.mark.parametrize(
    ("recipe_type", "recipe_kwargs"),
    [(MilesRecipe, {}), (SlimeRecipe, _SLIME_RECIPE_KWARGS)],
)
def test_harbor_dataset_supplies_default_reward_function(
    tmp_path: Path,
    recipe_type,
    recipe_kwargs,
) -> None:
    dataset = HarborDataset(path=str(tmp_path))
    config = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=dataset,
        recipe=recipe_type(**recipe_kwargs),
        merge_model_recipe=False,
    )

    prepared = config._prepare_recipe()

    assert prepared.custom_rm_function is dataset.reward_function


def test_harbor_reward_replaces_model_preset_reward_type(tmp_path: Path) -> None:
    dataset = HarborDataset(path=str(tmp_path))
    config = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=dataset,
        recipe=MilesRecipe(),
    )

    prepared = config._prepare_recipe()

    assert prepared.custom_rm_function is dataset.reward_function
    assert prepared.rm_type is None


@pytest.mark.parametrize(
    ("recipe_type", "recipe_kwargs"),
    [(MilesRecipe, {}), (SlimeRecipe, _SLIME_RECIPE_KWARGS)],
)
def test_explicit_reward_function_takes_precedence(
    tmp_path: Path,
    recipe_type,
    recipe_kwargs,
) -> None:
    def custom_reward(*args, **kwargs):
        return 0.5

    dataset = HarborDataset(path=str(tmp_path))
    config = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=dataset,
        recipe=recipe_type(
            custom_rm_function=custom_reward,
            **recipe_kwargs,
        ),
        merge_model_recipe=False,
    )

    assert config._prepare_recipe().custom_rm_function is custom_reward


@pytest.mark.parametrize(
    ("recipe_type", "recipe_kwargs"),
    [(MilesRecipe, {}), (SlimeRecipe, _SLIME_RECIPE_KWARGS)],
)
def test_explicit_reward_type_takes_precedence(
    tmp_path: Path,
    recipe_type,
    recipe_kwargs,
) -> None:
    dataset = HarborDataset(path=str(tmp_path))
    config = TrainConfig(
        model=Qwen3_5_4B(),
        dataset=dataset,
        recipe=recipe_type(
            rm_type="deepscaler",
            **recipe_kwargs,
        ),
        merge_model_recipe=False,
    )

    prepared = config._prepare_recipe()

    assert prepared.custom_rm_function is None
    assert prepared.rm_type == "deepscaler"
