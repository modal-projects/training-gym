from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from modal_training_gym.common.models import ToolCall
from modal_training_gym.train_recipes.base import BaseTrainRecipe

ENV_PATH = (
    Path(__file__).resolve().parents[1] / "tutorials" / "cross_tok_distill" / "env.py"
)


@pytest.fixture(scope="module")
def bfcl_env():
    spec = importlib.util.spec_from_file_location("cross_tok_distill_env", ENV_PATH)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


class _FakeEnvironment:
    def __init__(self, module) -> None:
        self.module = module
        self.actions: list[ToolCall] = []

    def step(self, action: ToolCall):
        self.actions.append(action)
        return self.module.StepResult(
            observation=self.module.Observation(text=f"result:{action.name}")
        )

    def evaluate(self):
        return self.module.EvalVerdict(passed=True)


def test_run_bfcl_episode_executes_calls_and_appends_observations(
    bfcl_env, monkeypatch
) -> None:
    environment = _FakeEnvironment(bfcl_env)
    monkeypatch.setattr(bfcl_env, "build_env", lambda label, start_step: environment)
    monkeypatch.setattr(
        bfcl_env,
        "build_prefix_messages",
        lambda label, start_step: [{"role": "user", "content": "start"}],
    )
    monkeypatch.setattr(
        bfcl_env,
        "tool_schemas_to_openai",
        lambda schemas: [{"type": "function", "function": {"name": "lookup"}}],
    )
    responses = iter(
        [
            {
                "content": "",
                "actions": [ToolCall(name="lookup", arguments={"key": "value"})],
            },
            {"content": "done", "actions": []},
        ]
    )
    generated_messages: list[list[dict]] = []

    def generate(messages: list[dict], tools: list[dict]) -> dict:
        generated_messages.append(list(messages))
        assert tools[0]["function"]["name"] == "lookup"
        return next(responses)

    result = bfcl_env.run_bfcl_episode(
        {},
        start_step=2,
        generate=generate,
        parse_response=lambda message: (message["content"], message["actions"]),
        max_turns=3,
    )

    assert result.verdict.passed
    assert result.exit_reason == "no_further_calls"
    assert result.final_response == "done"
    assert result.first_call == {"name": "lookup", "arguments": {"key": "value"}}
    assert result.execution_successes == [True]
    assert environment.actions[0].name == "lookup"
    assert generated_messages[1][-2]["tool_calls"][0]["function"]["arguments"] == {
        "key": "value"
    }
    assert generated_messages[1][-1] == {
        "role": "tool",
        "tool_call_id": "call_t0_0",
        "content": "result:lookup",
    }


def test_run_bfcl_episode_advances_to_later_user_turns(bfcl_env, monkeypatch) -> None:
    environment = _FakeEnvironment(bfcl_env)
    label = {
        "turns": [
            {"user": "first request", "calls": [{"name": "first"}]},
            {"user": "second request", "calls": [{"name": "second"}]},
        ]
    }
    monkeypatch.setattr(bfcl_env, "build_env", lambda label, start_step: environment)
    monkeypatch.setattr(
        bfcl_env,
        "build_prefix_messages",
        lambda label, start_step: [{"role": "user", "content": "first request"}],
    )
    monkeypatch.setattr(bfcl_env, "tool_schemas_to_openai", lambda schemas: [])
    responses = iter(
        [
            {"content": "", "actions": [ToolCall(name="first", arguments={})]},
            {"content": "first complete", "actions": []},
            {"content": "", "actions": [ToolCall(name="second", arguments={})]},
            {"content": "all complete", "actions": []},
        ]
    )
    generated_messages: list[list[dict]] = []

    def generate(messages: list[dict], tools: list[dict]) -> dict:
        generated_messages.append(list(messages))
        return next(responses)

    result = bfcl_env.run_bfcl_episode(
        label,
        start_step=0,
        generate=generate,
        parse_response=lambda message: (message["content"], message["actions"]),
        max_turns=4,
    )

    assert [action.name for action in environment.actions] == ["first", "second"]
    assert generated_messages[2][-2:] == [
        {"role": "assistant", "content": "first complete"},
        {"role": "user", "content": "second request"},
    ]
    assert result.final_response == "all complete"
    assert result.exit_reason == "no_further_calls"


def test_bfcl_prompt_defers_to_model_tool_format(bfcl_env) -> None:
    assert "<emoji>" not in bfcl_env.DEFAULT_SYSTEM_PROMPT
    assert "provided tool-calling interface" in bfcl_env.DEFAULT_SYSTEM_PROMPT


def test_bfcl_dataset_paths_are_split_specific(bfcl_env) -> None:
    train = bfcl_env.BfclMultiTurnDataset(split="train")
    evaluation = bfcl_env.BfclMultiTurnDataset(split="eval")

    train_path, _ = BaseTrainRecipe._resolve_data_paths(train)
    eval_path, _ = BaseTrainRecipe._resolve_data_paths(evaluation)

    assert train_path == "/data/BfclMultiTurnDataset/train.jsonl"
    assert eval_path == "/data/BfclMultiTurnDataset/eval.jsonl"
