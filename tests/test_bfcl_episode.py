from modal_training_gym.common.environments import bfcl
from modal_training_gym.common.environments.base import (
    EvalVerdict,
    Observation,
    StepResult,
    ToolCall,
)
from modal_training_gym.train_recipes.base import BaseTrainRecipe


class _FakeEnvironment:
    def __init__(self) -> None:
        self.actions: list[ToolCall] = []

    def step(self, action: ToolCall) -> StepResult:
        self.actions.append(action)
        return StepResult(observation=Observation(text=f"result:{action.name}"))

    def evaluate(self) -> EvalVerdict:
        return EvalVerdict(passed=True)


def test_run_bfcl_episode_executes_calls_and_appends_observations(monkeypatch) -> None:
    environment = _FakeEnvironment()
    monkeypatch.setattr(bfcl, "build_env", lambda label, start_step: environment)
    monkeypatch.setattr(
        bfcl,
        "build_prefix_messages",
        lambda label, start_step: [{"role": "user", "content": "start"}],
    )
    monkeypatch.setattr(
        bfcl,
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

    result = bfcl.run_bfcl_episode(
        {},
        start_step=2,
        generate=generate,
        parse_response=lambda message: (
            message["content"],
            message["actions"],
        ),
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


def test_run_bfcl_episode_advances_to_later_user_turns(monkeypatch) -> None:
    environment = _FakeEnvironment()
    label = {
        "turns": [
            {"user": "first request", "calls": [{"name": "first"}]},
            {"user": "second request", "calls": [{"name": "second"}]},
        ]
    }
    monkeypatch.setattr(bfcl, "build_env", lambda label, start_step: environment)
    monkeypatch.setattr(
        bfcl,
        "build_prefix_messages",
        lambda label, start_step: [{"role": "user", "content": "first request"}],
    )
    monkeypatch.setattr(bfcl, "tool_schemas_to_openai", lambda schemas: [])

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

    result = bfcl.run_bfcl_episode(
        label,
        start_step=0,
        generate=generate,
        parse_response=lambda message: (
            message["content"],
            message["actions"],
        ),
        max_turns=4,
    )

    assert [action.name for action in environment.actions] == ["first", "second"]
    assert generated_messages[2][-2:] == [
        {"role": "assistant", "content": "first complete"},
        {"role": "user", "content": "second request"},
    ]
    assert result.final_response == "all complete"
    assert result.exit_reason == "no_further_calls"


def test_bfcl_prompt_defers_to_model_tool_format() -> None:
    assert "<emoji>" not in bfcl.DEFAULT_SYSTEM_PROMPT
    assert "provided tool-calling interface" in bfcl.DEFAULT_SYSTEM_PROMPT


def test_bfcl_dataset_paths_are_split_specific() -> None:
    train = bfcl.BfclMultiTurnDataset(split="train")
    evaluation = bfcl.BfclMultiTurnDataset(split="eval")

    train_path = BaseTrainRecipe._resolve_data_path(train, "train")
    eval_path = BaseTrainRecipe._resolve_data_path(evaluation, "eval")

    assert train_path == "/data/BfclMultiTurnDataset/train.jsonl"
    assert eval_path == "/data/BfclMultiTurnDataset/eval.jsonl"
