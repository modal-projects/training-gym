import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from modal_training_gym import TrainConfig
from modal_training_gym.common.training_rollout import (
    _apply_parsed,
    _transcript_messages,
)


def test_transcript_turns_and_empty_final_content():
    raw = (
        '<tool_call>{"name":"bash"}</tool_call><|im_end|>\n'
        "<|im_start|>user\n<tool_response>tests passed</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>Done</think>Fixed.<|im_end|><|endoftext|>"
    )
    messages = _transcript_messages(raw)
    assert messages == [
        {"role": "assistant", "content": '<tool_call>{"name":"bash"}</tool_call>'},
        {"role": "tool", "content": "tests passed"},
        {"role": "assistant", "content": "<think>Done</think>Fixed."},
    ]
    row = {"response": raw, "parsed_response": {"content": "", "thinking": "Done"}}
    _apply_parsed([row])
    assert row["response"] == raw
    assert row["metadata"]["trajectory_messages"] == messages
    assert "thinking" not in row
    supplied = [{"role": "assistant", "content": "structured"}]
    row["metadata"]["trajectory_messages"] = supplied
    _apply_parsed([row])
    assert row["metadata"]["trajectory_messages"] == supplied


@pytest.mark.parametrize(
    "text", ["Plain response", "<|im_start|>assistant\nOnly one turn"]
)
def test_single_turn_is_not_a_transcript(text):
    assert _transcript_messages(text) is None


def test_coding_tutorial_smoke_configuration(monkeypatch, tmp_path):
    import modal

    captured = []
    monkeypatch.setattr(
        modal.Volume,
        "from_name",
        lambda *args, **kwargs: SimpleNamespace(
            listdir=lambda root: [
                SimpleNamespace(path=f"{root}/{name}.jsonl")
                for name in ("train-4", "eval-4")
            ]
        ),
    )

    def launch(config):
        captured.append(config)
        return SimpleNamespace(training_run_id="test", modal_app_url="test")

    monkeypatch.setattr(TrainConfig, "launch", launch)
    runpy.run_path(str(Path(__file__).parents[1] / "tutorials/coding_agent.py"))
    config = captured[0]
    assert config.dataset.path == Path("/data/swe_rebench_v2/train-4.jsonl")
    row = {
        "prompt": [{"role": "user", "content": "Fix the bug"}],
        "label": "task",
        "metadata": {
            "task_path": "swe_rebench_v2/tasks/example",
            "verifier": {"timeout_sec": 120},
        },
    }
    path = tmp_path / "train-4.jsonl"
    path.write_text(json.dumps(row) + "\n\n")
    dataset = type(config.dataset)(path)
    assert list(dataset.rows()) == [row]
    assert dataset.input_key() == "prompt"
    assert dataset.label_key() == "label"
    assert not dataset.apply_chat_template()
    assert dataset.cache_key() is None
    recipe = config.recipe
    assert recipe.gpu_allocation.total_gpus == 2
    assert recipe.gpu_type == "B300"
    assert recipe.num_rollout == 1
    assert recipe.global_batch_size == 2
    assert recipe.extra_config["agentic_max_steps"] == 2
    assert (
        recipe.extra_config["custom_generate_function_path"]
        == "agentic_rl.generate.generate"
    )
    assert (
        recipe.eval_config["datasets"][0]["path"] == "/data/swe_rebench_v2/eval-4.jsonl"
    )
