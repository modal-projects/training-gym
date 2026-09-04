"""Flattened multi-turn agent transcripts render turn by turn on the dashboard."""

from __future__ import annotations

from modal_training_gym.common.sample_extraction import _sample_to_dict
from modal_training_gym.common.training_rollout import (
    _apply_parsed,
    _transcript_messages,
)

# The shape the agentic Slime fork writes: the first assistant turn opens the
# string, later turns are appended verbatim with chat-template tokens.
TRANSCRIPT = (
    "<think>\nExplore first.\n</think>\n\nI'll look around.\n\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\nls /app\n</parameter>\n"
    "</function>\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\n<returncode>0</returncode>\n<output>\ncel\nREADME.md\n"
    "</output>\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n<think>\nNow read the README.\n</think>\n\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\ncat /app/README.md\n</parameter>\n"
    "</function>\n</tool_call><|im_end|>"
)


def test_transcript_splits_into_role_tagged_turns() -> None:
    messages = _transcript_messages(TRANSCRIPT)

    assert messages is not None
    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert messages[0]["content"].startswith("<think>\nExplore first.")
    assert "<|im_end|>" not in messages[0]["content"]
    # The environment's reply is unwrapped from its <tool_response> envelope.
    assert (
        messages[1]["content"]
        == "<returncode>0</returncode>\n<output>\ncel\nREADME.md\n</output>"
    )
    assert messages[2]["content"].startswith("<think>\nNow read the README.")


def test_single_turn_response_is_not_a_transcript() -> None:
    assert _transcript_messages("<think>hmm</think>The answer is 4.") is None
    assert _transcript_messages("") is None


def test_display_never_blanks_a_transcript_whose_last_turn_is_a_tool_call() -> None:
    # What the recorder stores: the model parser saw only the final turn, whose
    # text is empty once its tool call is stripped.
    row = {
        "response": TRANSCRIPT,
        "parsed_response": {
            "content": "",
            "thinking": "Now read the README.",
            "tool_calls": [
                {"name": "bash", "arguments": {"command": "cat /app/README.md"}}
            ],
        },
        "metadata": {"instance_id": "task-1"},
    }

    _apply_parsed([row])

    assert row["response"] == TRANSCRIPT
    assert row["raw_response"] == TRANSCRIPT
    assert [m["role"] for m in row["metadata"]["trajectory_messages"]] == [
        "assistant",
        "tool",
        "assistant",
    ]
    # Per-turn thinking lives in the trajectory; no stray top-level block.
    assert "thinking" not in row
    assert row["tool_calls"][0]["name"] == "bash"
    assert row["metadata"]["instance_id"] == "task-1"


def test_display_keeps_a_hook_supplied_trajectory() -> None:
    supplied = [{"role": "assistant", "content": "structured"}]
    row = {"response": TRANSCRIPT, "metadata": {"trajectory_messages": supplied}}

    _apply_parsed([row])

    assert row["metadata"]["trajectory_messages"] is supplied


def test_single_turn_display_is_unchanged() -> None:
    row = {
        "response": "<think>secret</think>raw answer",
        "parsed_response": {"content": "raw answer", "thinking": "secret"},
    }

    _apply_parsed([row])

    assert row["response"] == "raw answer"
    assert row["raw_response"] == "<think>secret</think>raw answer"
    assert row["thinking"] == "secret"
    assert "trajectory_messages" not in row.get("metadata", {})


def test_oversized_metadata_tag_keeps_its_small_entries() -> None:
    sample = {
        "prompt": "p",
        "response": "r",
        "reward": 0.0,
        "metadata": {
            "agentic": {
                "turns": 2,
                "is_solved": False,
                "timing": {"boot": 9.2, "verifier": 8.2},
                "full_prompt": "x" * 5000,
            },
            "note": "y" * 5000,
        },
    }

    out = _sample_to_dict(sample)

    assert out["metadata"]["agentic"] == {
        "turns": 2,
        "is_solved": False,
        "timing": {"boot": 9.2, "verifier": 8.2},
    }
    assert "note" not in out["metadata"]
