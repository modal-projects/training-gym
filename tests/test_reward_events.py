"""First-class transition reward extraction and persistence shape."""

from types import SimpleNamespace

import pytest

from modal_training_gym.common.sample import RewardEvent, Sample
from modal_training_gym.common.sample_extraction import _sample_to_dict
from modal_training_gym.common.token_rewards import build_token_reward_vectors


def _event(turn: int) -> dict[str, object]:
    return {
        "turn": turn,
        "state": "SELECTING_HAND",
        "progress": 0.01,
        "blind_clear_bonus": 0.0,
        "legal_action_bonus": 0.1,
        "invalid_penalty": 0.0,
        "total": 0.11,
    }


def test_large_legacy_balatro_trace_is_promoted_to_first_class_events():
    raw = SimpleNamespace(
        prompt="p",
        response="r",
        reward=10.0,
        metadata={
            "balatro": {
                "partial_reward_trace": [_event(turn) for turn in range(96)],
            },
            "small_tag": "kept",
        },
    )

    recorded = _sample_to_dict(raw)

    assert "balatro" not in recorded["metadata"]
    assert recorded["metadata"]["small_tag"] == "kept"
    assert recorded["reward_granularity"] == "transition"
    assert len(recorded["reward_events"]) == 96
    assert recorded["reward_events"][0] == {
        "turn": 0,
        "reward": 0.11,
        "cumulative_reward": 0.11,
        "label": "SELECTING_HAND",
        "components": {
            "progress": 0.01,
            "blind_clear_bonus": 0.0,
            "legal_action_bonus": 0.1,
            "invalid_penalty": 0.0,
        },
    }
    assert recorded["reward_events"][-1]["cumulative_reward"] == pytest.approx(
        0.11 * 96
    )


def test_first_class_events_keep_token_offsets_and_round_trip():
    raw = SimpleNamespace(
        prompt="p",
        response="r",
        reward=1.0,
        reward_granularity="transition",
        reward_events=[
            {
                "turn": 3,
                "reward": 0.25,
                "cumulative_reward": 0.25,
                "token_start": 12,
                "token_end": 18,
                "components": {"progress": 0.25},
            }
        ],
    )

    recorded = _sample_to_dict(raw)
    sample = Sample.model_validate(recorded)

    assert sample.reward_granularity == "transition"
    assert sample.reward_events == [
        RewardEvent(
            turn=3,
            reward=0.25,
            cumulative_reward=0.25,
            token_start=12,
            token_end=18,
            components={"progress": 0.25},
        )
    ]


def test_reward_events_are_bounded_individually():
    raw = SimpleNamespace(
        prompt="p",
        response="r",
        reward=1.0,
        reward_events=[_event(turn) for turn in range(600)],
    )

    recorded = _sample_to_dict(raw)

    assert len(recorded["reward_events"]) == 512


def test_transition_events_become_token_rewards_and_preserve_scalar_total():
    sample = SimpleNamespace(
        response_length=8,
        loss_mask=[1] * 8,
        reward=9.0,
        metadata={
            "reward_events": [
                {"reward": 2.0, "token_start": 0, "token_end": 2},
                {"reward": 3.0, "token_start": 4, "token_end": 6},
            ]
        },
    )

    vectors = build_token_reward_vectors([sample], [9.0])

    assert vectors == [[0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0]]
    assert sum(vectors[0]) == pytest.approx(9.0)


def test_samples_without_events_keep_scalar_fallback():
    sample = SimpleNamespace(
        response_length=3,
        loss_mask=[1, 1, 1],
        reward=2.0,
        metadata={},
    )

    assert build_token_reward_vectors([sample], [2.0]) is None


def test_token_advantages_are_return_to_go_with_scalar_mixed_batch():
    torch = pytest.importorskip("torch")
    from modal_training_gym.frameworks.slime.token_reward_advantages import (
        compute_token_reward_advantages,
    )

    data = {
        "kl": [torch.zeros(4), torch.zeros(3)],
        "rewards": [3.0, 5.0],
        "token_rewards": [[1.0, 0.0, 2.0, 0.0], None],
        "response_lengths": [4, 3],
        "total_lengths": [4, 3],
        "loss_masks": [torch.ones(4), torch.ones(3)],
    }

    compute_token_reward_advantages(
        SimpleNamespace(
            advantage_estimator="grpo",
            training_gym_token_reward_gamma=1.0,
        ),
        data,
    )

    assert data["advantages"][0].tolist() == [3.0, 2.0, 2.0, 0.0]
    assert data["advantages"][1].tolist() == [5.0, 5.0, 5.0]
