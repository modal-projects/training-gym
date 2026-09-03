from __future__ import annotations

from modal_training_gym.common.dataset import OnlineRollout


def test_online_rollout_rows() -> None:
    dataset = OnlineRollout(n_rows=4)
    assert list(dataset.rows()) == [
        {"prompt": "", "label": "0"},
        {"prompt": "", "label": "1"},
        {"prompt": "", "label": "2"},
        {"prompt": "", "label": "3"},
    ]
