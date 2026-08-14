"""The stitch trainer's argv, where it deviates from the miles defaults."""

import dataclasses

from modal_training_gym.common.status import MilesStatus, resolve_framework_status
from modal_training_gym.train_recipes.stitch_recipe import Qwen3_30B_A3B_Stitch_Train


def _fields(**overrides) -> dict:
    return dataclasses.replace(Qwen3_30B_A3B_Stitch_Train(), **overrides)._fields()


def test_no_save_path_without_a_save_interval() -> None:
    """Megatron asserts ``save_interval is not None`` whenever ``save`` is set,
    and this recipe keeps no trainer checkpoints by default."""
    fields = _fields()
    assert fields.get("save_interval") is None
    assert "save" not in fields
    assert "save_hf" not in fields


def test_save_path_survives_a_save_interval() -> None:
    fields = _fields(save_interval=10)
    assert fields["save_interval"] == 10
    assert fields["save"]


def test_stitch_reports_miles_phases() -> None:
    """The trainer is miles, so the dashboard has to accept its phase names for a
    run recorded as stitch."""
    assert (
        resolve_framework_status("generate_rollouts", "stitch")
        is MilesStatus.ROLLOUT_LOGGING
    )


def test_rollout_gating_matches_the_cookbook_config() -> None:
    """An exact pin races the TTL-cached pointer it is computed from: a request
    pinned to version N that lands on a replica already at N+1 can never be
    served. The ported cookbook config floors instead."""
    train = Qwen3_30B_A3B_Stitch_Train()
    assert train.rollout_request_weight_version_mode == "min"
    assert train.rollout_request_weight_version_lag == 1
