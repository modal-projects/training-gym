"""The stitch trainer's argv, where it deviates from the miles defaults."""

import dataclasses
import inspect

from modal_training_gym.common.status import MilesStatus, resolve_framework_status
from modal_training_gym.frameworks.stitch import launcher as stitch_launcher
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


def test_trainer_exports_the_dashboard_reporting_env() -> None:
    """miles' phase/rollout hooks run in Ray actors that read their run identity
    from the environment. The colocated launcher hands it over as a Ray
    runtime_env; this one runs miles as a subprocess, so it has to export the
    same names or every report is dropped and the dashboard stalls at launch."""
    source = inspect.getsource(stitch_launcher)
    for name in (
        "TRAINING_GYM_TRAINING_RUN_ID",
        "TRAINING_GYM_APP_NAME",
        "TRAINING_GYM_TOTAL_STEPS",
        "TRAINING_GYM_RESPONSE_PARSER_PATH",
        "TRAINING_GYM_FRAMEWORK_STATUS_URL",
        "TRAINING_GYM_FRAMEWORK_STATUS_TOKEN",
    ):
        assert name in source
