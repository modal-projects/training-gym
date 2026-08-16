"""The stitch trainer's argv, where it deviates from the miles defaults."""

import dataclasses

import pytest
from pydantic import ValidationError

from modal_training_gym.common.models.qwen3_30b import Qwen3_30B
from modal_training_gym.common.status import MilesStatus, resolve_framework_status
from modal_training_gym.deploy_recipes.sglang_recipe.recipe import SglangRecipe
from modal_training_gym.frameworks.miles.modal_helpers.patches import PATCHES_DIR
from modal_training_gym.frameworks.stitch import launcher as stitch_launcher
from modal_training_gym.train_recipes.stitch_recipe import (
    Qwen3_30B_A3B_Stitch_Recipe,
    Qwen3_30B_A3B_Stitch_Train,
    StitchRecipe,
    StitchServeConfig,
    StitchTrainConfig,
)
from modal_training_gym.train_recipes.stitch_recipe.pins import MILES_ROOT


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


@pytest.mark.parametrize(
    "field",
    [
        "custom_rm_function",
        "custom_generate_function",
        "custom_reward_post_process_function",
        "custom_rollout_log_function",
        "custom_eval_rollout_log_function",
        "rollout_function",
        "custom_megatron_before_log_prob_hook",
        "custom_megatron_before_train_step_hook",
    ],
)
def test_caller_side_hooks_are_rejected(field: str) -> None:
    with pytest.raises(
        ValidationError,
        match=rf"{field}.*does not ship caller-side hooks|does not ship caller-side hooks.*{field}",
    ):
        StitchTrainConfig(**{field: "caller_module.hook"})


def test_default_stitch_configs_keep_gym_reporting_hooks() -> None:
    expected = {
        "custom_rollout_log_function_path": (
            "modal_training_gym.frameworks.miles.phase_reporting.log_rollout_data"
        ),
        "custom_eval_rollout_log_function_path": (
            "modal_training_gym.frameworks.miles.phase_reporting.log_eval_rollout_data"
        ),
        "custom_megatron_before_log_prob_hook_path": (
            "modal_training_gym.frameworks.miles.phase_reporting.before_log_prob_hook"
        ),
        "custom_megatron_before_train_step_hook_path": (
            "modal_training_gym.frameworks.miles.phase_reporting.before_train_step_hook"
        ),
    }

    for train in (StitchTrainConfig(), Qwen3_30B_A3B_Stitch_Train()):
        fields = train._fields()
        assert {key: fields[key] for key in expected} == expected


def test_stitch_startup_timeout_is_owned_by_the_pool() -> None:
    assert StitchServeConfig().startup_timeout == 60 * 60
    assert StitchServeConfig(startup_timeout=123).startup_timeout == 123

    with pytest.raises(ValidationError, match="StitchServeConfig.startup_timeout"):
        StitchServeConfig(sglang=SglangRecipe(startup_timeout=123))


def test_trainer_exports_the_dashboard_reporting_env() -> None:
    """miles' phase/rollout hooks run in Ray actors that read their run identity
    from the environment. The colocated launcher hands it over as a Ray
    runtime_env; this one runs miles as a subprocess, so it has to export the
    same values or every report is dropped and the dashboard stalls at launch."""
    env = stitch_launcher.dashboard_env(
        training_run_id="tr-123",
        app_name="stitch-app",
        total_steps=7,
        model=Qwen3_30B(),
    )
    assert env == {
        "TRAINING_GYM_TRAINING_RUN_ID": "tr-123",
        "TRAINING_GYM_APP_NAME": "stitch-app",
        "TRAINING_GYM_TOTAL_STEPS": "7",
        "TRAINING_GYM_RESPONSE_PARSER_PATH": (
            "modal_training_gym.common.models.base.parse_qwen3_response"
        ),
    }


def test_get_base_recipe_pairs_the_model_like_the_other_frameworks() -> None:
    """The validation harness and diff_impact index every framework the same
    way, so a stitch model is looked up through the recipe, not a private map in
    the validation backend."""
    assert isinstance(StitchRecipe.get_base_recipe(Qwen3_30B()), StitchRecipe)


def test_a_recipe_needs_both_halves() -> None:
    """A publish-only trainer has no default topology and a pool has to be sized
    for the model it serves, so neither half can be defaulted."""
    with pytest.raises(ValidationError, match="Field required"):
        StitchRecipe()  # pyright: ignore[reportCallIssue]


def test_derived_cross_half_settings_dont_mutate_the_caller_s_trainer() -> None:
    """The pool owns rollout parallelism and the trainer follows, but a caller may
    reuse the config object it passed in."""
    train = Qwen3_30B_A3B_Stitch_Train(rollout_num_gpus_per_engine=1)
    recipe = Qwen3_30B_A3B_Stitch_Recipe(train=train)
    assert recipe.train.rollout_num_gpus_per_engine == recipe.serve.gpus_per_replica
    assert recipe.train.rollout_num_gpus == 0
    assert recipe.train.colocate is False
    assert train.rollout_num_gpus_per_engine == 1


def test_the_pool_serves_the_trainers_export_baseline() -> None:
    """One baseline, derived at launch: a delta is defined against the bytes the
    trainer exports from, so neither half has a private copy to disagree with,
    and preset merging can't leave a stale copy behind."""
    recipe = Qwen3_30B_A3B_Stitch_Recipe()
    assert not hasattr(StitchServeConfig, "served_checkpoint_path")
    assert not hasattr(StitchServeConfig, "bulletin_root")
    assert not hasattr(Qwen3_30B_A3B_Stitch_Train, "served_checkpoint_format")
    assert not hasattr(Qwen3_30B_A3B_Stitch_Train, "bf16_checkpoint_path")
    assert not hasattr(Qwen3_30B_A3B_Stitch_Train, "update_weight_delta_encoding")
    assert recipe.served_checkpoint_path.startswith("/")
    assert recipe.served_baseline(Qwen3_30B()) == recipe.served_checkpoint_path


def test_shared_fields_are_added_to_the_trainer_payload() -> None:
    recipe = Qwen3_30B_A3B_Stitch_Recipe(
        served_checkpoint_path="/checkpoints/served",
        served_checkpoint_format="nvfp4",
        bf16_checkpoint_path="/checkpoints/bf16",
        update_weight_delta_encoding="sparse-xor",
        update_weight_delta_checksum="sha256",
    )

    fields = recipe.to_payload(model=Qwen3_30B()).fields
    assert fields["hf_checkpoint"] == "/checkpoints/served"
    assert fields["ref_load"] == "/checkpoints/bf16"
    assert fields["update_weight_delta_encoding"] == "sparse-xor"
    assert fields["update_weight_delta_checksum"] == "sha256"


def test_reporting_patches_target_the_stitch_miles_checkout() -> None:
    """The trainer image reuses miles' reporting patches, which rewrite the
    checkout at a hardcoded path — so stitch has to clone miles there."""
    for name in ("patch_rollout_status_reporting", "patch_advantage_distribution"):
        assert MILES_ROOT in (PATCHES_DIR / f"{name}.py").read_text()
