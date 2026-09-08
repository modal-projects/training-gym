import os

import yaml
import pytest

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.train_result import TrainResult
from modal_training_gym.frameworks.miles.modal_helpers.utils import prepare_miles_config
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


def test_explicitly_disabled_saving_omits_both_flags():
    recipe = MilesRecipe(save_interval=None, save=None, ref_load="/checkpoints/initial")
    fields = recipe._fields()
    assert fields["save"] is None
    assert fields.get("save_interval") is None
    assert "--save" not in recipe.cli_args()
    assert "--save-interval" not in recipe.cli_args()
    assert fields["ref_load"] == "/checkpoints/initial"
    assert recipe.save is None


def test_enabled_saving_keeps_path_and_interval():
    fields = MilesRecipe(save_interval=10, save="/checkpoints/run")._fields()
    assert fields["save"] == "/checkpoints/run"
    assert fields["save_interval"] == 10


@pytest.mark.parametrize("enabled", [False, True])
def test_save_overrides_survive_yaml_materialization(tmp_path, enabled):
    override = {
        "save": "/checkpoints/override" if enabled else None,
        "save_interval": 10 if enabled else None,
    }
    recipe = MilesRecipe(
        save=None if enabled else "/checkpoints/original",
        save_interval=None if enabled else 10,
        extra_config=override,
    )
    for materialized in (False, True):
        if materialized:
            prepare_miles_config(recipe, None, str(tmp_path))
            with open(recipe.extra_config) as f:
                assert yaml.safe_load(f) == override
        assert "--save" not in recipe.cli_args()
        assert "--save-interval" not in recipe.cli_args()


def test_result_without_checkpoint_keeps_original_model_source():
    model = ModelConfig(model_name="org/model", model_path="/hf/model")
    result = TrainResult(
        app_name="test",
        framework=Framework.MILES,
        training_run_id="run",
        checkpoint_dir="",
        model_config=model,
    )
    assert result.checkpoints() == []
    assert result.model.model_path == "/hf/model"


def _resolve(save, extra_config, tmp_path):
    from modal_training_gym.frameworks.miles.launcher import resolve_effective_save

    return resolve_effective_save(
        save,
        extra_config,
        recipe_default_save_root="/checkpoints",
        mounted_save_root=str(tmp_path / "mounted"),
        training_run_id="run-1",
    )


def test_effective_save_uses_yaml_override_verbatim(tmp_path):
    override = str(tmp_path / "override")
    effective, save_root = _resolve(
        None, {"save": override + "/", "save_interval": 10}, tmp_path
    )
    assert effective == override + "/"
    assert save_root == override
    assert os.path.isdir(save_root)


def test_effective_save_yaml_wins_over_top_level(tmp_path):
    override = str(tmp_path / "override")
    _, save_root = _resolve(str(tmp_path / "top"), {"save": override}, tmp_path)
    assert save_root == override


def test_effective_save_top_level_is_run_scoped(tmp_path):
    effective, save_root = _resolve(
        str(tmp_path / "top"), {"save_interval": 5}, tmp_path
    )
    assert effective == str(tmp_path / "top")
    assert save_root == str(tmp_path / "top" / "run-1")


def test_effective_save_default_redirects_to_mount(tmp_path):
    _, save_root = _resolve("/checkpoints", None, tmp_path)
    assert save_root == str(tmp_path / "mounted" / "run-1")


@pytest.mark.parametrize("extra_config", [None, {"save": None}])
def test_effective_save_disabled_reports_none(tmp_path, extra_config):
    effective, save_root = _resolve(None, extra_config, tmp_path)
    assert effective is None
    assert save_root == str(tmp_path / "mounted" / "run-1")


def test_yaml_save_none_disables_top_level_save(tmp_path):
    effective, _ = _resolve(str(tmp_path / "top"), {"save": None}, tmp_path)
    assert effective is None
