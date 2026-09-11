import yaml
import pytest

from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.run import TrainingRun
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


def test_run_without_checkpoint_keeps_original_model_source(fake_volume):
    model = ModelConfig(model_name="org/model", model_path="/hf/model")
    run = TrainingRun(
        training_run_id="run",
        framework=Framework.MILES,
        config={},
        app_name="test",
        source_model=model,
    )
    assert run.checkpoints() == []
    assert run.model.model_path == "/hf/model"
