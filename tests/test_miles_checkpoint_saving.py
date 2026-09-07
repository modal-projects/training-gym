from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


def test_disabled_saving_omits_launcher_supplied_save_path():
    recipe = MilesRecipe(
        save_interval=None, save="/checkpoints/run", ref_load="/checkpoints/initial"
    )
    fields = recipe._fields()
    assert "save" not in fields
    assert fields.get("save_interval") is None
    assert "--save" not in recipe.cli_args()
    assert "--save-interval" not in recipe.cli_args()
    assert fields["ref_load"] == "/checkpoints/initial"
    assert recipe.save == "/checkpoints/run"


def test_enabled_saving_keeps_path_and_interval():
    fields = MilesRecipe(save_interval=10, save="/checkpoints/run")._fields()
    assert fields["save"] == "/checkpoints/run"
    assert fields["save_interval"] == 10
