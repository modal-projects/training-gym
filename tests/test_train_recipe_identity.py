import pytest

from modal_training_gym.common.dataset import HuggingFaceDataset
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.models import ModelConfig, Qwen3_4B, Qwen3_5_4B
from modal_training_gym.common.train import TrainConfig
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.miles_recipe.qwen3_5_4b import (
    Qwen3_5_4b_Miles_Recipe,
)
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe
from modal_training_gym.train_recipes.slime_recipe.qwen3_4b import Qwen3_4b_Recipe


def _config(recipe: SlimeRecipe | MilesRecipe, model: ModelConfig) -> TrainConfig:
    return TrainConfig(
        dataset=HuggingFaceDataset(
            hf_repo="some/dataset",
            input_column="prompt",
            output_column="answer",
        ),
        model=model,
        recipe=recipe,
    )


@pytest.mark.parametrize(
    ("recipe", "model", "framework"),
    [
        (
            Qwen3_4b_Recipe(),
            Qwen3_4B(),
            Framework.SLIME,
        ),
        (
            Qwen3_5_4b_Miles_Recipe(),
            Qwen3_5_4B(),
            Framework.MILES,
        ),
    ],
)
def test_recipe_subclass_selects_framework_and_builder(
    recipe, model, framework, monkeypatch
) -> None:
    calls: list[Framework] = []
    monkeypatch.setattr(
        "modal_training_gym.common.train._warn_if_external_build_app", lambda: None
    )
    monkeypatch.setattr(
        "modal_training_gym.common.train.build_slime_app",
        lambda **kwargs: calls.append(Framework.SLIME),
    )
    monkeypatch.setattr(
        "modal_training_gym.common.train.build_miles_app",
        lambda **kwargs: calls.append(Framework.MILES),
    )

    config = _config(recipe, model)

    assert config.framework is framework
    config._build_app("run-id")
    assert calls == [framework]
