"""Validating a model by running disaggregated base training on stitch."""

from __future__ import annotations

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.models.qwen3_30b import Qwen3_30B
from modal_training_gym.train_recipes.stitch_recipe import (
    Qwen3_30B_A3B_Stitch_Recipe,
    StitchRecipe,
)

from .miles import DapoMath17kDataset

# Keyed by model, like the frameworks' ``get_base_recipe``: a stitch recipe is a
# trainer half plus a serving half, so there is no architecture-derived default.
_BASE_RECIPES: dict[str, type[StitchRecipe]] = {
    Qwen3_30B.model_name: Qwen3_30B_A3B_Stitch_Recipe,
}


def build_stitch_validation(
    model_config: ModelConfig, step_count: int
) -> tuple[StitchRecipe, DatasetConfig]:
    """The model's base stitch recipe and its validation dataset.

    The trainer is miles, so validation runs the same math-RL/deepscaler setup
    as the colocated miles backend — what stitch adds is the disaggregated
    rollout pool the recipe's serving half describes.
    """
    recipe_cls = _BASE_RECIPES.get(model_config.model_name)
    if recipe_cls is None:
        raise TrainingGymConfigError(
            f"no base stitch recipe for model {model_config.model_name!r}, "
            "which is registered as a stitch validation target"
        )
    recipe = recipe_cls()
    recipe.train.num_rollout = step_count
    return recipe, DapoMath17kDataset(
        n_rows=recipe.train.rollout_batch_size * step_count
    )
