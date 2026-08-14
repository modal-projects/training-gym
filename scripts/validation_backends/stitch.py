"""Validating a model by running disaggregated base training on stitch."""

from __future__ import annotations

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.stitch_recipe import StitchRecipe

from .miles import DapoMath17kDataset


def build_stitch_validation(
    model_config: ModelConfig, step_count: int
) -> tuple[StitchRecipe, DatasetConfig]:
    """The model's base stitch recipe and its validation dataset.

    The trainer is miles, so validation runs the same math-RL/deepscaler setup
    as the colocated miles backend — what stitch adds is the disaggregated
    rollout pool the recipe's serving half describes.
    """
    recipe = StitchRecipe.get_base_recipe(model_config)
    if recipe is None:
        raise TrainingGymConfigError(
            f"no base stitch recipe for model {model_config.model_name!r}, "
            "which is registered as a stitch validation target"
        )
    recipe.train.num_rollout = step_count
    # Spend the budget on the requested steps, like the miles backend.
    recipe.train.skip_eval_before_train = True
    return recipe, DapoMath17kDataset(
        n_rows=recipe.train.rollout_batch_size * step_count
    )
