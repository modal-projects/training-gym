"""Validating a model by running base training on miles."""

from __future__ import annotations

from dataclasses import replace

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe

from .datasets import dapo_math_dataset, media_dataset


def build_miles_validation(
    model_config: ModelConfig, modality: str, step_count: int
) -> tuple[MilesRecipe, DatasetConfig]:
    """The model's base miles recipe and the dataset for one modality.

    Every rollout step consumes ``rollout_batch_size`` prompts, so materialize
    enough for the whole run rather than assuming the loader wraps epochs.

    The recipe is used as it comes out of ``get_base_recipe``, image included:
    the image a miles model trains on belongs in ``MilesRecipe``, not in the
    validation harness.
    """
    recipe = MilesRecipe.get_base_recipe(model_config)
    if recipe is None:
        raise TrainingGymConfigError(
            f"no base miles recipe for model {model_config.model_name!r}, "
            "which is registered as a miles validation target"
        )
    recipe.skip_eval_before_train = True
    recipe.rm_type = "deepscaler"
    prompts_per_step = max(
        recipe.rollout_batch_size, recipe.over_sampling_batch_size or 0
    )
    n_rows = prompts_per_step * step_count
    if modality != "text":
        recipe = replace(recipe, modality="vision" if modality == "image" else modality)
        dataset = media_dataset(modality, n_rows=n_rows)
    else:
        dataset = dapo_math_dataset(n_rows=n_rows)
    return recipe, dataset
