"""Validating a model by running base training on slime."""

from __future__ import annotations

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.slime_recipe import SlimeRecipe

from .datasets import Gsm8kDataset, media_dataset

VALIDATION_EPHEMERAL_DISK_MIB = 2_097_152


def build_slime_validation(
    model_config: ModelConfig, modality: str, step_count: int
) -> tuple[SlimeRecipe, DatasetConfig]:
    """The model's base slime recipe and the dataset for one modality."""
    recipe = SlimeRecipe.get_base_recipe(model_config)
    recipe.rm_type = "deepscaler"
    recipe.train_function_kwargs = {
        **dict(recipe.train_function_kwargs or {}),
        "ephemeral_disk": VALIDATION_EPHEMERAL_DISK_MIB,
    }
    if modality == "text":
        return recipe, Gsm8kDataset(n_rows=10)
    return recipe, media_dataset(modality)
