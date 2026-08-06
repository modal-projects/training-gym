"""Validating a model by running base training on miles."""

from __future__ import annotations

from typing import ClassVar

from modal_training_gym.common.dataset import DatasetConfig, HuggingFaceDataset
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.models.validation import (
    ValidationFramework,
    ValidationTarget,
)
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe

from . import RecipeOverrides, ValidationBackend


class DapoMath17kDataset(HuggingFaceDataset):
    """DAPO-Math-17k prompts, as used by the Kimi multinode tutorials."""

    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = ""
    output_column = ""
    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True


class MilesValidationBackend(ValidationBackend):
    framework: ClassVar[ValidationFramework] = ValidationFramework.MILES

    supported_overrides: ClassVar[frozenset[str]] = frozenset(
        {"eval_interval", "save_interval", "docker_image"}
    )

    def _build_recipe(
        self,
        target: ValidationTarget,
        model_config: ModelConfig,
        step_count: int,
        overrides: RecipeOverrides,
    ) -> MilesRecipe:
        recipe = MilesRecipe.get_base_recipe(model_config)
        if recipe is None:
            raise TrainingGymConfigError(
                f"no base miles recipe for model {model_config.model_name!r}; "
                f"{target.name} is registered as a miles validation target"
            )
        if overrides.docker_image is not None:
            recipe.docker_image = overrides.docker_image
        return recipe

    def pick_dataset(
        self,
        target: ValidationTarget,
        model_config: ModelConfig,
        recipe: MilesRecipe,
        step_count: int,
    ) -> DatasetConfig:
        """Validation dataset for miles models.

        Every miles recipe today is a math-RL recipe scored by ``deepscaler``,
        so they all validate against DAPO-Math-17k. Every rollout step consumes
        ``rollout_batch_size`` prompts, so materialize enough for the whole run
        rather than assuming the loader wraps epochs.
        """
        return DapoMath17kDataset(n_rows=recipe.rollout_batch_size * step_count)

    def docker_image(self, recipe: MilesRecipe) -> str | None:
        return recipe.docker_image
