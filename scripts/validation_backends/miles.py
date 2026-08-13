"""Validating a model by running base training on miles."""

from __future__ import annotations

from modal_training_gym.common.dataset import DatasetConfig, HuggingFaceDataset
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe


class DapoMath17kDataset(HuggingFaceDataset):
    """DAPO-Math-17k prompts for Miles math-RL validation."""

    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = ""
    output_column = ""
    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    needs_chat_template = True
    requires_refresh_before_training = True


def build_miles_validation(
    model_config: ModelConfig, step_count: int
) -> tuple[MilesRecipe, DatasetConfig]:
    """The model's base miles recipe and its validation dataset.

    Every miles recipe today is a math-RL recipe scored by ``deepscaler``, so
    they all validate against DAPO-Math-17k. Every rollout step consumes
    ``rollout_batch_size`` prompts, so materialize enough for the whole run
    rather than assuming the loader wraps epochs.

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
    prompts_per_step = max(
        recipe.rollout_batch_size, recipe.over_sampling_batch_size or 0
    )
    return recipe, DapoMath17kDataset(n_rows=prompts_per_step * step_count)
