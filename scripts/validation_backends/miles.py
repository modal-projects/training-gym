"""Validating a model by running base training on miles."""

from __future__ import annotations

from modal_training_gym.common.dataset import (
    DatasetConfig,
    EmbeddingProjectorDataset,
    HuggingFaceDataset,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.train_recipes.miles_recipe import MilesRecipe
from modal_training_gym.train_recipes.miles_recipe.glm_5_2 import (
    GLM_5_2_Projector_Recipe,
)


class DapoMath17kDataset(HuggingFaceDataset):
    """DAPO-Math-17k prompts for Miles math-RL validation."""

    hf_repo = "zhuzilin/dapo-math-17k"
    input_column = ""
    output_column = ""
    input_key = "prompt"
    label_key = "label"
    output_format = "jsonl"
    apply_chat_template = True
    always_prepare = True


def build_miles_validation(
    model_config: ModelConfig, step_count: int
) -> tuple[MilesRecipe, DatasetConfig]:
    """The model's base miles recipe and its validation dataset.

    Miles recipes are math-RL recipes scored by ``deepscaler``, so they validate
    against DAPO-Math-17k. Every rollout step consumes ``rollout_batch_size``
    prompts, so materialize enough for the whole run rather than assuming the
    loader wraps epochs.

    Projector-only recipes are the exception: they train supervised on external
    embeddings, which no prompt dataset carries, so they validate on synthetic
    embedding rows sized to the projector's input dimension.

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
    if isinstance(recipe, GLM_5_2_Projector_Recipe):
        return recipe, EmbeddingProjectorDataset.synthetic(
            n_rows=recipe.rollout_batch_size * step_count,
            input_dim=recipe.projector.input_dim,
        )
    prompts_per_step = max(
        recipe.rollout_batch_size, recipe.over_sampling_batch_size or 0
    )
    return recipe, DapoMath17kDataset(n_rows=prompts_per_step * step_count)
