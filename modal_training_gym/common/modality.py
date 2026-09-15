from __future__ import annotations

from typing import TYPE_CHECKING

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models.base import ModelConfig

if TYPE_CHECKING:
    from modal_training_gym.train_recipes.base import BaseTrainRecipe

_ALLOWED_MODALITIES = frozenset({"image", "audio"})


def requested_modalities(dataset: DatasetConfig) -> frozenset[str]:
    keys = dataset.multimodal_keys
    if not keys:
        return frozenset()
    unknown = sorted(key for key in keys if key not in _ALLOWED_MODALITIES)
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_MODALITIES))
        bad = ", ".join(repr(key) for key in unknown)
        raise TrainingGymConfigError(
            f"unknown multimodal_keys key {bad}; allowed: {allowed}"
        )
    return frozenset(keys)


def validate_modalities(
    recipe: BaseTrainRecipe, model: ModelConfig, dataset: DatasetConfig
) -> None:
    requested = requested_modalities(dataset)
    active = recipe.active_modalities()
    untrainable = sorted(active - recipe.trainable_modalities)
    if untrainable:
        allowed = ", ".join(sorted(recipe.trainable_modalities)) or "none"
        raise TrainingGymConfigError(
            f"{type(recipe).__name__} cannot train {', '.join(untrainable)} "
            f"(trains {allowed})."
        )
    if not requested:
        return

    unsupported_fw = sorted(requested - active)
    if unsupported_fw:
        allowed = ", ".join(sorted(active)) or "none"
        raise TrainingGymConfigError(
            f"{type(recipe).__name__} cannot serve {', '.join(unsupported_fw)} "
            f"(serves {allowed})."
        )

    unsupported_model = sorted(requested - model.supported_modalities)
    if unsupported_model:
        raise TrainingGymConfigError(
            f"{type(model).__name__} cannot serve {', '.join(unsupported_model)} "
            f"(supports {', '.join(sorted(model.supported_modalities))})."
        )
