from __future__ import annotations

from typing import TYPE_CHECKING

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models.base import ModelConfig

if TYPE_CHECKING:
    from modal_training_gym.train_recipes.base import BaseTrainRecipe

_ALLOWED_MODALITIES = frozenset({"image", "audio"})


def requested_modalities(dataset: DatasetConfig) -> frozenset[str]:
    requested = frozenset(dataset.modalities)
    unknown = sorted(name for name in requested if name not in _ALLOWED_MODALITIES)
    if unknown:
        allowed = ", ".join(sorted(_ALLOWED_MODALITIES))
        bad = ", ".join(repr(name) for name in unknown)
        raise TrainingGymConfigError(f"unknown modality {bad}; allowed: {allowed}")
    return requested


def multimodal_key_map(dataset: DatasetConfig) -> dict[str, str] | None:
    requested = requested_modalities(dataset)
    if not requested:
        return None
    if dataset.media_column and len(requested) == 1:
        return {next(iter(requested)): dataset.media_column}
    return {name: f"{name}s" for name in requested}


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
