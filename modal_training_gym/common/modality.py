from __future__ import annotations

from typing import Literal, Protocol

from modal_training_gym.common.dataset import DatasetConfig
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.models.base import ModelConfig

MediaModality = Literal["image", "audio", "video"]


class _ServedMedia(Protocol):
    def served_media(self) -> frozenset[str]: ...


_MEDIA: tuple[MediaModality, ...] = ("image", "audio", "video")


def requested_media_modalities(dataset: DatasetConfig) -> frozenset[MediaModality]:
    keys = dataset.multimodal_keys
    if not keys:
        return frozenset()
    return frozenset(key for key in _MEDIA if key in keys)


def validate_served_modalities(
    recipe: _ServedMedia, model: ModelConfig, dataset: DatasetConfig
) -> None:
    requested = requested_media_modalities(dataset)
    if not requested:
        return

    served = recipe.served_media()
    unsupported_fw = sorted(requested - served)
    if unsupported_fw:
        allowed = ", ".join(sorted(served)) or "none"
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
