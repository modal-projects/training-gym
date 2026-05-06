"""Common cross-framework utilities and compatibility re-exports.

Exports shared constants + helpers used by every framework package. Each
framework's launcher merges ``COMMON_TRAINING_GYM_TAGS`` with its own
``_modal_framework`` tag and the user's ``config.app_tags`` when constructing
its ``modal.App``.
"""

from modal_training_gym.utils.gpu import GPUType
from modal_training_gym.utils.metadata import (
    METADATA_VOLUME_NAME,
    MetadataStore,
    vol_get,
    vol_list,
    vol_put,
)

COMMON_TRAINING_GYM_TAGS: dict[str, str] = {
    "training": "True",
    "source": "training-gym",
    "_modal_job_type": "training",
}

__all__ = [
    "COMMON_TRAINING_GYM_TAGS",
    "GPUType",
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "vol_get",
    "vol_list",
    "vol_put",
]
