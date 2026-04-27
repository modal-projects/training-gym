"""Common cross-framework utilities.

Exports shared constants + classes used by every framework package. Each
framework's launcher merges `COMMON_TRAINING_GYM_TAGS` with its own
`_modal_framework` tag and the user's `config.app_tags` when constructing its
`modal.App`.
"""

from typing import Literal

COMMON_TRAINING_GYM_TAGS: dict[str, str] = {
    "training": "True",
    "source": "training-gym",
    "_modal_job_type": "training",
}

# Supported Modal GPU families across all frameworks. Framework configs
# that accept a GPU name should annotate it with this type.
GPUType = Literal[
    "A100", "A100-40GB", "A100-80GB",
    "H100", "H200", "B200",
]
