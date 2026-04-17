"""Common cross-framework utilities.

Exports shared constants + classes used by every framework package. Each
framework's launcher merges `COMMON_TRAINING_GYM_TAGS` with its own
`framework` tag and the user's `config.app_tags` when constructing its
`modal.App`.
"""

COMMON_TRAINING_GYM_TAGS: dict[str, str] = {
    "training": "True",
    "source": "training-gym",
}
