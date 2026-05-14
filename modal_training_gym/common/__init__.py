"""Common cross-framework utilities and compatibility re-exports.

Exports shared constants + helpers used by every framework package. Each
framework's launcher merges ``COMMON_TRAINING_GYM_TAGS`` with its own
``_modal_framework`` tag and the user's ``config.app_tags`` when constructing
its ``modal.App``.
"""

from __future__ import annotations

import os

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


def hf_secrets() -> list:
    """Return a list of Modal Secrets providing ``HF_TOKEN`` to containers.

    Priority:
      1. Local ``HF_TOKEN`` env var → ``Secret.from_dict`` (also silences
         the local-side ``huggingface_hub`` "unauthenticated" warning).
      2. Named Modal secret ``huggingface-secret`` → used if it exists in
         the workspace.
      3. Empty list → container runs without an HF token.
    """
    from modal import Secret

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        return [Secret.from_dict({"HF_TOKEN": hf_token})]

    try:
        secret = Secret.from_name("huggingface-secret")
        secret.hydrate()
        return [secret]
    except Exception:
        return []


__all__ = [
    "COMMON_TRAINING_GYM_TAGS",
    "GPUType",
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "hf_secrets",
    "vol_get",
    "vol_list",
    "vol_put",
]
