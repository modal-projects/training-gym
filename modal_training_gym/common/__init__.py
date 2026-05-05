"""Common cross-framework utilities.

Exports shared constants + classes used by every framework package. Each
framework's launcher merges `COMMON_TRAINING_GYM_TAGS` with its own
`_modal_framework` tag and the user's `config.app_tags` when constructing its
`modal.App`.
"""

from __future__ import annotations

import io
import json
from typing import Any, Literal

COMMON_TRAINING_GYM_TAGS: dict[str, str] = {
    "training": "True",
    "source": "training-gym",
    "_modal_job_type": "training",
}

# Supported Modal GPU families across all frameworks. Framework configs
# that accept a GPU name should annotate it with this type.
GPUType = Literal["H100", "H200", "B200", "B300"]

METADATA_VOLUME_NAME = "training-gym-metadata"


def _metadata_volume():
    import modal

    return modal.Volume.from_name(METADATA_VOLUME_NAME, create_if_missing=True)


def vol_put(store_name: str, key: str, value: dict[str, Any]) -> None:
    vol = _metadata_volume()
    data = json.dumps(value).encode()
    with vol.batch_upload() as batch:
        batch.put_file(io.BytesIO(data), f"{store_name}/{key}.json")


def vol_get(store_name: str, key: str) -> dict[str, Any]:
    vol = _metadata_volume()
    try:
        data = b"".join(vol.read_file(f"{store_name}/{key}.json"))
        return json.loads(data)
    except FileNotFoundError:
        raise KeyError(key)


def vol_list(store_name: str) -> list[dict[str, Any]]:
    vol = _metadata_volume()
    results = []
    try:
        for entry in vol.iterdir(store_name):
            if entry.path.endswith(".json"):
                data = b"".join(vol.read_file(entry.path))
                results.append(json.loads(data))
    except FileNotFoundError:
        pass
    return results
