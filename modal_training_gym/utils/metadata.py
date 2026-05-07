"""Enum-backed helpers for metadata stored in the shared Modal Volume."""

from __future__ import annotations

import io
import json
from enum import Enum
from typing import Any

METADATA_VOLUME_NAME = "training-gym-metadata"


class MetadataStore(Enum):
    TRAINING_RUNS = "training-runs"
    TRAIN_RESULTS = "train-results"
    EVAL_RESULTS = "eval-results"
    EVAL_CONFIGS = "eval-configs"
    DEPLOYMENTS = "deployments"


def _metadata_volume():
    import modal

    return modal.Volume.from_name(METADATA_VOLUME_NAME, create_if_missing=True)


def _store_path(store: MetadataStore | str) -> str:
    if isinstance(store, MetadataStore):
        return store.value
    return store


def vol_put(store: MetadataStore | str, key: str, value: dict[str, Any]) -> None:
    from modal.exception import InvalidError, NotFoundError

    vol = _metadata_volume()
    data = json.dumps(value).encode()
    path = f"{_store_path(store)}/{key}.json"
    try:
        vol.remove_file(path)
    except (FileNotFoundError, NotFoundError):
        pass
    except InvalidError as exc:
        if "No such file or directory" not in str(exc):
            raise
    with vol.batch_upload() as batch:
        batch.put_file(io.BytesIO(data), path)


def vol_get(store: MetadataStore | str, key: str) -> dict[str, Any]:
    vol = _metadata_volume()
    try:
        data = b"".join(vol.read_file(f"{_store_path(store)}/{key}.json"))
        return json.loads(data)
    except FileNotFoundError:
        raise KeyError(key) from None


def vol_list(store: MetadataStore | str) -> list[dict[str, Any]]:
    vol = _metadata_volume()
    results = []
    try:
        for entry in vol.iterdir(_store_path(store)):
            if entry.path.endswith(".json"):
                data = b"".join(vol.read_file(entry.path))
                results.append(json.loads(data))
    except FileNotFoundError:
        pass
    return results


__all__ = [
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "vol_get",
    "vol_list",
    "vol_put",
]
