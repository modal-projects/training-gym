"""Enum-backed helpers for metadata stored in the shared Modal Volume."""

from __future__ import annotations

import io
import json
from enum import Enum
from typing import Any

METADATA_VOLUME_NAME = "training-gym-metadata"


class MetadataStore(Enum):
    TRAINING_RUNS = "training-runs"
    TRAINING_RUNS_SUMMARY = "training-runs-summary"
    TRAIN_RESULTS = "train-results"
    TRAIN_RESULTS_SUMMARY = "train-results-summary"
    EVAL_RESULTS = "eval-results"
    EVALS = "evals"
    EVAL_SUMMARIES = "eval-summaries"
    EVAL_CONFIGS = "eval-configs"
    DEPLOYMENTS = "deployments"
    DEPLOYMENTS_SUMMARY = "deployments-summary"


SUMMARY_KEY = "summary"
SUMMARY_ITEMS_KEY = "items"


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


def summary_items_from_payload(
    payload: Any,
    payload_key: str = SUMMARY_ITEMS_KEY,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get(payload_key, [])
    else:
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def vol_get_summary_items(
    store: MetadataStore | str,
    *,
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
) -> list[dict[str, Any]] | None:
    try:
        payload = vol_get(store, key)
    except KeyError:
        return None
    return summary_items_from_payload(payload, payload_key=payload_key)


def vol_put_summary_items(
    store: MetadataStore | str,
    items: list[dict[str, Any]],
    *,
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
) -> None:
    vol_put(store, key, {payload_key: items})


def vol_upsert_summary_item(
    store: MetadataStore | str,
    item: dict[str, Any],
    *,
    item_id_key: str,
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
    sort_key: Any = None,
    reverse: bool = False,
) -> None:
    item_id = item.get(item_id_key)
    if item_id is None:
        raise KeyError(f"Missing summary item id key {item_id_key!r}")

    items = vol_get_summary_items(store, key=key, payload_key=payload_key) or []
    items = [existing for existing in items if existing.get(item_id_key) != item_id]
    items.append(item)
    if sort_key is not None:
        items.sort(key=sort_key, reverse=reverse)
    vol_put_summary_items(store, items, key=key, payload_key=payload_key)


__all__ = [
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "SUMMARY_ITEMS_KEY",
    "SUMMARY_KEY",
    "summary_items_from_payload",
    "vol_get",
    "vol_list",
    "vol_put",
    "vol_get_summary_items",
    "vol_put_summary_items",
    "vol_upsert_summary_item",
]
