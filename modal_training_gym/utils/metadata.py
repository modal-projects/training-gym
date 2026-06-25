"""Enum-backed helpers for metadata stored in the shared Modal Volume."""

from __future__ import annotations

import io
import json
from enum import Enum
from typing import Any, Callable

METADATA_VOLUME_NAME = "training-gym-metadata"


class MetadataStore(Enum):
    TRAINING_RUNS = "training-runs"
    TRAINING_RUNS_SUMMARY = "training-runs-summary"
    FRAMEWORK_STATUS_TOKENS = "framework-status-tokens"
    TRAIN_RESULTS = "train-results"
    TRAIN_RESULTS_SUMMARY = "train-results-summary"
    TRAINING_ROLLOUTS = "training-rollouts"
    TRAINING_ROLLOUTS_SUMMARY = "training-rollouts-summary"
    EVAL_RESULTS = "eval-results"
    EVALS = "evals"
    EVAL_SUMMARIES = "eval-summaries"
    EVAL_CONFIGS = "eval-configs"
    DEPLOYMENTS = "deployments"
    DEPLOYMENTS_SUMMARY = "deployments-summary"


SUMMARY_KEY = "summary"
SUMMARY_ITEMS_KEY = "items"


# Summary stores whose canonical per-item files share the summary's shape, so a
# collapsed/stale summary can be rebuilt from the canonical files rather than
# trusted blindly. Rollouts are intentionally excluded: their canonical files
# hold full sample payloads, not the reduced summary shape.
class _SummaryCompaction:
    __slots__ = ("item_store", "item_id_key", "sort_key", "reverse")

    def __init__(self, item_store, item_id_key, sort_key, reverse):
        self.item_store = item_store
        self.item_id_key = item_id_key
        self.sort_key = sort_key
        self.reverse = reverse


_SUMMARY_COMPACTION: dict[MetadataStore, _SummaryCompaction] = {
    MetadataStore.TRAINING_RUNS_SUMMARY: _SummaryCompaction(
        item_store=MetadataStore.TRAINING_RUNS,
        item_id_key="training_run_id",
        sort_key=lambda item: (
            int(item.get("created_at", 0) or 0),
            str(item.get("training_run_id", "")),
        ),
        reverse=True,
    ),
    MetadataStore.TRAIN_RESULTS_SUMMARY: _SummaryCompaction(
        item_store=MetadataStore.TRAIN_RESULTS,
        item_id_key="training_run_id",
        sort_key=lambda item: str(item.get("training_run_id", "")),
        reverse=True,
    ),
    MetadataStore.DEPLOYMENTS_SUMMARY: _SummaryCompaction(
        item_store=MetadataStore.DEPLOYMENTS,
        item_id_key="deployment_id",
        sort_key=lambda item: (
            str(item.get("deployment_config", {}).get("app_name", "")),
            str(item.get("deployment_id", "")),
        ),
        reverse=True,
    ),
}

_SUMMARY_CANONICAL_STORES: dict[MetadataStore, MetadataStore] = {
    summary: cfg.item_store for summary, cfg in _SUMMARY_COMPACTION.items()
}


def _canonical_items_for(
    store: MetadataStore | str, item_id_key: str, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    item_store = (
        _SUMMARY_CANONICAL_STORES.get(store)
        if isinstance(store, MetadataStore)
        else None
    )
    if item_store is None:
        return [
            item for item in items if isinstance(item, dict) and item.get(item_id_key)
        ]
    return [
        item
        for item in vol_list(item_store)
        if isinstance(item, dict) and item.get(item_id_key) is not None
    ]


async def _canonical_items_for_async(
    store: MetadataStore | str, item_id_key: str, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    item_store = (
        _SUMMARY_CANONICAL_STORES.get(store)
        if isinstance(store, MetadataStore)
        else None
    )
    if item_store is None:
        return [
            item for item in items if isinstance(item, dict) and item.get(item_id_key)
        ]
    return [
        item
        for item in await vol_list_async(item_store)
        if isinstance(item, dict) and item.get(item_id_key) is not None
    ]


def _metadata_volume():
    import modal

    return modal.Volume.from_name(METADATA_VOLUME_NAME, create_if_missing=True)


def _safe_reload(vol) -> None:
    try:
        vol.reload()
    except RuntimeError:
        pass


async def _safe_reload_async(vol) -> None:
    try:
        await vol.reload.aio()
    except RuntimeError:
        pass


def _store_path(store: MetadataStore | str) -> str:
    if isinstance(store, MetadataStore):
        return store.value
    return store


def vol_remove(store: MetadataStore | str, key: str) -> bool:
    """Delete a single item from a store. Returns True if removed."""
    from modal.exception import InvalidError, NotFoundError

    vol = _metadata_volume()
    path = f"{_store_path(store)}/{key}.json"
    try:
        vol.remove_file(path)
        return True
    except (FileNotFoundError, NotFoundError):
        return False
    except InvalidError as exc:
        if "No such file or directory" in str(exc):
            return False
        raise


def vol_put(store: MetadataStore | str, key: str, value: dict[str, Any]) -> None:
    # force=True overwrites in place at commit, so the file is never absent
    # mid-write. A remove-then-upload sequence leaves a window where readers
    # see no file and treat the store as empty — which silently collapses
    # read-modify-write summaries down to a single item.
    vol = _metadata_volume()
    data = json.dumps(value).encode()
    path = f"{_store_path(store)}/{key}.json"
    with vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(data), path)


async def vol_put_async(
    store: MetadataStore | str, key: str, value: dict[str, Any]
) -> None:
    vol = _metadata_volume()
    data = json.dumps(value).encode()
    path = f"{_store_path(store)}/{key}.json"
    async with vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(data), path)


def vol_get(store: MetadataStore | str, key: str) -> dict[str, Any]:
    vol = _metadata_volume()
    path = f"{_store_path(store)}/{key}.json"
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except FileNotFoundError:
        _safe_reload(vol)
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except FileNotFoundError:
        raise KeyError(key) from None


async def vol_get_async(store: MetadataStore | str, key: str) -> dict[str, Any]:
    vol = _metadata_volume()
    path = f"{_store_path(store)}/{key}.json"
    try:
        chunks = [chunk async for chunk in vol.read_file.aio(path)]
        return json.loads(b"".join(chunks))
    except FileNotFoundError:
        await _safe_reload_async(vol)
    try:
        chunks = [chunk async for chunk in vol.read_file.aio(path)]
        return json.loads(b"".join(chunks))
    except FileNotFoundError:
        raise KeyError(key) from None


def vol_list(store: MetadataStore | str) -> list[dict[str, Any]]:
    import time as _time

    vol = _metadata_volume()
    _safe_reload(vol)
    results = []
    for attempt in range(3):
        try:
            for entry in vol.iterdir(_store_path(store)):
                if entry.path.endswith(".json"):
                    data = b"".join(vol.read_file(entry.path))
                    results.append(json.loads(data))
            return results
        except FileNotFoundError:
            return results
        except Exception as exc:
            if "rate limit" in str(exc).lower() and attempt < 2:
                _time.sleep(2**attempt)
                results = []
                continue
            raise
    return results


async def vol_list_async(store: MetadataStore | str) -> list[dict[str, Any]]:
    vol = _metadata_volume()
    await _safe_reload_async(vol)
    results = []
    try:
        async for entry in vol.iterdir.aio(_store_path(store)):
            if entry.path.endswith(".json"):
                chunks = [chunk async for chunk in vol.read_file.aio(entry.path)]
                results.append(json.loads(b"".join(chunks)))
    except FileNotFoundError:
        pass
    return results


def vol_count_items(store: MetadataStore | str) -> int:
    """Count canonical ``.json`` files in a store without reading them.

    A single directory listing, used to cheaply detect a collapsed summary
    (summary item count < canonical file count) before paying for a full
    rebuild via ``vol_list``.
    """
    vol = _metadata_volume()
    _safe_reload(vol)
    try:
        return sum(
            1 for e in vol.iterdir(_store_path(store)) if e.path.endswith(".json")
        )
    except FileNotFoundError:
        return 0


def compact_summary_store(summary_store: MetadataStore) -> list[dict[str, Any]]:
    """Rebuild a registered summary from its canonical per-item files."""
    cfg = _SUMMARY_COMPACTION[summary_store]
    return vol_compact_summary_items(
        summary_store,
        cfg.item_store,
        item_id_key=cfg.item_id_key,
        sort_key=cfg.sort_key,
        reverse=cfg.reverse,
    )


def vol_get_summary_items_healed(summary_store: MetadataStore) -> list[dict[str, Any]]:
    """Read a summary, rebuilding from canonical files if it looks collapsed.

    Self-heals the read path: if a racing writer clobbered the summary down to
    fewer items than there are canonical files, rebuild from canonical instead
    of surfacing the truncated list. The canonical count is a cheap one-op
    directory listing, so the expensive rebuild only runs when actually needed.
    """
    items = vol_get_summary_items(summary_store) or []
    cfg = _SUMMARY_COMPACTION.get(summary_store)
    if cfg is None:
        return items
    if vol_count_items(cfg.item_store) > len(items):
        return compact_summary_store(summary_store)
    return items


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
    vol = _metadata_volume()
    _safe_reload(vol)
    try:
        payload = vol_get(store, key)
    except KeyError:
        return None
    return summary_items_from_payload(payload, payload_key=payload_key)


async def vol_get_summary_items_async(
    store: MetadataStore | str,
    *,
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
) -> list[dict[str, Any]] | None:
    vol = _metadata_volume()
    await _safe_reload_async(vol)
    try:
        payload = await vol_get_async(store, key)
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


async def vol_put_summary_items_async(
    store: MetadataStore | str,
    items: list[dict[str, Any]],
    *,
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
) -> None:
    await vol_put_async(store, key, {payload_key: items})


def vol_compact_summary_items(
    summary_store: MetadataStore | str,
    item_store: MetadataStore | str,
    *,
    item_id_key: str,
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
    sort_key: Callable[[dict[str, Any]], Any] | None = None,
    reverse: bool = False,
) -> list[dict[str, Any]]:
    """Rebuild a denormalized summary from canonical per-item metadata files.

    Summary files are a list cache. Writers persist the canonical item file first,
    then best-effort update the summary. If parallel read-modify-write summary
    upserts clobber each other, compaction merges the canonical files back into
    the summary so list readers become self-healing.
    """
    summary_items = (
        vol_get_summary_items(summary_store, key=key, payload_key=payload_key) or []
    )
    canonical_items = vol_list(item_store)

    items_by_id = {
        item[item_id_key]: item
        for item in summary_items
        if item.get(item_id_key) is not None
    }
    for item in canonical_items:
        item_id = item.get(item_id_key)
        if item_id is None:
            continue
        items_by_id[item_id] = {**items_by_id.get(item_id, {}), **item}

    items = list(items_by_id.values())
    if sort_key is not None:
        items.sort(key=sort_key, reverse=reverse)
    vol_put_summary_items(summary_store, items, key=key, payload_key=payload_key)
    return items


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
    if not items:
        items = _canonical_items_for(store, item_id_key, items)
    items = [existing for existing in items if existing.get(item_id_key) != item_id]
    items.append(item)
    if sort_key is not None:
        items.sort(key=sort_key, reverse=reverse)
    vol_put_summary_items(store, items, key=key, payload_key=payload_key)


async def vol_upsert_summary_item_async(
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

    items = (
        await vol_get_summary_items_async(store, key=key, payload_key=payload_key) or []
    )
    if not items:
        items = await _canonical_items_for_async(store, item_id_key, items)
    items = [existing for existing in items if existing.get(item_id_key) != item_id]
    items.append(item)
    if sort_key is not None:
        items.sort(key=sort_key, reverse=reverse)
    await vol_put_summary_items_async(store, items, key=key, payload_key=payload_key)


__all__ = [
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "SUMMARY_ITEMS_KEY",
    "SUMMARY_KEY",
    "summary_items_from_payload",
    "vol_get",
    "vol_get_async",
    "vol_list",
    "vol_list_async",
    "vol_count_items",
    "compact_summary_store",
    "vol_get_summary_items_healed",
    "vol_put",
    "vol_put_async",
    "vol_remove",
    "vol_get_summary_items",
    "vol_get_summary_items_async",
    "vol_put_summary_items",
    "vol_put_summary_items_async",
    "vol_compact_summary_items",
    "vol_upsert_summary_item",
    "vol_upsert_summary_item_async",
]
