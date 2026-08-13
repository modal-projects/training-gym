"""Trainer-side stitch hooks: publish a version, claim the pool, gate requests.

Vendored from the stitch cookbook (``cookbook/common/hooks.py``). miles resolves
these by dotted path off the recipe:

- :func:`commit_and_wake` — ``custom_update_weight_post_write_path``: make the
  written version durable, advance ``latest``, wake the Flash pool.
- :func:`gated_rollout_request_hook` — ``StitchRecipe.custom_rollout_request_hook_path``:
  pin each rollout request to a bounded-staleness version.
- :func:`claim_pool` — called by the launcher before training publishes, to reset
  every replica to base for this run.

Each hook reads the run's coordinates off the trainer's ``args`` namespace (miles
``setattr``\\ s every ``--custom-config-path`` key onto it) and drives the stitch
core against a ``ModalVolumeStore`` + ``ModalFlashPool``. The store is scoped to
the run: ``run_id`` is the run's fence token, so a stale trainer can't advance a
pointer another run owns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from stitch.pools.modal_flash import ModalFlashPool
from stitch.publish import claim_run, constrain_request, publish_version
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.types import PointerRewind

from modal_training_gym.frameworks.stitch import sidecar_process

logger = logging.getLogger(__name__)


def sample_affinity_key(sample: Any) -> str | None:
    """One stable routing key for a rollout trajectory or GRPO group, so siblings
    land on the same replica."""
    group_index = getattr(sample, "group_index", None)
    if group_index is not None:
        return f"group-{group_index}"
    for name in ("routing_key", "session_id"):
        value = getattr(sample, name, None)
        if value is not None:
            return str(value)
    return None


# ── Publish ───────────────────────────────────────────────────────────────────


def commit_and_wake(args: Any, published_dir: str, rollout_engines: Any = None) -> None:
    """Bridge miles' disk-delta publish to the stitch store.

    miles fires this at each durability boundary: a version dir (``weight_vNNNNNN``,
    holding the HF index) and — at baseline/pointer commit — the run dir. Every rank
    flushes its writes, then version-dir calls rendezvous before rank 0 advances the
    pointer, so no index naming another rank's shard is ever exposed early. Keying on
    the dir name (rather than reading an index) keeps run-dir calls a clean no-op.
    """
    del rollout_engines
    store = _store(args)
    store.commit()
    if not Path(published_dir).name.startswith("weight_v"):
        return
    sidecar_process.dist_barrier()
    if sidecar_process.dist_rank() not in (None, 0):
        return
    try:
        publish_version(store, _pool(args), published_dir, run_id=_run_id(args))
    except PointerRewind:
        # A same-run republish (e.g. a retried step) — drop it rather than serve stale.
        logger.warning(
            "publish of %s would rewind latest; dropping", published_dir, exc_info=True
        )


def claim_pool(args: Any) -> None:
    """Launch hook (rank 0): reset every replica to base before the first publish, so
    a cold pool — or one still warm from a finished run — starts this run clean."""
    if sidecar_process.dist_rank() not in (None, 0):
        return
    claim_run(_store(args), _pool(args), _run_id(args))


# ── Staleness-gated rollout requests ──────────────────────────────────────────


async def gated_rollout_request_hook(
    args: Any, sample: Any, request: dict[str, Any]
) -> None:
    """Pin each request to a bounded-staleness version, so a too-stale replica returns
    a retryable 409 (nudging it to sync) instead of the trainer spending rollout
    compute on weights beyond its lag bound."""
    payload, headers = request["payload"], dict(request.get("headers") or {})
    mode = str(getattr(args, "rollout_request_weight_version_mode", "min"))
    affinity = str(
        getattr(args, "rollout_session_affinity_header", "x-session-affinity")
    )

    latest = exact = None
    lag = 0
    if mode != "none":
        floor = await _latest.get(args)
        lag = int(getattr(args, "rollout_request_weight_version_lag", 0))
        if mode == "exact":
            exact = max(0, floor - lag)
        else:
            latest = floor
    constrain_request(
        payload,
        headers,
        latest=latest,
        lag=lag,
        exact=exact,
        session_id=sample_affinity_key(sample),
        affinity_header=affinity,
    )
    request["headers"] = headers
    request["max_retries"] = int(
        getattr(args, "rollout_request_retry_attempts", request.get("max_retries", 60))
    )
    request["retry_sleep"] = float(
        getattr(args, "rollout_request_retry_sleep", request.get("retry_sleep", 1.0))
    )


class _CachedPointer:
    """TTL-cached ``latest`` version. The per-request hook gets no rollout id, so the
    staleness floor comes from the published pointer (already advanced by the publish
    hook), cached with a Volume reload so it isn't reloaded once per request."""

    def __init__(self) -> None:
        self._version = 0
        self._at = -1e9
        self._store: ModalVolumeStore | None = None

    async def get(self, args: Any, ttl: float = 2.0) -> int:
        store = self._store
        root = Path(_transport_root(args))
        run_id = _run_id(args)
        volume = _volume_name(args)
        if (
            store is None
            or store.root != root
            or store.run_id != run_id
            or store.volume_name != (volume or None)
        ):
            store = self._store = _store(args)
            self._version = 0
            self._at = -1e9
        now = time.monotonic()
        if now - self._at >= ttl:
            self._at = now
            try:
                # reload is blocking; keep the event loop free.
                await asyncio.to_thread(store.refresh)
                pointer = store.read_pointer()
                self._version = pointer.version if pointer else 0
            except Exception:  # noqa: BLE001
                logger.warning(
                    "gate: could not read latest; using cached %s",
                    self._version,
                    exc_info=True,
                )
        return self._version


_latest = _CachedPointer()


# ── args → run coordinates ────────────────────────────────────────────────────


def _store(args: Any) -> ModalVolumeStore:
    volume = _volume_name(args)
    return ModalVolumeStore(
        _transport_root(args),
        volume_name=volume or None,
        run_id=_run_id(args),
    )


def _pool(args: Any) -> ModalFlashPool:
    return ModalFlashPool(
        getattr(args, "rollout_modal_flash_app_name", None),
        getattr(args, "rollout_modal_flash_server_cls_name", None) or "Server",
    )


def _volume_name(args: Any) -> str | None:
    return getattr(args, "update_weight_delta_volume_name", None)


def _transport_root(args: Any) -> str:
    # miles owns <run>/updates (it may recreate the dir); stitch owns <run>/latest,
    # so the Store is rooted one level up from the trainer's write dir.
    write_dir = getattr(args, "update_weight_disk_dir", None)
    if not write_dir:
        raise ValueError("update_weight_disk_dir is required")
    write_dir = Path(write_dir)
    if write_dir.name != "updates":
        raise ValueError(
            f"update_weight_disk_dir must end in /updates: path={str(write_dir)!r}"
        )
    return str(write_dir.parent)


def _run_id(args: Any) -> str:
    run_id = getattr(args, "run_id", None)
    if not run_id:
        raise ValueError(
            "run_id is required (pass it via custom_config_path) — it is the run's fence token"
        )
    return str(run_id)
