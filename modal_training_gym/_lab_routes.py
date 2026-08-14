"""Learning-agent (LAB) observability routes for the dashboard.

Optional add-on: reads the run records that the LAB observatory ingestion
pipeline (learning_agent_bench/observatory) writes to its own Modal Volume
and serves the agent's learning research log alongside the training-gym
data. Strictly read-only — the observatory pipeline remains the only writer.

Volume layout read here (documented in observatory/DESIGN.md):

    runs/<run_id>/record.json   # full run record; index_row + scores.learning_log
    runs/<run_id>/status.json   # tiny {state, updated_at, num_events, ...}

The volume is MOUNTED on the dashboard function (see ``lab_obs_mount()`` and
its use in ``_dashboard.py``) and read through the filesystem with a
throttled ``Volume.reload()`` — the same pattern the observatory viewer
uses. Client-API streaming reads (``Volume.read_file``) are deliberately not
used: live runs rewrite record.json every few seconds, and a streaming read
of a hot file loses the race with the writer ("block not found") almost
every time. Mounted reads serve a consistent snapshot between reloads.

Configuration comes from env vars captured into the image at deploy time:

    LAB_OBS_VOLUME  observatory volume name; unset disables the whole feature
    LAB_OBS_URL     deployed observatory base URL, used for deep-dive links

Every route degrades to empty data when the mount is absent, so upstream
deployments without a LAB volume are unaffected.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import modal

LAB_OBS_VOLUME_ENV_KEY = "LAB_OBS_VOLUME"
LAB_OBS_URL_ENV_KEY = "LAB_OBS_URL"

LAB_OBS_MOUNT_PATH = "/lab_obs"
# Local-dev override (e.g. a stub server pointing at a directory of runs).
LAB_OBS_MOUNT_PATH_ENV_KEY = "LAB_OBS_MOUNT_PATH"

# Same charset the observatory viewer enforces for run ids.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# record.json sections dropped from the detail response. `events` alone is
# ~800 KB per run and these pages render neither traces nor telemetry — the
# observatory run view stays the deep-dive surface for both.
_DETAIL_EXCLUDED_SECTIONS = frozenset({"events", "system_monitor"})

_LIST_CACHE_TTL_SECONDS = 15.0
_RELOAD_EVERY_SECONDS = 10.0

# A run that says "running" but whose watcher heartbeat (status.json
# updated_at) is this old has lost its container. Same rule and threshold as
# observatory/app.py so both viewers agree on what counts as live.
_STALE_AFTER_SECONDS = 15 * 60

_EVENTS_DEFAULT_LIMIT = 100
_EVENTS_MAX_LIMIT = 500


def lab_obs_volume_name() -> str:
    return os.environ.get(LAB_OBS_VOLUME_ENV_KEY, "")


def lab_obs_enabled() -> bool:
    return bool(lab_obs_volume_name())


def lab_obs_mount() -> dict[str, modal.Volume]:
    """Volume mount for the dashboard function; empty when not configured."""
    if not lab_obs_enabled():
        return {}
    return {LAB_OBS_MOUNT_PATH: modal.Volume.from_name(lab_obs_volume_name())}


def lab_obs_url() -> str:
    return os.environ.get(LAB_OBS_URL_ENV_KEY, "").rstrip("/")


def _obs_run_url(run_id: str) -> str | None:
    base = lab_obs_url()
    return f"{base}/run?id={run_id}" if base else None


def _runs_root() -> Path:
    return Path(os.environ.get(LAB_OBS_MOUNT_PATH_ENV_KEY, LAB_OBS_MOUNT_PATH)) / "runs"


def _read_json_file(path: Path) -> Any | None:
    """Parse one JSON file; None when missing, replaced mid-read, or invalid."""
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _apply_staleness(row: dict[str, Any]) -> None:
    """Downgrade a dead-heartbeat "running" state to "stale", in place."""
    if row.get("state") != "running":
        return
    stamp = None
    ts = row.get("updated_at")
    if isinstance(ts, str) and ts:
        try:
            from datetime import datetime

            stamp = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            stamp = None
    if stamp is None or (time.time() - stamp) > _STALE_AFTER_SECONDS:
        row["state"] = "stale"


class LabRunStore:
    """Read-side cache over the mounted observatory volume.

    Index rows and status blobs are cached per file mtime, so a refresh
    re-parses only what changed — finished runs are read once, live runs on
    each record update. The assembled list is additionally cached for
    ``_LIST_CACHE_TTL_SECONDS`` so the frontend's 5s poll doesn't rescan the
    tree every time. All methods are synchronous (mounted-FS I/O); routes
    call them via ``asyncio.to_thread``.
    """

    def __init__(self) -> None:
        self._index_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._status_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._list_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
        # Single-slot cache of one run's parsed trace events (run_id, mtime,
        # events) — bounds memory while making the common browse pattern
        # (page through one run's trace) parse record.json once per update.
        self._events_cache: tuple[str, float, list[Any]] | None = None
        # Single-slot cache of one run's parsed workspace.json (run_id,
        # mtime, tree-sans-content, {path: entry-with-content}). The raw file
        # is tens of MB, so it must not be re-parsed per click.
        self._workspace_cache: (
            tuple[str, float, dict[str, Any], dict[str, dict[str, Any]]] | None
        ) = None
        self._last_reload = 0.0

    def _maybe_reload(self) -> None:
        """Throttled Volume.reload() so mounted reads see recent writes.

        No-op outside a Modal container (local dev reads a plain directory).
        """
        if not lab_obs_enabled() or not os.environ.get("MODAL_IS_REMOTE"):
            return
        now = time.monotonic()
        if now - self._last_reload < _RELOAD_EVERY_SECONDS:
            return
        self._last_reload = now
        try:
            modal.Volume.from_name(lab_obs_volume_name()).reload()
        except Exception:
            pass

    def list_runs(self) -> list[dict[str, Any]]:
        expires_at, rows = self._list_cache
        if time.monotonic() < expires_at:
            return rows
        try:
            rows = self._build_run_list()
        except Exception:
            # Serve the last known list on transient errors and back off a
            # full TTL rather than failing the endpoint.
            rows = self._list_cache[1]
        self._list_cache = (time.monotonic() + _LIST_CACHE_TTL_SECONDS, rows)
        return rows

    def _build_run_list(self) -> list[dict[str, Any]]:
        self._maybe_reload()
        root = _runs_root()
        if not root.is_dir():
            return []
        rows = []
        for entry in root.iterdir():
            if not _RUN_ID_RE.match(entry.name) or not entry.is_dir():
                continue
            row = self._load_index_row(entry)
            if row is not None:
                rows.append(row)
        rows.sort(key=lambda row: str(row.get("launched_at") or ""), reverse=True)
        return rows

    def _load_index_row(self, run_dir: Path) -> dict[str, Any] | None:
        run_id = run_dir.name
        record_path = run_dir / "record.json"
        try:
            record_mtime = record_path.stat().st_mtime
        except OSError:
            return None

        cached = self._index_cache.get(run_id)
        if cached is not None and cached[0] == record_mtime:
            row = dict(cached[1])
        else:
            record = _read_json_file(record_path)
            if not isinstance(record, dict):
                # Replaced mid-read or unreadable: fall back to the cached
                # row rather than dropping the run from the list.
                if cached is None:
                    return None
                row = dict(cached[1])
            else:
                index_row = record.get("index_row")
                if not isinstance(index_row, dict):
                    return None
                index_row = dict(index_row)
                # The record is already parsed here, so surface the research
                # log size on the list row (index_row itself doesn't carry it).
                log = (record.get("scores") or {}).get("learning_log")
                index_row["learning_log_entries"] = (
                    len(log) if isinstance(log, list) else 0
                )
                self._index_cache[run_id] = (record_mtime, index_row)
                row = dict(index_row)

        row.setdefault("run_id", run_id)
        status = self._load_status(run_dir)
        if status is not None:
            # status.json is written more often than record.json; its
            # liveness fields win, mirroring the observatory index view.
            for key in ("state", "updated_at", "num_events", "exit_code"):
                if status.get(key) is not None:
                    row[key] = status[key]
            if status.get("last_event_ts") is not None:
                row["last_event_ts"] = status["last_event_ts"]
        _apply_staleness(row)
        row["obs_url"] = _obs_run_url(str(row.get("run_id", run_id)))
        return row

    def _load_status(self, run_dir: Path) -> dict[str, Any] | None:
        run_id = run_dir.name
        status_path = run_dir / "status.json"
        try:
            status_mtime = status_path.stat().st_mtime
        except OSError:
            return None
        cached = self._status_cache.get(run_id)
        if cached is not None and cached[0] == status_mtime:
            return cached[1]
        status = _read_json_file(status_path)
        if not isinstance(status, dict):
            return cached[1] if cached is not None else None
        self._status_cache[run_id] = (status_mtime, status)
        return status

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        if not _RUN_ID_RE.match(run_id):
            return None
        self._maybe_reload()
        run_dir = _runs_root() / run_id
        record = _read_json_file(run_dir / "record.json")
        if not isinstance(record, dict):
            # One immediate retry: a mid-read replacement resolves instantly
            # on the mounted snapshot.
            record = _read_json_file(run_dir / "record.json")
        if not isinstance(record, dict):
            return None

        detail = {
            key: value
            for key, value in record.items()
            if key not in _DETAIL_EXCLUDED_SECTIONS
        }
        scores = detail.get("scores")
        if isinstance(scores, dict) and isinstance(scores.get("results"), list):
            # per_question can dwarf the rest of the payload on finished runs
            # and nothing here renders it.
            scores = dict(scores)
            scores["results"] = [
                {k: v for k, v in entry.items() if k != "per_question"}
                if isinstance(entry, dict)
                else entry
                for entry in scores["results"]
            ]
            detail["scores"] = scores

        status = _read_json_file(run_dir / "status.json")
        if isinstance(status, dict):
            _apply_staleness(status)
            detail["status"] = status
        detail["obs_url"] = _obs_run_url(run_id)
        return detail

    def get_events(
        self, run_id: str, offset: int | None, limit: int
    ) -> dict[str, Any] | None:
        """One page of the run's normalized trace events.

        ``offset`` is the absolute index of the first event returned; None
        means "the tail" (the newest ``limit`` events). The full events list
        is the heavy part of record.json (~800 KB and growing on live runs),
        which is why it's paged here instead of shipped with the detail.
        """
        if not _RUN_ID_RE.match(run_id):
            return None
        self._maybe_reload()
        record_path = _runs_root() / run_id / "record.json"
        try:
            record_mtime = record_path.stat().st_mtime
        except OSError:
            return None

        cached = self._events_cache
        if cached is not None and cached[0] == run_id and cached[1] == record_mtime:
            events = cached[2]
        else:
            record = _read_json_file(record_path)
            if not isinstance(record, dict):
                record = _read_json_file(record_path)
            if not isinstance(record, dict):
                return None
            raw = record.get("events")
            events = raw if isinstance(raw, list) else []
            self._events_cache = (run_id, record_mtime, events)

        total = len(events)
        limit = max(1, min(limit, _EVENTS_MAX_LIMIT))
        if offset is None:
            offset = max(0, total - limit)
        offset = max(0, min(offset, total))
        return {
            "total": total,
            "offset": offset,
            "events": events[offset : offset + limit],
        }

    def get_monitor(self, run_id: str) -> list[dict[str, Any]] | None:
        """System-monitor samples (agent CPU container telemetry), downsampled.

        The samples live in record.json; ~400 points is plenty for the rail
        charts, so long runs are strided down before shipping.
        """
        if not _RUN_ID_RE.match(run_id):
            return None
        self._maybe_reload()
        record = _read_json_file(_runs_root() / run_id / "record.json")
        if not isinstance(record, dict):
            return None
        samples = record.get("system_monitor")
        if not isinstance(samples, list):
            return []
        stride = max(1, len(samples) // 400)
        return [s for s in samples[::stride] if isinstance(s, dict)]

    def _load_workspace(
        self, run_id: str
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
        ws_path = _runs_root() / run_id / "workspace.json"
        try:
            ws_mtime = ws_path.stat().st_mtime
        except OSError:
            return None
        cached = self._workspace_cache
        if cached is not None and cached[0] == run_id and cached[1] == ws_mtime:
            return cached[2], cached[3]
        ws = _read_json_file(ws_path)
        if not isinstance(ws, dict):
            return None
        files = ws.get("files")
        files = files if isinstance(files, list) else []
        by_path: dict[str, dict[str, Any]] = {}
        tree_files = []
        for entry in files:
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            by_path[str(entry["path"])] = entry
            tree_files.append(
                {k: entry.get(k) for k in ("path", "size", "inline", "truncated")}
            )
        tree = {
            key: ws.get(key)
            for key in ("built_at", "root", "total_files", "total_bytes", "inlined_files")
        }
        tree["files"] = tree_files
        self._workspace_cache = (run_id, ws_mtime, tree, by_path)
        return tree, by_path

    def get_workspace(self, run_id: str) -> dict[str, Any] | None:
        """The workspace snapshot's file tree, contents stripped."""
        if not _RUN_ID_RE.match(run_id):
            return None
        self._maybe_reload()
        loaded = self._load_workspace(run_id)
        return loaded[0] if loaded else None

    def get_workspace_file(self, run_id: str, path: str) -> dict[str, Any] | None:
        """One workspace file's snapshot entry, inlined content included."""
        if not _RUN_ID_RE.match(run_id):
            return None
        self._maybe_reload()
        loaded = self._load_workspace(run_id)
        if not loaded:
            return None
        return loaded[1].get(path)

    def get_trajectory(self, run_id: str) -> dict[str, Any] | None:
        """The full trace as a Harbor ATIF-v1.7 trajectory (see _lab_atif)."""
        if not _RUN_ID_RE.match(run_id):
            return None
        self._maybe_reload()
        record_path = _runs_root() / run_id / "record.json"
        record = _read_json_file(record_path)
        if not isinstance(record, dict):
            record = _read_json_file(record_path)
        if not isinstance(record, dict):
            return None

        from modal_training_gym._lab_atif import events_to_atif

        raw = record.get("events")
        return events_to_atif(record, raw if isinstance(raw, list) else [])


def register_lab_routes(web: Any) -> None:
    """Attach the learning-agent routes to the dashboard FastAPI app.

    Called from ``fastapi_app()`` in ``_dashboard.py``. The routes inherit
    the app-wide password middleware (they are intentionally NOT in
    ``PASSWORD_EXEMPT_PATHS`` — that set is for token-authenticated write
    endpoints only).
    """
    from fastapi import HTTPException

    store = LabRunStore()

    @web.get("/api/learning-runs")
    async def learning_runs() -> list[dict[str, Any]]:
        return await asyncio.to_thread(store.list_runs)

    @web.get("/api/learning-runs/{run_id}")
    async def learning_run_detail(run_id: str) -> dict[str, Any]:
        detail = await asyncio.to_thread(store.get_run, run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return detail

    @web.get("/api/learning-runs/{run_id}/events")
    async def learning_run_events(
        run_id: str,
        offset: int | None = None,
        limit: int = _EVENTS_DEFAULT_LIMIT,
    ) -> dict[str, Any]:
        page = await asyncio.to_thread(store.get_events, run_id, offset, limit)
        if page is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return page

    @web.get("/api/learning-runs/{run_id}/monitor")
    async def learning_run_monitor(run_id: str) -> list[dict[str, Any]]:
        samples = await asyncio.to_thread(store.get_monitor, run_id)
        if samples is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        return samples

    @web.get("/api/learning-runs/{run_id}/workspace")
    async def learning_run_workspace(run_id: str) -> dict[str, Any]:
        tree = await asyncio.to_thread(store.get_workspace, run_id)
        if tree is None:
            raise HTTPException(
                status_code=404, detail=f"no workspace snapshot for {run_id!r}"
            )
        return tree

    @web.get("/api/learning-runs/{run_id}/workspace/file")
    async def learning_run_workspace_file(run_id: str, path: str) -> dict[str, Any]:
        entry = await asyncio.to_thread(store.get_workspace_file, run_id, path)
        if entry is None:
            raise HTTPException(
                status_code=404, detail=f"no workspace file {path!r} in {run_id!r}"
            )
        return entry

    @web.get("/api/learning-runs/{run_id}/trajectory")
    async def learning_run_trajectory(run_id: str, download: bool = False):
        from fastapi.responses import JSONResponse

        trajectory = await asyncio.to_thread(store.get_trajectory, run_id)
        if trajectory is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        headers = {}
        if download:
            # ATIF's standard file name is trajectory.json; prefix the run id
            # so multiple downloads stay distinguishable.
            headers["Content-Disposition"] = (
                f'attachment; filename="{run_id}.trajectory.json"'
            )
        return JSONResponse(trajectory, headers=headers)
