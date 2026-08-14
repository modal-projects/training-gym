#!/usr/bin/env python3
"""Learning Agent Observatory web app — read-only viewer over the observability volume.

One FastAPI factory (build_asgi), two modes:

  Modal (deployed):  MODAL_ENVIRONMENT=junlin-dev modal deploy observatory/app.py
      app "lab-observatory", volume $MODAL_OBS_VOLUME mounted at /obs,
      static/ + schema.py baked into the image. Public URL, no auth (v1).
  Local dev:         python3 observatory/app.py --data-dir /tmp/obs_stage [--port 8900]
      uvicorn against any directory shaped like the volume
      (runs/<run_id>/record.json ...). OBS_DATA_DIR is the --data-dir default.

API: see DESIGN.md. The app never writes; volume freshness comes from a
throttled volume.reload() per request (no-op in local mode).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Dual-path schema import: repo checkout vs image copy (/root/obs_schema.py).
if (_HERE / "schema.py").is_file() and str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
try:
    from observatory import schema
except ImportError:
    import obs_schema as schema

NAME_RE = re.compile(r"[A-Za-z0-9._-]+")


def _safe_name(name: str) -> bool:
    return bool(NAME_RE.fullmatch(name)) and ".." not in name


# Volume freshness (Modal only; local mode leaves _volume None).
_volume = None
_last_reload = 0.0
RELOAD_EVERY_S = 5.0


def _maybe_reload() -> None:
    global _last_reload
    if _volume is None:
        return
    now = time.monotonic()
    if now - _last_reload < RELOAD_EVERY_S:
        return
    _last_reload = now
    try:
        _volume.reload()
    except Exception:
        pass  # a stale read beats a 500; next request retries


# Index cache: record.json path -> (mtime, index_row).
_index_cache: dict[str, tuple[float, dict]] = {}

# Operator test scores: leaderboard.jsonl at the volume root, pushed by
# `python3 -m observatory.cli sync-scores`. Rows carrying run_id +
# submission_mean are run-level operator scorings; /api/runs overlays them so
# the table shows self-reported dev and official test side by side.
_lb_cache: tuple[float, dict] | None = None


def _test_overlays(root: Path) -> dict:
    global _lb_cache
    p = root / "leaderboard.jsonl"
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    if _lb_cache is None or _lb_cache[0] != mtime:
        overlays: dict[str, dict] = {}
        for line in p.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rid = row.get("run_id")
            if not rid or not isinstance(row.get("submission_mean"), (int, float)):
                continue
            overlays[rid] = {          # later rows win: re-scorings supersede
                "test_score": row["submission_mean"],
                "test_ci": row.get("submission_ci95"),
                "test_margin": row.get("margin"),
                "test_judge": row.get("judge_model"),
                "test_canonical": row.get("canonical"),
            }
        _lb_cache = (mtime, overlays)
    return dict(_lb_cache[1])


# Metered GPU-hours: gpu_metered.json at the volume root, pushed by
# `python3 -m observatory.gpu_metering` (control-plane truth; GPU_LOG.jsonl
# stays visible as the agent's self-report).
_gpu_cache: tuple[float, dict] | None = None


def _gpu_overlays(root: Path) -> dict:
    global _gpu_cache
    p = root / "gpu_metered.json"
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return {}
    if _gpu_cache is None or _gpu_cache[0] != mtime:
        data = _read_json(p)
        runs = data.get("runs") if isinstance(data, dict) else None
        overlays = {}
        for rid, e in (runs or {}).items():
            if isinstance(e, dict) and isinstance(e.get("gpu_hours"), (int, float)):
                overlays[rid] = {"gpu_hours_metered": e["gpu_hours"],
                                 "gpu_metered_shared": bool(e.get("shared_window"))}
        _gpu_cache = (mtime, overlays)
    return dict(_gpu_cache[1])


def _read_json(path: Path):
    try:
        with open(path, "rb") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _index_row(rec: Path):
    """index_row of one record.json, cached per mtime. Returns a copy."""
    try:
        mtime = rec.stat().st_mtime
    except OSError:
        return None
    cached = _index_cache.get(str(rec))
    if cached is None or cached[0] != mtime:
        data = _read_json(rec)
        if not isinstance(data, dict):
            return None
        cached = (mtime, data.get("index_row") or {})
        _index_cache[str(rec)] = cached
    return dict(cached[1])


# A run that says "running" but whose watcher heartbeat (status.json
# updated_at, refreshed every ingest) is this old has lost its container —
# killed mid-run or crashed before solve_status.txt was written. Present it
# as "stale" rather than an eternal "running".
_STALE_AFTER_S = 15 * 60


def _apply_staleness(row: dict) -> None:
    if row.get("state") != "running":
        return
    ts = row.get("updated_at")
    stamp = None
    if isinstance(ts, str) and ts:
        try:
            from datetime import datetime
            stamp = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except ValueError:
            stamp = None
    if stamp is None or (time.time() - stamp) > _STALE_AFTER_S:
        row["state"] = "stale"


def _static_dir() -> Path:
    baked = Path("/root/static")
    return baked if baked.is_dir() else _HERE / "static"


def build_asgi(data_root: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import (FileResponse, PlainTextResponse,
                                   RedirectResponse, Response)

    root = Path(data_root)
    static = _static_dir()
    api = FastAPI(title="lab-observatory", docs_url=None, redoc_url=None)

    @api.middleware("http")
    async def _fresh(request, call_next):
        _maybe_reload()
        resp = await call_next(request)
        # every page/API response revalidates (ETag/re-fetch) — stale browser
        # copies of the index or /api/runs hid new runs and UI fixes
        resp.headers.setdefault("Cache-Control", "no-cache")
        return resp

    def _run_dir(run_id: str) -> Path:
        if not _safe_name(run_id):
            raise HTTPException(status_code=404, detail="invalid run_id")
        return root / schema.RUNS_PREFIX / run_id

    def _json_file(path: Path, what: str) -> Response:
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"{what} not found")
        return Response(path.read_bytes(), media_type="application/json")

    @api.get("/healthz")
    def healthz():
        return {"ok": True}

    @api.get("/api/runs")
    def list_runs():
        rows = []
        runs = root / schema.RUNS_PREFIX
        overlays = _test_overlays(root)
        gpu_overlays = _gpu_overlays(root)
        if runs.is_dir():
            for rec in runs.glob(f"*/{schema.RECORD_FILE}"):
                row = _index_row(rec)
                if row is None:
                    continue
                status = _read_json(rec.parent / schema.STATUS_FILE)
                if isinstance(status, dict):
                    for k in ("state", "updated_at", "num_events"):
                        if k in status:
                            row[k] = status[k]
                _apply_staleness(row)
                ov = overlays.get(row.get("run_id"))
                if ov:
                    row.update(ov)
                gv = gpu_overlays.get(row.get("run_id"))
                if gv:
                    row.update(gv)
                rows.append(row)
        rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
        return rows

    @api.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        path = _run_dir(run_id) / schema.RECORD_FILE
        gv = _gpu_overlays(root).get(run_id)
        if gv:  # merge path: parse, overlay, re-serialize
            rec = _read_json(path)
            if isinstance(rec, dict):
                if isinstance(rec.get("index_row"), dict):
                    rec["index_row"].update(gv)
                return Response(json.dumps(rec), media_type="application/json")
        return _json_file(path, f"run {run_id}")  # fast raw-bytes path

    @api.get("/api/runs/{run_id}/status")
    def get_status(run_id: str):
        st = _read_json(_run_dir(run_id) / schema.STATUS_FILE)
        if not isinstance(st, dict):
            raise HTTPException(status_code=404, detail=f"status of {run_id} not found")
        _apply_staleness(st)
        return st

    @api.get("/api/runs/{run_id}/workspace")
    def get_workspace(run_id: str):
        return _json_file(_run_dir(run_id) / schema.WORKSPACE_FILE, f"workspace of {run_id}")

    @api.get("/api/runs/{run_id}/raw/{name}")
    def get_raw(run_id: str, name: str):
        if not _safe_name(name):
            raise HTTPException(status_code=404, detail="invalid artifact name")
        path = _run_dir(run_id) / schema.RAW_DIR / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"raw/{name} not found for {run_id}")
        # binary artifacts (workspace.tar.gz) must not go through text decoding
        if name.endswith((".tar.gz", ".tgz", ".gz", ".zip")):
            return FileResponse(path, filename=name, media_type="application/octet-stream")
        return PlainTextResponse(path.read_text(errors="replace"))

    def _page(name: str):
        path = static / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"{name} missing (frontend not built)")
        return FileResponse(path)

    @api.get("/")
    def index_page():
        return _page("index.html")

    @api.get("/run")
    def run_page():
        return _page("run.html")

    # Docs: one file per page, docs-<slug>.html; /docs is the introduction.
    DOCS_PAGES = ("workflow", "tasks", "agent", "toolbox", "training", "eval", "integrity", "runs")

    @api.get("/docs")
    def docs_index():
        return _page("docs.html")

    @api.get("/docs/{page}")
    def docs_page(page: str):
        if page not in DOCS_PAGES:
            raise HTTPException(status_code=404, detail=f"no docs page '{page}'")
        return _page(f"docs-{page}.html")

    @api.get("/tools")
    def tools_page():
        return _page("tools.html")

    @api.get("/how")
    def how_page():
        # the old single-page walkthrough grew into the docs site
        return RedirectResponse("/docs", status_code=307)

    @api.get("/assets/{name:path}")
    def assets(name: str):
        # Serve static/assets/<name> when that layout exists, else static/<name>.
        base = (static / "assets") if (static / "assets").is_dir() else static
        base = base.resolve()
        path = (base / name).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        # no-cache = revalidate every load (ETag makes that a cheap 304).
        # Without it browsers heuristically cache off Last-Modified and keep
        # serving a stale run.js/styles.css long after a redeploy.
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    return api


# --- Modal deployment surface (import stays optional for local dev) ---
try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("fastapi[standard]")
        .add_local_dir(str(_HERE / "static"), remote_path="/root/static")
        .add_local_file(str(_HERE / "schema.py"), remote_path="/root/obs_schema.py")
    )
    volume = modal.Volume.from_name(
        os.environ.get("MODAL_OBS_VOLUME", "lab-observatory"), create_if_missing=True
    )
    app = modal.App("lab-observatory")

    @app.function(image=image, volumes={"/obs": volume}, scaledown_window=300)
    @modal.asgi_app()
    def web():
        global _volume
        _volume = volume
        return build_asgi("/obs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Observatory viewer, local dev mode (no volume).")
    ap.add_argument("--data-dir", default=os.environ.get("OBS_DATA_DIR") or None,
                    help="dir shaped like the volume (runs/<run_id>/record.json ...); "
                         "default $OBS_DATA_DIR")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if not args.data_dir:
        ap.error("--data-dir (or OBS_DATA_DIR) is required")
    import uvicorn
    uvicorn.run(build_asgi(args.data_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
