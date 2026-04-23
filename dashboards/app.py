"""Modal-hosted Svelte + FastAPI dashboard for training-gym runs.

Shows all training runs (``_modal_job_type=training``), grouped into
collapsible sections by ``_modal_framework``.  Framework pills in the
toolbar let users toggle which sections are visible.

Architecture:
    The Svelte frontend is built at image-build time via ``npm install &&
    npm run build`` and served as static files by FastAPI.  A cron job
    (``refresh_training_metadata``) runs every 5 minutes, fetches all
    Modal apps, checks their tags, and writes training run metadata into a
    ``modal.Dict``.  The ASGI dashboard reads from the Dict for instant
    page loads.

Deploy with:

    uv run modal deploy dashboards/app.py

Modal will print the public URL; open it in a browser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import modal


frontend_path = Path(__file__).parent / "frontend"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
    )
    .pip_install(
        "fastapi[standard]==0.118.0",
        "modal",
    )
    .add_local_dir(str(frontend_path), remote_path="/app/frontend", copy=True)
    .run_commands(
        "cd /app/frontend && npm install && npm run build",
    )
    .add_local_python_source("modal_training_gym", copy=True)
)

OBS_VOLUME_NAME = "training-gym-harbor-observability"
obs_volume = modal.Volume.from_name(OBS_VOLUME_NAME, create_if_missing=True)
OBS_DIR = "/obs"

app = modal.App(
    "training-gym-dashboard",
    image=image,
)

training_runs_dict = modal.Dict.from_name("training-gym-runs", create_if_missing=True)

RUNS_KEY = "runs"
STATIC_DIR = "/app/frontend/dist"


@dataclass
class TrainingMetadata:
    framework: str = "(untagged)"


@dataclass
class TrainingRun:
    app_id: str
    name: str
    description: str
    state: str
    created_at: float
    stopped_at: float | None
    metadata: TrainingMetadata
    tags: dict[str, str] = field(default_factory=dict)


_FETCH_SCRIPT = r"""
import asyncio, json, sys

async def _fetch():
    from modal.client import _Client
    from modal_proto import api_pb2

    result = __import__("subprocess").run(
        ["modal", "app", "list", "--json"],
        capture_output=True, text=True,
    )
    app_list = json.loads(result.stdout)
    client = await _Client.from_env()

    runs = []
    for info in app_list:
        app_id = info["App ID"]
        try:
            resp = await client.stub.AppGetTags(
                api_pb2.AppGetTagsRequest(app_id=app_id)
            )
            tags = dict(resp.tags)
        except Exception:
            continue
        if tags.get("_modal_job_type") != "training":
            continue
        runs.append({
            "app_id": app_id,
            "name": info["Description"],
            "state": info.get("State", "unknown"),
            "created_at": info.get("Created at", ""),
            "stopped_at": info.get("Stopped at"),
            "framework": tags.get("_modal_framework", "(untagged)"),
            "tags": tags,
        })
    json.dump(runs, sys.stdout)

asyncio.run(_fetch())
"""


@app.function(
    schedule=modal.Period(minutes=5),
    timeout=120,
)
def refresh_training_metadata():
    """Cron job: runs a subprocess to list apps and fetch tags (avoids
    the container's gRPC event-loop mismatch), then writes to the Dict."""
    import json
    import subprocess
    from datetime import datetime, timezone

    result = subprocess.run(
        ["python", "-c", _FETCH_SCRIPT],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode != 0:
        print(f"Fetch script failed: {result.stderr}")
        return

    raw_runs = json.loads(result.stdout)

    def _parse_ts(s):
        if not s:
            return 0.0
        try:
            dt = datetime.fromisoformat(s)
            return dt.replace(tzinfo=dt.tzinfo or timezone.utc).timestamp()
        except (ValueError, TypeError):
            return 0.0

    runs: list[dict] = []
    for r in raw_runs:
        meta = TrainingMetadata(framework=r["framework"])
        run = TrainingRun(
            app_id=r["app_id"],
            name=r["name"],
            description=r["name"],
            state=r["state"].title(),
            created_at=_parse_ts(r["created_at"]),
            stopped_at=_parse_ts(r["stopped_at"]) if r["stopped_at"] else None,
            metadata=meta,
            tags=r["tags"],
        )
        runs.append(asdict(run))

    runs.sort(key=lambda r: r["created_at"], reverse=True)
    training_runs_dict[RUNS_KEY] = runs
    print(f"Refreshed: {len(runs)} training run(s)")


@app.function(
    volumes={OBS_DIR: obs_volume},
)
@modal.asgi_app()
def fastapi_app():
    import asyncio
    import json
    import time
    from pathlib import Path

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from modal_training_gym.frameworks.harbor.observability import (
        list_rollout_ids,
        list_run_manifests,
        load_rollout,
        summarize_run,
    )

    web = FastAPI()
    obs_path = Path(OBS_DIR)

    _RELOAD_INTERVAL = 5
    _last_reload = 0.0
    _reload_lock = asyncio.Lock()

    async def _maybe_reload():
        nonlocal _last_reload
        now = time.monotonic()
        if now - _last_reload < _RELOAD_INTERVAL:
            return
        async with _reload_lock:
            if time.monotonic() - _last_reload < _RELOAD_INTERVAL:
                return
            await obs_volume.reload.aio()
            _last_reload = time.monotonic()

    @web.get("/api/runs")
    async def api_runs() -> JSONResponse:
        try:
            runs = await training_runs_dict.get.aio(RUNS_KEY)
        except KeyError:
            runs = []
        return JSONResponse({"runs": runs if runs is not None else []})

    @web.get("/api/harbor/runs")
    async def api_harbor_runs():
        await _maybe_reload()
        return list_run_manifests(obs_path)

    @web.get("/api/harbor/runs/{run_id}")
    async def api_harbor_run(run_id: str):
        await _maybe_reload()
        try:
            return summarize_run(obs_path, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @web.get("/api/harbor/runs/{run_id}/rollouts")
    async def api_harbor_rollouts(run_id: str):
        await _maybe_reload()
        return list_rollout_ids(obs_path, run_id)

    @web.get("/api/harbor/runs/{run_id}/rollouts/{rollout_id}")
    async def api_harbor_rollout(run_id: str, rollout_id: str):
        await _maybe_reload()
        try:
            return load_rollout(obs_path, run_id, rollout_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    web.mount("/assets", StaticFiles(directory=f"{STATIC_DIR}/assets"), name="assets")

    @web.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(f"{STATIC_DIR}/index.html")

    return web
