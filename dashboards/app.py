"""Training-gym dashboard: SPA host + data API.

The Svelte frontend is built at image-build time. Data endpoints read
directly from the shared metadata volume via ``modal_training_gym.utils.metadata``.

Deploy with:

    uv run modal deploy dashboards/app.py
"""

from __future__ import annotations

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
    .pip_install("fastapi[standard]==0.118.0", "modal")
    .add_local_dir(
        str(frontend_path),
        remote_path="/app/frontend",
        copy=True,
        ignore=["node_modules", "dist"],
    )
    .add_local_python_source("modal_training_gym", copy=True)
    .run_commands(
        "cd /app/frontend && npm install && npm run build",
    )
)

app = modal.App(
    "training-gym-dashboard",
    image=image,
)

STATIC_DIR = "/app/frontend/dist"


@app.function()
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from modal_training_gym.utils.metadata import MetadataStore, vol_get, vol_list

    web = FastAPI()

    web.mount("/assets", StaticFiles(directory=f"{STATIC_DIR}/assets"), name="assets")

    # ── Training runs ────────────────────────────────────────────────────

    @web.get("/api/runs")
    async def runs():
        return JSONResponse(vol_list(MetadataStore.TRAINING_RUNS))

    # ── Train results ────────────────────────────────────────────────────

    @web.get("/api/train-results")
    async def train_results():
        return JSONResponse(vol_list(MetadataStore.TRAIN_RESULTS))

    @web.get("/api/train-results/{training_run_id}")
    async def train_result(training_run_id: str):
        try:
            return JSONResponse(vol_get(MetadataStore.TRAIN_RESULTS, training_run_id))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"TrainResult {training_run_id!r} not found")

    # ── Eval results ─────────────────────────────────────────────────────

    @web.get("/api/evals")
    async def evals():
        return JSONResponse(vol_list(MetadataStore.EVAL_RESULTS))

    # ── Deployments ──────────────────────────────────────────────────────

    @web.get("/api/deployments")
    async def deployments():
        return JSONResponse(vol_list(MetadataStore.DEPLOYMENTS))

    # ── SPA fallback ─────────────────────────────────────────────────────

    @web.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(f"{STATIC_DIR}/index.html")

    return web
