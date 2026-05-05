"""Static SPA host for the training-gym dashboard.

The Svelte frontend is built at image-build time. All data fetching
happens client-side against the central gym server at gymserver.modal.dev.

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
    .pip_install("fastapi[standard]==0.118.0")
    .add_local_dir(
        str(frontend_path),
        remote_path="/app/frontend",
        copy=True,
        ignore=["node_modules", "dist"],
    )
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
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI()

    web.mount("/assets", StaticFiles(directory=f"{STATIC_DIR}/assets"), name="assets")

    @web.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(f"{STATIC_DIR}/index.html")

    return web
