"""Modal app serving the Starlight documentation site.

Builds the site at image build time and serves the static files via
FastAPI + StaticFiles.

Deploy with:

    uv run modal deploy docs-next/docs_next_app.py

Local development:

    cd docs-next && npm run dev
"""

from __future__ import annotations

from pathlib import Path

import modal

DOCS_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash -")
    .apt_install("nodejs")
    .pip_install("fastapi[standard]==0.118.0")
    .add_local_dir(str(DOCS_DIR), remote_path="/docs-next", condition=lambda pth: ".astro" not in pth and "node_modules" not in pth and "dist" not in pth)
    .run_commands("cd /docs-next && npm install && npm run build")
)

app = modal.App("training-gym-docs", image=image)


@app.function(keep_warm=1, allow_concurrent_inputs=100)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    web = FastAPI()
    dist_dir = Path("/docs-next/dist")

    @web.get("/")
    async def index():
        return FileResponse(dist_dir / "index.html")

    web.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

    return web
