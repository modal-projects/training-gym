"""Modal app serving the Starlight documentation site.

Builds the site at image build time — runs all generators (API reference,
tutorial pages, README pages) then ``npm run build``. Serves the resulting
static files via FastAPI + StaticFiles.

Deploy with:

    uv run modal deploy docs-next/docs_next_app.py

Local development:

    cd docs-next && npm run dev
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs-next"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .run_commands("curl -fsSL https://deb.nodesource.com/setup_22.x | bash -")
    .apt_install("nodejs")
    .pip_install("fastapi[standard]==0.118.0", "pydantic>=2.0", "modal", "cloudpickle")
    .add_local_dir(
        str(DOCS_DIR),
        remote_path="/repo/docs-next",
        copy=True,
        ignore=["node_modules", ".astro", "dist"],
    )
    .add_local_dir(str(REPO_ROOT / "scripts"), remote_path="/repo/scripts", copy=True)
    .add_local_dir(str(REPO_ROOT / "modal_training_gym"), remote_path="/repo/modal_training_gym", copy=True)
    .add_local_dir(str(REPO_ROOT / "tutorials"), remote_path="/repo/tutorials", copy=True, ignore=["__pycache__"])
    .add_local_file(str(REPO_ROOT / "README.md"), remote_path="/repo/README.md", copy=True)
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), remote_path="/repo/pyproject.toml", copy=True)
    .run_commands(
        "cd /repo && python scripts/generate_docs_pages.py",
        "cd /repo && PYTHONPATH=/repo:/repo/scripts python scripts/generate_api_reference.py",
        "cd /repo && PYTHONPATH=/repo:/repo/scripts python scripts/generate_tutorial_pages.py",
        "cd /repo/docs-next && npm install && npm run build",
    )
)

app = modal.App("training-gym-docs", image=image)


@app.function(min_containers=1)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    web = FastAPI()
    dist_dir = Path("/repo/docs-next/dist")

    web.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")

    return web
