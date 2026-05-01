"""Modal-hosted Svelte + FastAPI dashboard for training-gym runs.

Shows all training runs from the ``TrainingRun`` store, grouped into
collapsible sections by framework. For each run, checks whether a
``TrainResult`` exists and displays it inline when available.

Architecture:
    The Svelte frontend is built at image-build time via ``npm install &&
    npm run build`` and served as static files by FastAPI.  The ASGI
    endpoint reads directly from the ``TrainingRun`` and ``TrainResult``
    modal.Dict stores on each request — no cron intermediary needed.

Deploy with:

    uv run modal deploy dashboards/app.py

Modal will print the public URL; open it in a browser.
"""

from __future__ import annotations

from dataclasses import asdict
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
        "wandb",
    )
    .add_local_dir(
        str(frontend_path),
        remote_path="/app/frontend",
        copy=True,
        ignore=["node_modules", "dist"],
    )
    .run_commands(
        "cd /app/frontend && npm install && npm run build",
    )
    .add_local_python_source("modal_training_gym", copy=True)
)

app = modal.App(
    "training-gym-dashboard",
    image=image,
)

STATIC_DIR = "/app/frontend/dist"


def _safe_str(v) -> str:
    if hasattr(v, "value"):
        return v.value
    return str(v) if v is not None else ""


def _extract_config_summary(config: dict) -> dict:
    if not config or not isinstance(config, dict):
        return {}
    model = config.get("model") or {}
    preset = config.get("preset") or {}
    wandb = config.get("wandb") or {}
    return {
        "model_name": _safe_str(model.get("model_name", "")),
        "gpu_type": _safe_str(preset.get("gpu_type", "")),
        "actor_num_nodes": preset.get("actor_num_nodes", 0),
        "actor_num_gpus_per_node": preset.get("actor_num_gpus_per_node", 0),
        "lr": config.get("lr", 0),
        "global_batch_size": config.get("global_batch_size", 0),
        "wandb_project": wandb.get("project", ""),
        "wandb_group": wandb.get("group", ""),
    }


def _summarize_result(result_dict: dict) -> dict:
    wandb_training_run_id = result_dict.get("wandb_training_run_id", "")
    wandb_project = result_dict.get("wandb_project", "")
    wandb_entity = result_dict.get("wandb_entity", "")

    wandb_url = None
    if wandb_training_run_id and wandb_project:
        entity = wandb_entity or "_"
        wandb_url = f"https://wandb.ai/{entity}/{wandb_project}/runs/{wandb_training_run_id}"

    return {
        "training_run_id": result_dict.get("training_run_id", ""),
        "app_name": result_dict.get("app_name", ""),
        "checkpoint_dir": result_dict.get("checkpoint_dir", ""),
        "wandb_url": wandb_url,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_training_run_id": wandb_training_run_id,
    }


def _fetch_runs() -> list[dict]:
    """Read TrainingRun and TrainResult stores, join by run ID."""
    from modal import Dict

    from modal_training_gym.common.run import TRAINING_RUNS_STORE_NAME
    from modal_training_gym.common.train_result import TRAIN_RESULTS_STORE_NAME

    runs_store = Dict.from_name(TRAINING_RUNS_STORE_NAME, create_if_missing=True)
    results_store = Dict.from_name(TRAIN_RESULTS_STORE_NAME, create_if_missing=True)

    results_by_id: dict[str, dict] = {}
    for result_dict in results_store.values():
        rid = result_dict.get("training_run_id")
        if rid:
            results_by_id[rid] = result_dict

    runs: list[dict] = []
    for run_dict in runs_store.values():
        run_id = run_dict.get("run_id", "")
        framework = _safe_str(run_dict.get("framework", ""))
        config = run_dict.get("config", {})
        train_result = results_by_id.get(run_id)

        runs.append({
            "run_id": run_id,
            "modal_app_id": run_dict.get("modal_app_id", ""),
            "framework": framework or "(untagged)",
            "config_summary": _extract_config_summary(config),
            "train_result": _summarize_result(train_result) if train_result else None,
        })

    return runs


@app.function()
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI()

    @web.get("/api/runs")
    async def api_runs() -> JSONResponse:
        runs = _fetch_runs()
        return JSONResponse({"runs": runs})

    # ── W&B metrics endpoints ──────────────────────────────────────────────

    @web.get("/api/wandb/{training_run_id}/metrics")
    async def api_wandb_metrics(training_run_id: str, keys: str = "", samples: int = 500):
        from modal_training_gym.common.train_result import TrainResult

        try:
            result = TrainResult.load(training_run_id)
        except (LookupError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not result.wandb_training_run_id:
            raise HTTPException(status_code=404, detail="No W&B run linked")

        key_list = [k.strip() for k in keys.split(",") if k.strip()] or None
        metrics = result.wandb_metrics(keys=key_list, samples=samples)
        return JSONResponse(
            {
                "training_run_id": training_run_id,
                "wandb_url": result.wandb_url(),
                "wandb_training_run_id": result.wandb_training_run_id,
                "metrics": metrics,
            }
        )

    @web.get("/api/wandb/{training_run_id}/summary")
    async def api_wandb_summary(training_run_id: str):
        from modal_training_gym.common.train_result import TrainResult

        try:
            result = TrainResult.load(training_run_id)
        except (LookupError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not result.wandb_training_run_id:
            raise HTTPException(status_code=404, detail="No W&B run linked")

        summary = result.wandb_summary()
        return JSONResponse(
            {
                "training_run_id": training_run_id,
                "wandb_url": result.wandb_url(),
                "summary": summary,
            }
        )

    @web.get("/api/train-results/{training_run_id}")
    async def api_train_result(training_run_id: str):
        from modal_training_gym.common.train_result import TrainResult

        try:
            result = TrainResult.load(training_run_id)
        except (LookupError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(asdict(result))

    web.mount("/assets", StaticFiles(directory=f"{STATIC_DIR}/assets"), name="assets")

    @web.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(f"{STATIC_DIR}/index.html")

    return web
