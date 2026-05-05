"""Central gym server: REST API backed by modal.Dict stores.

All persistence for training runs, train results, eval results, and
deployments flows through this server. Model classes in
``modal_training_gym.common`` hit these endpoints via the HTTP client
in ``modal_training_gym.common.client``.

Deploy with:

    uv run modal deploy modal_training_gym/server.py
"""

from typing import Any

import modal

from modal_training_gym.common.run import TRAINING_RUNS_STORE_NAME
from modal_training_gym.common.train_result import TRAIN_RESULTS_STORE_NAME
from modal_training_gym.common.eval import EVAL_RESULTS_STORE_NAME
from modal_training_gym.common.deployment import DEPLOYMENTS_STORE_NAME

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.118.0",
        "modal",
    )
    .add_local_python_source("modal_training_gym", copy=True)
)

app = modal.App(
    "training-gym-server",
    image=image,
)


def _dict(name: str) -> "modal.Dict":
    return modal.Dict.from_name(name, create_if_missing=True)


def _volume(name: str) -> "modal.Volume":
    return modal.Volume.from_name(name, create_if_missing=True)


# TODO(joy): migrate to Modal Volume
def _save_to_volume(
    volume: str, namespace: str, key: str, value: dict[str, Any]
) -> None:
    import tempfile
    import os
    import json

    volume = _volume(volume)
    # Save to local file in temp dir
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, f"{key}.json")
    with open(temp_file, "w") as f:
        json.dump(value, f)
    volume.put(temp_file, name=f"{namespace}/{key}.json")


@app.function(
    min_containers=1,
)
@modal.asgi_app(
    custom_domains=["gymserver.modal.dev"],
)
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from modal_training_gym.common.deployment import ModelDeployment
    from modal_training_gym.common.eval import EvalResult
    from modal_training_gym.common.run import TrainingRun

    web = FastAPI()
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Training runs ────────────────────────────────────────────────────

    @web.post("/run")
    async def record_run(run: TrainingRun) -> None:
        store = _dict(TRAINING_RUNS_STORE_NAME)
        store[run.run_id] = run.model_dump(mode="json")

    @web.get("/runs")
    async def list_runs():
        store = _dict(TRAINING_RUNS_STORE_NAME)
        return JSONResponse(list(store.values()))

    @web.get("/runs/{run_id}")
    async def get_run(run_id: str):
        store = _dict(TRAINING_RUNS_STORE_NAME)
        try:
            return JSONResponse(store[run_id])
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Run {run_id!r} not found"
            ) from exc

    # ── Train results ────────────────────────────────────────────────────

    @web.post("/train-result")
    async def record_train_result(result: dict[str, Any]) -> None:
        store = _dict(TRAIN_RESULTS_STORE_NAME)
        training_run_id = result.get("training_run_id", "")
        if not training_run_id:
            raise HTTPException(status_code=422, detail="training_run_id is required")
        store[training_run_id] = result

    @web.get("/train-results")
    async def list_train_results():
        store = _dict(TRAIN_RESULTS_STORE_NAME)
        return JSONResponse(list(store.values()))

    @web.get("/train-results/{training_run_id}")
    async def get_train_result(training_run_id: str):
        store = _dict(TRAIN_RESULTS_STORE_NAME)
        try:
            return JSONResponse(store[training_run_id])
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"TrainResult {training_run_id!r} not found"
            ) from exc

    # ── Deployments ──────────────────────────────────────────────────────

    @web.post("/deployment")
    async def record_deployment(deployment: ModelDeployment) -> None:
        store = _dict(DEPLOYMENTS_STORE_NAME)
        store[deployment.app_name] = deployment.model_dump()

    @web.get("/deployments")
    async def list_deployments():
        store = _dict(DEPLOYMENTS_STORE_NAME)
        return JSONResponse(list(store.values()))

    # ── Eval results ─────────────────────────────────────────────────────

    @web.post("/eval")
    async def record_eval(eval_result: EvalResult) -> None:
        store = _dict(EVAL_RESULTS_STORE_NAME)
        store[eval_result.eval_id] = eval_result.model_dump()

    @web.get("/evals")
    async def list_evals():
        store = _dict(EVAL_RESULTS_STORE_NAME)
        return JSONResponse(list(store.values()))

    @web.get("/evals/{eval_id}")
    async def get_eval(eval_id: str):
        store = _dict(EVAL_RESULTS_STORE_NAME)
        try:
            return JSONResponse(store[eval_id])
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Eval {eval_id!r} not found"
            ) from exc

    return web
