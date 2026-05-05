"""Central gym server: REST API backed by modal.Volume.

Model classes in ``modal_training_gym.common`` persist directly to the
shared volume via ``modal_training_gym.common``. The REST API is kept
for the dashboard.

Deploy with:

    uv run modal deploy modal_training_gym/server.py
"""

import json
import os
from typing import Any

import modal

from modal_training_gym.common.run import TRAINING_RUNS_STORE_NAME
from modal_training_gym.common.train_result import TRAIN_RESULTS_STORE_NAME
from modal_training_gym.common.eval import EVAL_RESULTS_STORE_NAME
from modal_training_gym.common.deployment import DEPLOYMENTS_STORE_NAME
from modal_training_gym.common import METADATA_VOLUME_NAME

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]==0.118.0",
        "modal",
    )
    .add_local_python_source("modal_training_gym", copy=True)
)

METADATA_MOUNT_PATH = "/metadata"

metadata_volume = modal.Volume.from_name(METADATA_VOLUME_NAME, create_if_missing=True)

app = modal.App(
    "training-gym-server",
    image=image,
)


def _vol_put(store_name: str, key: str, value: dict[str, Any]) -> None:
    dir_path = f"{METADATA_MOUNT_PATH}/{store_name}"
    os.makedirs(dir_path, exist_ok=True)
    with open(f"{dir_path}/{key}.json", "w") as f:
        json.dump(value, f)
    metadata_volume.commit()


def _vol_get(store_name: str, key: str) -> dict[str, Any]:
    metadata_volume.reload()
    path = f"{METADATA_MOUNT_PATH}/{store_name}/{key}.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise KeyError(key)


def _vol_list(store_name: str) -> list[dict[str, Any]]:
    metadata_volume.reload()
    dir_path = f"{METADATA_MOUNT_PATH}/{store_name}"
    if not os.path.isdir(dir_path):
        return []
    results = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".json"):
            with open(os.path.join(dir_path, fname)) as f:
                results.append(json.load(f))
    return results


@app.function(
    min_containers=1,
    volumes={METADATA_MOUNT_PATH: metadata_volume},
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
        _vol_put(TRAINING_RUNS_STORE_NAME, run.run_id, run.model_dump(mode="json"))

    @web.get("/runs")
    async def list_runs():
        return JSONResponse(_vol_list(TRAINING_RUNS_STORE_NAME))

    @web.get("/runs/{run_id}")
    async def get_run(run_id: str):
        try:
            return JSONResponse(_vol_get(TRAINING_RUNS_STORE_NAME, run_id))
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Run {run_id!r} not found"
            ) from exc

    # ── Train results ────────────────────────────────────────────────────

    @web.post("/train-result")
    async def record_train_result(result: dict[str, Any]) -> None:
        training_run_id = result.get("training_run_id", "")
        if not training_run_id:
            raise HTTPException(status_code=422, detail="training_run_id is required")
        _vol_put(TRAIN_RESULTS_STORE_NAME, training_run_id, result)

    @web.get("/train-results")
    async def list_train_results():
        return JSONResponse(_vol_list(TRAIN_RESULTS_STORE_NAME))

    @web.get("/train-results/{training_run_id}")
    async def get_train_result(training_run_id: str):
        try:
            return JSONResponse(_vol_get(TRAIN_RESULTS_STORE_NAME, training_run_id))
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"TrainResult {training_run_id!r} not found"
            ) from exc

    # ── Deployments ──────────────────────────────────────────────────────

    @web.post("/deployment")
    async def record_deployment(deployment: ModelDeployment) -> None:
        _vol_put(
            DEPLOYMENTS_STORE_NAME,
            deployment.app_name,
            deployment.model_dump(mode="json"),
        )

    @web.get("/deployments")
    async def list_deployments():
        return JSONResponse(_vol_list(DEPLOYMENTS_STORE_NAME))

    # ── Eval results ─────────────────────────────────────────────────────

    @web.post("/eval")
    async def record_eval(eval_result: EvalResult) -> None:
        _vol_put(
            EVAL_RESULTS_STORE_NAME,
            eval_result.eval_id,
            eval_result.model_dump(mode="json"),
        )

    @web.get("/evals")
    async def list_evals():
        return JSONResponse(_vol_list(EVAL_RESULTS_STORE_NAME))

    @web.get("/evals/{eval_id}")
    async def get_eval(eval_id: str):
        try:
            return JSONResponse(_vol_get(EVAL_RESULTS_STORE_NAME, eval_id))
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Eval {eval_id!r} not found"
            ) from exc

    return web
