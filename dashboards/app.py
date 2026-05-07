"""Training-gym dashboard: SPA host + data API.

The Svelte frontend is built at image-build time. Data endpoints read
directly from the shared metadata volume via ``modal_training_gym.utils.metadata``.

Deploy with:

    uv run modal deploy dashboards/app.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import monotonic
from typing import Any, Awaitable, Callable

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


@app.function(min_containers=1)
@modal.asgi_app()
def fastapi_app():
    from fastapi.concurrency import run_in_threadpool
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from modal_training_gym.utils.metadata import (
        MetadataStore,
        SUMMARY_ITEMS_KEY,
        vol_get,
        vol_get_summary_items,
        vol_list,
        vol_put,
        vol_put_summary_items,
    )

    web = FastAPI()
    cache_ttl_seconds = 5.0
    eval_summary_store = MetadataStore.EVALS
    eval_summary_key = "summary"
    eval_summary_payload_key = "summaries"
    cache_entries: dict[str, tuple[float, list[dict[str, Any]]]] = {
        "runs": (0.0, []),
        "train_results": (0.0, []),
        "evals": (0.0, []),
        "deployments": (0.0, []),
    }
    cache_locks = {
        "runs": asyncio.Lock(),
        "train_results": asyncio.Lock(),
        "evals": asyncio.Lock(),
        "deployments": asyncio.Lock(),
    }

    web.mount("/assets", StaticFiles(directory=f"{STATIC_DIR}/assets"), name="assets")

    async def get_cached_list(
        key: str, loader: Callable[[], Awaitable[list[dict[str, Any]]]]
    ) -> list[dict[str, Any]]:
        now = monotonic()
        expires_at, values = cache_entries[key]
        if now < expires_at:
            return values

        async with cache_locks[key]:
            now = monotonic()
            expires_at, values = cache_entries[key]
            if now < expires_at:
                return values
            values = await loader()
            cache_entries[key] = (now + cache_ttl_seconds, values)
            return values

    def list_from_payload(
        payload: Any,
        *,
        payload_key: str,
    ) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            items = payload.get(payload_key, [])
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []

    def persist_eval_summaries(summaries: list[dict[str, Any]]) -> None:
        vol_put(
            eval_summary_store,
            eval_summary_key,
            {eval_summary_payload_key: summaries},
        )

    async def load_eval_summaries() -> list[dict[str, Any]]:
        try:
            payload = await run_in_threadpool(
                vol_get, eval_summary_store, eval_summary_key
            )
        except KeyError:
            payload = None
        else:
            if isinstance(payload, (dict, list)):
                return list_from_payload(
                    payload,
                    payload_key=eval_summary_payload_key,
                )

        legacy_summaries = await run_in_threadpool(vol_list, MetadataStore.EVAL_SUMMARIES)
        if legacy_summaries:
            await run_in_threadpool(persist_eval_summaries, legacy_summaries)
            return legacy_summaries

        eval_results = await run_in_threadpool(vol_list, MetadataStore.EVAL_RESULTS)
        generated_summaries: list[dict[str, Any]] = []
        for result in eval_results:
            rows = result.get("rows", [])
            total = len(rows) if isinstance(rows, list) else 0
            mean = (
                sum(row.get("score", 0) for row in rows) / total
                if total
                else 0.0
            )
            summary = {key: value for key, value in result.items() if key != "rows"}
            summary["total"] = total
            summary["mean"] = mean
            generated_summaries.append(summary)

        await run_in_threadpool(persist_eval_summaries, generated_summaries)
        return generated_summaries

    async def load_list_summary(
        *,
        summary_store: MetadataStore,
        detail_store: MetadataStore,
    ) -> list[dict[str, Any]]:
        items = await run_in_threadpool(vol_get_summary_items, summary_store)
        if items is not None:
            return items

        items = await run_in_threadpool(vol_list, detail_store)
        await run_in_threadpool(
            vol_put_summary_items,
            summary_store,
            items,
            key=eval_summary_key,
            payload_key=SUMMARY_ITEMS_KEY,
        )
        return items

    async def load_runs() -> list[dict[str, Any]]:
        return await load_list_summary(
            summary_store=MetadataStore.TRAINING_RUNS_SUMMARY,
            detail_store=MetadataStore.TRAINING_RUNS,
        )

    async def load_train_results() -> list[dict[str, Any]]:
        return await load_list_summary(
            summary_store=MetadataStore.TRAIN_RESULTS_SUMMARY,
            detail_store=MetadataStore.TRAIN_RESULTS,
        )

    async def load_deployments() -> list[dict[str, Any]]:
        return await load_list_summary(
            summary_store=MetadataStore.DEPLOYMENTS_SUMMARY,
            detail_store=MetadataStore.DEPLOYMENTS,
        )

    # ── Training runs ────────────────────────────────────────────────────

    @web.get("/api/runs")
    async def runs():
        try:
            data = await get_cached_list("runs", load_runs)
        except Exception:
            data = []
        return JSONResponse(data)

    # ── Train results ────────────────────────────────────────────────────

    @web.get("/api/train-results")
    async def train_results():
        try:
            data = await get_cached_list("train_results", load_train_results)
        except Exception:
            data = []
        return JSONResponse(data)

    @web.get("/api/train-results/{training_run_id}")
    async def train_result(training_run_id: str):
        try:
            data = await run_in_threadpool(
                vol_get, MetadataStore.TRAIN_RESULTS, training_run_id
            )
            return JSONResponse(data)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"TrainResult {training_run_id!r} not found"
            )

    # ── Eval results ─────────────────────────────────────────────────────

    @web.get("/api/evals")
    async def evals():
        try:
            data = await get_cached_list("evals", load_eval_summaries)
        except Exception:
            data = []
        return JSONResponse(data)

    # ── Deployments ──────────────────────────────────────────────────────

    @web.get("/api/deployments")
    async def deployments():
        try:
            data = await get_cached_list("deployments", load_deployments)
        except Exception:
            data = []
        return JSONResponse(data)

    # ── SPA fallback ─────────────────────────────────────────────────────

    @web.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(f"{STATIC_DIR}/index.html")

    return web
