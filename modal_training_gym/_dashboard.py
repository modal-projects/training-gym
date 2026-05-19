"""Self-contained training-gym dashboard app.

When deployed from a pip install (no local repo checkout), the image build
clones the frontend source from GitHub. When running from a repo checkout,
it uses the local ``dashboards/frontend`` directory instead.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from time import monotonic
from typing import Any, Awaitable, Callable

import modal

REPO_URL = "https://github.com/modal-projects/training-gym.git"
REPO_BRANCH = "main"

_repo_frontend = Path(__file__).resolve().parents[1] / "dashboards" / "frontend"
_has_local_frontend = _repo_frontend.is_dir()


def _build_image() -> modal.Image:
    base = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl")
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "apt-get install -y nodejs",
        )
        .pip_install("fastapi[standard]==0.118.0", "modal")
    )

    if _has_local_frontend:
        base = base.add_local_dir(
            str(_repo_frontend),
            remote_path="/app/frontend",
            copy=True,
            ignore=["node_modules", "dist"],
        )
    else:
        base = base.apt_install("git").run_commands(
            f"git clone --depth 1 -b {REPO_BRANCH} {REPO_URL} /tmp/training-gym",
            "mkdir -p /app && cp -r /tmp/training-gym/dashboards/frontend /app/frontend",
            "rm -rf /tmp/training-gym",
        )

    return base.add_local_python_source("modal_training_gym", copy=True).run_commands(
        "cd /app/frontend && npm install && npm run build",
    )


image = _build_image()

app = modal.App("training-gym-dashboard", image=image)

STATIC_DIR = "/app/frontend/dist"


@app.function(min_containers=1)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from modal_training_gym.common.modal_urls import modal_app_dashboard_url
    from modal_training_gym.utils.metadata import (
        MetadataStore,
        vol_get,
        vol_get_summary_items,
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

    def add_modal_app_urls(
        items: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        updated = []
        changed = False
        for item in items:
            new_item = dict(item)
            if not new_item.get("modal_app_url"):
                app_id = str(new_item.get("modal_app_id", "") or "")
                if app_id:
                    new_item["modal_app_url"] = modal_app_dashboard_url(app_id)
                    changed = True
            updated.append(new_item)
        return updated, changed

    async def load_eval_summaries() -> list[dict[str, Any]]:
        try:
            payload = await run_in_threadpool(
                vol_get, eval_summary_store, eval_summary_key
            )
        except KeyError:
            return []

        summaries = list_from_payload(payload, payload_key=eval_summary_payload_key)
        if not summaries:
            return []

        eval_ids = [
            str(s.get("eval_id", ""))
            for s in summaries
            if isinstance(s, dict) and s.get("eval_id")
        ]

        async def fetch_result(eval_id: str) -> tuple[str, dict | None]:
            try:
                r = await run_in_threadpool(
                    vol_get, MetadataStore.EVAL_RESULTS, eval_id
                )
                return eval_id, r
            except KeyError:
                return eval_id, None

        fetched = await asyncio.gather(*(fetch_result(eid) for eid in eval_ids))
        results_by_id = {eid: r for eid, r in fetched if r is not None}

        eval_config_ids = [
            str(s.get("eval_config_id", ""))
            for s in summaries
            if isinstance(s, dict) and s.get("eval_config_id")
        ]

        async def fetch_eval_config(
            eval_config_id: str,
        ) -> tuple[str, dict | None]:
            try:
                cfg = await run_in_threadpool(
                    vol_get, MetadataStore.EVAL_CONFIGS, eval_config_id
                )
                return eval_config_id, cfg
            except KeyError:
                return eval_config_id, None

        fetched_configs = await asyncio.gather(
            *(fetch_eval_config(cfg_id) for cfg_id in eval_config_ids)
        )
        configs_by_id = {
            cfg_id: cfg for cfg_id, cfg in fetched_configs if cfg is not None
        }

        enriched: list[dict[str, Any]] = []
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            eval_id = str(summary.get("eval_id", ""))
            eval_config_id = str(summary.get("eval_config_id", ""))
            result = results_by_id.get(eval_id)
            eval_config = configs_by_id.get(eval_config_id)
            if not result:
                merged = dict(summary)
                if eval_config:
                    merged["eval_config"] = eval_config
                    for field in (
                        "dataset_name",
                        "eval_fn_name",
                        "prompt_column",
                        "generate_kwargs",
                    ):
                        value = eval_config.get(field)
                        if field not in merged and value is not None:
                            merged[field] = value
                enriched.append(merged)
                continue
            merged = dict(summary)
            for field in ("deployment_id", "config", "status"):
                value = result.get(field)
                if field not in merged and value is not None:
                    merged[field] = value
            if eval_config:
                merged["eval_config"] = eval_config
                for field in (
                    "dataset_name",
                    "eval_fn_name",
                    "prompt_column",
                    "generate_kwargs",
                ):
                    value = eval_config.get(field)
                    if field not in merged and value is not None:
                        merged[field] = value
            enriched.append(merged)
        return enriched

    async def load_list_summary(
        summary_store: MetadataStore,
    ) -> list[dict[str, Any]]:
        items = await run_in_threadpool(vol_get_summary_items, summary_store)
        if items is None:
            return []
        items, changed = add_modal_app_urls(items)
        if changed:
            await run_in_threadpool(vol_put_summary_items, summary_store, items)
        return items

    async def load_runs() -> list[dict[str, Any]]:
        return await load_list_summary(MetadataStore.TRAINING_RUNS_SUMMARY)

    async def load_train_results() -> list[dict[str, Any]]:
        return await load_list_summary(MetadataStore.TRAIN_RESULTS_SUMMARY)

    async def load_deployments() -> list[dict[str, Any]]:
        return await load_list_summary(MetadataStore.DEPLOYMENTS_SUMMARY)

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
                status_code=404,
                detail=f"TrainResult {training_run_id!r} not found",
            )

    # ── Eval results ─────────────────────────────────────────────────────

    @web.get("/api/evals")
    async def evals():
        try:
            data = await get_cached_list("evals", load_eval_summaries)
        except Exception:
            data = []
        return JSONResponse(data)

    @web.get("/api/evals/{eval_id}")
    async def eval_detail(eval_id: str):
        try:
            data = await run_in_threadpool(
                vol_get, MetadataStore.EVAL_RESULTS, eval_id
            )
            return JSONResponse(data)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"EvalResult {eval_id!r} not found",
            )

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
