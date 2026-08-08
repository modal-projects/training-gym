"""Self-contained training-gym dashboard app.

When deployed from a pip install (no local repo checkout), the image build
clones the frontend source from GitHub. When running from a repo checkout,
it uses the local ``dashboards/frontend`` directory instead.
"""

from __future__ import annotations

import asyncio
import os
import secrets as _secrets
import time
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable, TypedDict

import modal

if TYPE_CHECKING:
    from google.protobuf.timestamp_pb2 import Timestamp
    from modal.client import _Client
    from modal_proto import api_pb2

# Imported at module scope so FastAPI can resolve the ``request: Request``
# annotation in stream_run_logs(). Under ``from __future__ import
# annotations`` all type hints are strings, and FastAPI evaluates them
# against the *defining function's* ``__globals__`` (i.e. this module).
# Importing ``Request`` only inside ``fastapi_app()`` makes the name
# invisible to FastAPI's introspection, which then mistakes the parameter
# for a query string and 422s with ``{"loc": ["query", "request"]}``.
from starlette.requests import Request

# Used as endpoint parameter annotations, so — like ``Request`` above — these
# must resolve from this module's globals.
from modal_training_gym.common.advantage_distribution import AdvantageDistribution
from modal_training_gym.common.config import (
    DASHBOARD_PROXY_AUTH_PATH,
    dashboard_requires_proxy_auth,
)
from modal_training_gym.common.dashboard import DASHBOARD_APP_NAME
from modal_training_gym.common.run import FrameworkStatusUpdate, TrainingRun
from modal_training_gym.common.run_list import (
    filter_run_summaries,
    run_list_field_metadata,
)
from modal_training_gym.common.run_summary import (
    JsonDict,
    RunSummary,
    build_run_summary,
    build_run_summaries,
)
from modal_training_gym.common.step_timing import (
    RoleTimingRecord,
    is_safe_run_id,
    legacy_run_to_records,
    load_run_async,
    rollout_lanes,
)
from modal_training_gym.common.time import parse_time as _parse_log_time
from modal_training_gym.common.training_rollout import (
    TrainingRolloutResult,
    TrainingRolloutSummary,
    _apply_parsed,
)

SummaryLoader = Callable[[], Awaitable[list[JsonDict]]]


# A single historical log line from ``AppFetchLogs``
class LogEntry(TypedDict):
    task_id: str
    line: str
    fd: int
    ts: float | None
    ts_ns: int | None


REPO_URL = "https://github.com/modal-projects/training-gym.git"
REPO_BRANCH = "main"

DASHBOARD_REQUIRES_PROXY_AUTH_ENV_KEY = "DASHBOARD_REQUIRES_PROXY_AUTH"

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

    return (
        base.run_commands("cd /app/frontend && npm install && npm run build")
        .add_local_python_source("modal_training_gym", copy=True)
        .env(
            {
                DASHBOARD_REQUIRES_PROXY_AUTH_ENV_KEY: "true"
                if dashboard_requires_proxy_auth()
                else "false"
            }
        )
    )


image = _build_image()

app = modal.App(DASHBOARD_APP_NAME, image=image)

STATIC_DIR = "/app/frontend/dist"


# Underscore-prefixed so it shows up as an auto-managed secret in the Modal
# Secrets UI and is auto-created on first deploy from ~/.modal.toml.
MODAL_CREDS_SECRET_NAME = "_training-gym-modal-creds"

# Holds DASHBOARD_PASSWORD. An empty value means the dashboard is open (no
# auth) — that's the default so existing deployments keep working untouched.
# Set a real value via ``training-gym set-password``.
DASHBOARD_PASSWORD_SECRET_NAME = "_training-gym-dashboard-password"

# Routes that must bypass Basic Auth. These endpoints authenticate with their
# own per-run bearer token; the proxy-auth status route must report only the
# Modal-layer setting, independent of dashboard password protection.
PASSWORD_EXEMPT_PATHS = frozenset(
    {
        DASHBOARD_PROXY_AUTH_PATH,
        "/api/framework-status",
        "/api/training-rollouts",
        "/api/advantage-distributions",
        "/api/timing-events",
    }
)

# Only ever the *expected* side of a comparison, so publishing it is safe.
_MISSING_TOKEN_DUMMY = "training-gym-missing-token-dummy-never-issued"


def _is_local() -> bool:
    """True when we're not running inside a Modal container."""
    return not os.environ.get("MODAL_IS_REMOTE")


def _resolve_log_window(
    since_ts: float | None,
    until_ts: float | None,
    *,
    default_since: float,
    default_until: float,
    now: float,
) -> tuple[float, float]:
    """Resolve the ``[since, until]`` log window to concrete epoch seconds.

    ``since_ts`` / ``until_ts`` are the already-parsed request bounds (``None``
    when the caller didn't supply one). Unset bounds fall back to
    ``default_since`` / ``default_until`` (typically the run's lifetime), and
    ``until`` defaults to ``now`` whenever it resolves to a non-positive value
    (e.g. the run hasn't ended). ``since`` is floored at 0 so a negative
    relative age can't reach before the epoch.
    """
    if since_ts is None:
        since_ts = float(default_since)
    if until_ts is None:
        until_ts = float(default_until)
    if until_ts <= 0:
        until_ts = now
    return max(0.0, since_ts), until_ts


def _to_timestamp(secs: float) -> Timestamp:
    """Convert epoch seconds (with fractional part) to a protobuf ``Timestamp``."""
    from google.protobuf.timestamp_pb2 import Timestamp

    ts = Timestamp()
    ts.seconds = int(secs)
    ts.nanos = int((secs - int(secs)) * 1e9)
    return ts


def _parse_log_batches(batches: Iterable[api_pb2.TaskLogsBatch]) -> list[LogEntry]:
    """Flatten Modal ``AppFetchLogs`` batches into ``LogEntry`` dicts."""
    logs: list[LogEntry] = []
    for batch in batches:
        for item in batch.items:
            if not item.data:
                continue
            ts = float(getattr(item, "timestamp", 0) or 0)
            ts_ns = int(getattr(item, "timestamp_ns", 0) or 0)
            entry: LogEntry = {
                "task_id": batch.task_id,
                "line": item.data,
                "fd": int(getattr(item, "file_descriptor", 0) or 0),
                "ts": ts or None,
                "ts_ns": ts_ns or None,
            }
            logs.append(entry)
    return logs


def _compute_next_page(logs: list[LogEntry], limit: int) -> tuple[bool, float | None]:
    """Derive ``(has_more, next_until)`` for keyset pagination."""
    has_more = len(logs) >= limit
    next_until: float | None = None
    if has_more and logs:
        oldest = logs[0]
        oldest_ns = oldest.get("ts_ns") or int(
            (oldest.get("ts") or 0.0) * 1_000_000_000
        )
        next_until = (oldest_ns - 1) / 1_000_000_000
    return has_more, next_until


def ensure_creds_secret(interactive: bool = False) -> bool:
    """Make sure the ``_training-gym-modal-creds`` Modal Secret exists.

    Idempotent: returns True if the secret was already present or if we
    successfully created it from ``~/.modal.toml``. Returns False if we
    can't find creds and ``interactive`` is False (or the user skipped).

    Called both from ``training-gym setup`` and at module-load of this file
    so that ``modal deploy dashboards/app.py`` works without any prior
    setup step — as long as the user has a valid ``~/.modal.toml``.
    """
    if not _is_local():
        # Inside a Modal container we have no credentials and no need to
        # create anything; the secret was already provisioned at deploy.
        return True

    from modal_training_gym.common.config import resolve_modal_creds

    token_id, token_secret, source = resolve_modal_creds()

    if not token_id or not token_secret:
        if not interactive:
            return False
        from getpass import getpass

        print(
            "\nThe dashboard needs Modal workspace credentials to stream "
            "training-run logs into the UI.\n"
            "Couldn't find creds in MODAL_TOKEN_* env vars or "
            "~/.modal.toml — provide them now (or Ctrl-C to skip).\n"
            "Find your tokens at https://modal.com/settings/tokens.\n"
        )
        try:
            token_id = input("MODAL_TOKEN_ID: ").strip()
            token_secret = getpass("MODAL_TOKEN_SECRET (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping Modal Secret setup.")
            return False
        source = "user input"

    if not token_id or not token_secret:
        return False

    try:
        env_values = {"MODAL_TOKEN_ID": token_id, "MODAL_TOKEN_SECRET": token_secret}
        modal.Secret.objects.create(
            MODAL_CREDS_SECRET_NAME, env_values, allow_existing=True
        )
        modal.Secret.from_name(MODAL_CREDS_SECRET_NAME).update(env_values)
        print(f"Provisioned Modal Secret {MODAL_CREDS_SECRET_NAME!r} (from {source}).")
        return True
    except Exception as exc:
        print(
            f"WARNING: failed to create Modal Secret {MODAL_CREDS_SECRET_NAME!r}: {exc}"
        )
        return False


def _password_secret_exists() -> bool:
    """True if the operator has configured a dashboard password Secret.

    Checked at deploy time (local) to decide whether to mount the Secret on
    the ASGI function — if it was never created, the dashboard stays open.
    """
    try:
        modal.Secret.from_name(DASHBOARD_PASSWORD_SECRET_NAME).hydrate()
        return True
    except Exception:
        return False


def _function_secrets() -> list[modal.Secret]:
    """Secrets mounted on the ASGI function.

    The password Secret is optional and mounted only when it exists; absent
    it, ``DASHBOARD_PASSWORD`` is never injected and the dashboard is open.
    """
    secrets = [modal.Secret.from_name(MODAL_CREDS_SECRET_NAME)]
    if _password_secret_exists():
        secrets.append(modal.Secret.from_name(DASHBOARD_PASSWORD_SECRET_NAME))
    return secrets


def set_dashboard_password(password: str) -> None:
    """Set (or clear) the dashboard password.

    Creates the ``_training-gym-dashboard-password`` Secret on demand. An empty
    string disables auth. The new value only takes effect once the dashboard app
    is redeployed, since containers read it from the environment at startup.
    """
    env_values = {"DASHBOARD_PASSWORD": password}
    modal.Secret.objects.create(
        DASHBOARD_PASSWORD_SECRET_NAME, env_values, allow_existing=True
    )
    modal.Secret.from_name(DASHBOARD_PASSWORD_SECRET_NAME).update(env_values)


# Auto-create the creds secret at module-load so `modal deploy
# dashboards/app.py` works out of the box. Gated to local context so it never
# fires inside the deployed container. The password secret is intentionally
# NOT auto-created — no secret means no auth.
if _is_local():
    ensure_creds_secret(interactive=False)


def _run_compact_sync() -> None:
    """Rebuild all summary stores from canonical per-item metadata."""
    from modal_training_gym.utils.metadata import (
        MetadataStore,
        compact_summary_store,
    )

    for summary_store in (
        MetadataStore.TRAINING_RUNS_SUMMARY,
        MetadataStore.TRAIN_RESULTS_SUMMARY,
        MetadataStore.DEPLOYMENTS_SUMMARY,
    ):
        compact_summary_store(summary_store)


@app.function(schedule=modal.Cron("*/30 * * * *"), retries=3, timeout=1800)
def compact_summaries() -> None:
    """Scheduled compaction of summary stores (every 30 min)."""
    _run_compact_sync()
    print("Compaction complete.")


@app.function(
    schedule=modal.Cron("*/30 * * * *"),
    secrets=_function_secrets(),
    retries=3,
    timeout=1800,
)
def reconcile() -> None:
    """Reconcile orphaned training runs and deployments every 30 minutes."""
    from modal_training_gym.common.reconcile import reconcile as _reconcile

    outcome = _reconcile()
    if outcome.runs:
        print(f"Reconciled {len(outcome.runs)} orphaned run(s):")
        for result in outcome.runs:
            print(f"  {result.training_run_id}: {result.reason}")
    else:
        print("No orphaned runs to reconcile.")

    if outcome.deployments:
        print(f"Reconciled {len(outcome.deployments)} orphaned deployment(s):")
        for result in outcome.deployments:
            print(f"  {result.deployment_id}: {result.reason}")
    else:
        print("No orphaned deployments to reconcile.")


@app.function(
    min_containers=1,
    secrets=_function_secrets(),
)
@modal.asgi_app(requires_proxy_auth=dashboard_requires_proxy_auth())
def fastapi_app():
    import base64
    import binascii

    from fastapi import (
        FastAPI,
        Header,
        HTTPException,
        Path as FastAPIPath,
    )  # Request imported at module scope
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import (
        FileResponse,
        JSONResponse,
        Response,
        StreamingResponse,
    )
    from fastapi.staticfiles import StaticFiles

    from modal_training_gym.common.modal_urls import modal_app_dashboard_url
    from modal_training_gym.utils.metadata import (
        MetadataStore,
        summary_items_from_payload,
        vol_get,
        vol_get_summary_items_healed,
        vol_put_summary_items,
    )

    web = FastAPI()

    # ── Optional password protection ──────────────────────────────────────
    # When DASHBOARD_PASSWORD is set we gate the whole app behind HTTP Basic
    # Auth (the username is ignored). An empty value means open access.
    dashboard_password = os.environ.get("DASHBOARD_PASSWORD", "")

    def _password_ok(authorization: str | None) -> bool:
        scheme, _, encoded = (authorization or "").partition(" ")
        if scheme.lower() != "basic" or not encoded:
            return False
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            return False
        _user, _, supplied = decoded.partition(":")
        return _secrets.compare_digest(supplied, dashboard_password)

    @web.middleware("http")
    async def require_password(request: Request, call_next):
        if dashboard_password and request.url.path not in PASSWORD_EXEMPT_PATHS:
            if not _password_ok(request.headers.get("Authorization")):
                return Response(
                    status_code=401,
                    headers={"WWW-Authenticate": 'Basic realm="training-gym"'},
                )
        return await call_next(request)

    @web.get(DASHBOARD_PROXY_AUTH_PATH)
    async def proxy_auth_status() -> bool:
        return os.environ.get(DASHBOARD_REQUIRES_PROXY_AUTH_ENV_KEY, "false") == "true"

    cache_ttl_seconds = 30.0
    cache_keys = ("runs", "train_results", "evals", "deployments")
    # Each entry holds (expires_at, values, loaded_at). ``loaded_at == 0.0``
    # means "never successfully loaded", which lets the very first request block
    # for real data instead of flashing an empty list.
    cache_entries: dict[str, tuple[float, list[JsonDict], float]] = {
        key: (0.0, [], 0.0) for key in cache_keys
    }

    TIMING_CACHE_MAX_RUNS = 64
    TIMING_CACHE_TTL_S = 15.0
    TIMING_CACHE_FINAL_TTL_S = 60.0

    class TimingEntry:
        def __init__(self) -> None:
            self.lanes: JsonDict = {}
            self.read_at: float | None = None
            self.final = False
            self.dirty = False
            self.lock = asyncio.Lock()

        @property
        def fresh(self) -> bool:
            if self.read_at is None:
                return False
            ttl = TIMING_CACHE_FINAL_TTL_S if self.final else TIMING_CACHE_TTL_S
            return time.monotonic() - self.read_at < ttl

    timing_cache: dict[str, TimingEntry] = {}
    cache_locks = {key: asyncio.Lock() for key in cache_keys}
    # Hold strong refs to background refresh tasks so they aren't GC'd mid-flight.
    refresh_tasks: set[asyncio.Task[list[JsonDict]]] = set()
    web.mount("/assets", StaticFiles(directory=f"{STATIC_DIR}/assets"), name="assets")

    # ── Shared Modal client ───────────────────────────────────────────────
    # Opens a client at startup and reuses it across all requests.
    modal_client: _Client | None = None
    modal_client_lock = asyncio.Lock()

    async def get_modal_client() -> _Client | None:
        """Return the shared, connection-open Modal client.

        Creates it once (guarded by a lock) and reuses it thereafter. Returns
        ``None`` when no credentials are configured.
        """
        nonlocal modal_client
        if modal_client is not None:
            return modal_client

        token_id = os.environ.get("MODAL_TOKEN_ID", "")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
        if not token_id or not token_secret:
            return None

        from modal.client import _Client

        async with modal_client_lock:
            if modal_client is None:
                modal_client = await _Client.from_credentials(token_id, token_secret)
            return modal_client

    @web.on_event("startup")
    async def _open_modal_client() -> None:
        # Warm the shared client at startup. Failures here are non-fatal:
        # endpoints fall back to lazy init and surface error if creds are missing.
        try:
            await get_modal_client()
        except Exception:
            pass

    async def refresh_cache(key: str, loader: SummaryLoader) -> list[JsonDict]:
        async with cache_locks[key]:
            now = time.monotonic()
            expires_at, values, loaded_at = cache_entries[key]
            if now < expires_at:
                return values
            try:
                values = await loader()
                cache_entries[key] = (now + cache_ttl_seconds, values, now)
            except Exception:
                # Keep serving the last known data and back off so a slow/failing
                # loader (e.g. a heavy summary rebuild) can't be retried on every
                # request — it must never block or break the endpoint.
                cache_entries[key] = (now + cache_ttl_seconds, values, loaded_at)
            return values

    def invalidate_cache(key: str) -> None:
        # Force the next read to revalidate, but keep the last values and the
        # "loaded once" marker so it refreshes in the background instead of
        # blocking on a cold rebuild.
        _expires_at, values, loaded_at = cache_entries[key]
        cache_entries[key] = (0.0, values, loaded_at)

    async def get_cached_list(key: str, loader: SummaryLoader) -> list[JsonDict]:
        now = time.monotonic()
        expires_at, values, loaded_at = cache_entries[key]
        if now < expires_at:
            return values

        # Nothing cached yet: block once on the loader so the first paint has
        # real data rather than an empty flash.
        if loaded_at == 0.0:
            return await refresh_cache(key, loader)

        # Stale-while-revalidate: return the last good data immediately and
        # refresh in the background. The expensive runs rebuild then happens off
        # the request path, so the UI is never left waiting on it.
        if not cache_locks[key].locked():
            task = asyncio.create_task(refresh_cache(key, loader))
            refresh_tasks.add(task)
            task.add_done_callback(refresh_tasks.discard)
        return values

    def add_modal_app_urls(
        items: list[JsonDict],
    ) -> tuple[list[JsonDict], bool]:
        updated: list[JsonDict] = []
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

    def merge_missing_fields(
        merged: JsonDict, source: JsonDict, fields: tuple[str, ...]
    ) -> None:
        for field in fields:
            value = source.get(field)
            if field not in merged and value is not None:
                merged[field] = value

    async def fetch_by_id(
        store: MetadataStore, key: str
    ) -> tuple[str, JsonDict | None]:
        try:
            return key, await run_in_threadpool(vol_get, store, key)
        except KeyError:
            return key, None

    async def fetch_all_by_id(
        store: MetadataStore, keys: list[str]
    ) -> dict[str, JsonDict]:
        fetched = await asyncio.gather(*(fetch_by_id(store, key) for key in keys))
        return {key: value for key, value in fetched if value is not None}

    async def load_eval_summaries() -> list[JsonDict]:
        try:
            payload = await run_in_threadpool(vol_get, MetadataStore.EVALS, "summary")
        except KeyError:
            return []

        summaries = summary_items_from_payload(payload, payload_key="summaries")
        if not summaries:
            return []

        results_by_id = await fetch_all_by_id(
            MetadataStore.EVAL_RESULTS,
            [str(s.get("eval_id") or "") for s in summaries if s.get("eval_id")],
        )
        configs_by_id = await fetch_all_by_id(
            MetadataStore.EVAL_CONFIGS,
            [
                str(s.get("eval_config_id") or "")
                for s in summaries
                if s.get("eval_config_id")
            ],
        )

        enriched: list[JsonDict] = []
        for summary in summaries:
            merged = dict(summary)
            result = results_by_id.get(str(summary.get("eval_id") or ""))
            if result:
                merge_missing_fields(
                    merged, result, ("deployment_id", "config", "status")
                )
            eval_config = configs_by_id.get(str(summary.get("eval_config_id") or ""))
            if eval_config:
                merged["eval_config"] = eval_config
                merge_missing_fields(
                    merged,
                    eval_config,
                    (
                        "dataset_name",
                        "eval_fn_name",
                        "prompt_column",
                        "generate_kwargs",
                    ),
                )
            enriched.append(merged)
        return enriched

    async def load_list_summary(
        summary_store: MetadataStore,
    ) -> list[JsonDict]:
        items = await run_in_threadpool(vol_get_summary_items_healed, summary_store)
        if not items:
            return []
        items, changed = add_modal_app_urls(items)
        if changed:
            await run_in_threadpool(vol_put_summary_items, summary_store, items)
        return items

    async def load_runs() -> list[JsonDict]:
        run_records = await load_list_summary(MetadataStore.TRAINING_RUNS_SUMMARY)
        try:
            result_records = await load_list_summary(
                MetadataStore.TRAIN_RESULTS_SUMMARY
            )
        except Exception:
            # A result-store outage must not hide otherwise healthy run records.
            result_records = []
        return [
            summary.model_dump(mode="json")
            for summary in build_run_summaries(run_records, result_records)
        ]

    async def load_train_results() -> list[JsonDict]:
        return await load_list_summary(MetadataStore.TRAIN_RESULTS_SUMMARY)

    async def load_deployments() -> list[JsonDict]:
        return await load_list_summary(MetadataStore.DEPLOYMENTS_SUMMARY)

    def _bearer_token(authorization: str | None) -> str:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer":
            return ""
        return token.strip()

    async def _require_framework_status_token(
        training_run_id: str,
        authorization: str | None,
    ) -> None:
        """403 unless ``authorization`` carries the run's status token.

        Handlers must call this *before* any lookup that can 404, or the
        status code tells an anonymous caller which run ids exist. For the
        same reason an unknown run is indistinguishable from a wrong token.
        """
        try:
            expected_token = str(
                (
                    await run_in_threadpool(
                        vol_get,
                        MetadataStore.FRAMEWORK_STATUS_TOKENS,
                        training_run_id,
                    )
                ).get("token", "")
            )
        except KeyError:
            expected_token = ""
        supplied = _bearer_token(authorization)
        if not expected_token:
            _secrets.compare_digest(supplied, _MISSING_TOKEN_DUMMY)
            raise HTTPException(status_code=403, detail="Invalid status token")
        if not _secrets.compare_digest(supplied, expected_token):
            raise HTTPException(status_code=403, detail="Invalid status token")

    async def _get_run_or_404(training_run_id: str) -> TrainingRun:
        try:
            return await TrainingRun.from_id(training_run_id, is_async=True)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"TrainingRun {training_run_id!r} not found",
            )

    # ── Training runs ────────────────────────────────────────────────────

    @web.get("/api/runs", response_model=list[RunSummary])
    async def runs(
        request: Request,
        since: int | None = None,
        limit: int | None = None,
    ):
        if limit is not None and limit < 1:
            raise HTTPException(status_code=400, detail="Limit must be positive")
        try:
            data = await get_cached_list("runs", load_runs)
        except Exception:
            data = []
        summaries = [
            RunSummary.model_validate(item) for item in data if isinstance(item, dict)
        ]
        filters = {
            name: request.query_params.get(name, "")
            for name, metadata in run_list_field_metadata().items()
            if metadata.get("filterable")
        }
        filtered = filter_run_summaries(
            summaries,
            filters=filters,
            since=since,
            limit=limit,
        )
        return filtered

    @web.get("/api/runs/{training_run_id}", response_model=RunSummary)
    async def get_run(training_run_id: str):
        try:
            run = await TrainingRun.from_id(training_run_id, is_async=True)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Training run {training_run_id!r} not found",
            )
        try:
            result = await vol_get(
                MetadataStore.TRAIN_RESULTS,
                training_run_id,
                is_async=True,
            )
        except KeyError:
            result = None
        return build_run_summary(run.model_dump(mode="json"), result)

    @web.post("/api/framework-status")
    async def framework_status(
        update: FrameworkStatusUpdate,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(update.training_run_id, authorization)
        run = await _get_run_or_404(update.training_run_id)

        status = await run_in_threadpool(run.apply_framework_status, update)
        if status is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported framework status {update.phase!r} "
                    f"for {run.framework.value}"
                ),
            )
        await run.save(is_async=True)
        invalidate_cache("runs")
        return JSONResponse({"status": "ok", "framework_status": status.value})

    # ── Training rollouts ────────────────────────────────────────────────

    @web.post("/api/training-rollouts")
    async def training_rollout(
        result: TrainingRolloutResult,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(result.training_run_id, authorization)
        run = await _get_run_or_404(result.training_run_id)

        await run_in_threadpool(result.save)
        run.record_latest_rollout(result)
        await run.save(is_async=True)
        invalidate_cache("runs")

        return JSONResponse(
            {"status": "ok", "rollout_id": result.rollout_id, "mean": result.mean}
        )

    @web.get(
        "/api/runs/{training_run_id}/rollouts",
        response_model=list[TrainingRolloutSummary],
    )
    async def list_run_rollouts(training_run_id: str):
        return await run_in_threadpool(
            TrainingRolloutResult.list_summaries_for_run, training_run_id
        )

    @web.get("/api/runs/{training_run_id}/rollouts/{rollout_id}")
    async def get_run_rollout(training_run_id: str, rollout_id: int):
        key = f"{training_run_id}__{int(rollout_id):08d}"
        try:
            data = await run_in_threadpool(
                vol_get, MetadataStore.TRAINING_ROLLOUTS, key
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Rollout {rollout_id} for run {training_run_id!r} not found",
            )

        if isinstance(data, dict):
            _apply_parsed(data.get("samples"))
        return JSONResponse(data)

    # ── Per-group advantage distributions ────────────────────────────────

    @web.post("/api/advantage-distributions")
    async def advantage_distribution(
        shard: AdvantageDistribution,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(shard.training_run_id, authorization)
        await _get_run_or_404(shard.training_run_id)

        await shard.save(is_async=True)
        return JSONResponse(
            {
                "status": "ok",
                "rollout_id": shard.rollout_id,
                "dp_rank": shard.dp_rank,
                "samples": len(shard.samples),
            }
        )

    @web.get("/api/runs/{training_run_id}/advantages")
    async def list_run_advantages(training_run_id: str):
        steps = await run_in_threadpool(
            AdvantageDistribution.list_steps_for_run, training_run_id
        )
        return JSONResponse(steps)

    @web.get("/api/runs/{training_run_id}/advantages/{rollout_id}")
    async def get_run_advantages(training_run_id: str, rollout_id: int):
        merged = await run_in_threadpool(
            AdvantageDistribution.merged_for_step, training_run_id, rollout_id
        )
        if merged is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No advantage distribution for rollout {rollout_id} "
                    f"of run {training_run_id!r}"
                ),
            )
        return JSONResponse(merged)

    @web.post("/api/timing-events")
    async def timing_event(
        record: RoleTimingRecord,
        authorization: str | None = Header(default=None),
    ):
        await _require_framework_status_token(record.training_run_id, authorization)

        await record.save(is_async=True)
        entry = timing_cache.get(record.training_run_id)
        if entry is not None:
            entry.dirty = True
            entry.final = False
            if not entry.lock.locked():
                rollout_key = (
                    "pre-loop" if record.rollout_id is None else str(record.rollout_id)
                )
                async with entry.lock:
                    lanes = entry.lanes.setdefault(rollout_key, {"roles": {}})
                    lanes["roles"].update(
                        rollout_lanes([record.model_dump(mode="json")])["roles"]
                    )
        return JSONResponse({"status": "ok"})

    async def _run_has_ended(training_run_id: str) -> bool:
        ended_statuses = frozenset({"completed", "failed", "stopped", "cancelled"})
        _expires_at, runs_cached, loaded_at = cache_entries["runs"]
        if loaded_at != 0.0:
            for run in runs_cached:
                if isinstance(run, dict) and run.get("training_run_id") == (
                    training_run_id
                ):
                    return run.get("status") in ended_statuses
        try:
            run_record = await TrainingRun.from_id(training_run_id, is_async=True)
        except KeyError:
            return False
        return run_record.status.value in ended_statuses

    async def _read_run_timings(
        training_run_id: str,
    ) -> tuple[JsonDict, bool]:
        found, had_read_failures = await load_run_async(training_run_id)
        legacy_derived = False
        if not found and not had_read_failures:
            try:
                run = await _get_run_or_404(training_run_id)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
            else:
                legacy_records = legacy_run_to_records(run.substep_times)
                legacy_derived = bool(legacy_records)
                for record in legacy_records:
                    found.setdefault(int(record["rollout_id"]), []).append(record)
        timings = {
            ("pre-loop" if rollout_id is None else str(rollout_id)): rollout_lanes(
                records
            )
            for rollout_id, records in sorted(
                found.items(), key=lambda item: (item[0] is None, item[0] or 0)
            )
        }
        if legacy_derived:
            timings["metadata"] = {"legacy_derived": True}
        return timings, had_read_failures

    async def _run_timings(training_run_id: str) -> JsonDict:
        entry = timing_cache.get(training_run_id)
        if entry is None:
            if len(timing_cache) >= TIMING_CACHE_MAX_RUNS:
                evictable = [
                    (other.read_at or 0.0, run_id)
                    for run_id, other in timing_cache.items()
                    if not other.lock.locked()
                ]
                if evictable:
                    del timing_cache[min(evictable)[1]]
            entry = timing_cache.setdefault(training_run_id, TimingEntry())
        if entry.fresh:
            return entry.lanes
        async with entry.lock:
            if not entry.fresh:
                entry.dirty = False
                entry.lanes, had_read_failures = await _read_run_timings(
                    training_run_id
                )
                entry.read_at = time.monotonic()
                entry.final = (
                    not had_read_failures
                    and not entry.dirty
                    and await _run_has_ended(training_run_id)
                )
            return entry.lanes

    @web.get("/api/runs/{training_run_id}/timings")
    async def get_run_timings(
        training_run_id: str = FastAPIPath(),
    ):
        if not is_safe_run_id(training_run_id):
            raise HTTPException(status_code=422, detail="unsafe training run id")
        return JSONResponse(await _run_timings(training_run_id))

    # ── Live Modal log stream (SSE, pure pass-through) ───────────────────

    @web.get("/api/runs/{training_run_id}/logs/stream")
    async def stream_run_logs(
        training_run_id: str,
        request: Request,
        search: str = "",
        max_lines_per_sec: int = 0,
    ):
        """Server-Sent Events stream of the underlying Modal app's logs.

        Pure pass-through: we open a long-poll ``AppGetLogs`` stream against
        the run's ``modal_app_id`` and forward each batch as an SSE ``data``
        event. Nothing is persisted on the dashboard side.

        Query params:
          - ``search``: case-insensitive substring filter; lines that don't
            match are silently dropped.
          - ``max_lines_per_sec``: integer rate cap. Lines exceeding the cap
            in any 1-second window are dropped; a single ``dropped`` event
            is emitted per second summarizing the count.
        """
        import json

        from modal_proto import api_pb2

        run = await _get_run_or_404(training_run_id)

        app_id = (run.modal_app_id or "").strip()
        if not app_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"TrainingRun {training_run_id!r} has no modal_app_id "
                    "yet — logs not available."
                ),
            )

        search_lower = search.strip().lower() if search else ""
        rate_cap = max(0, int(max_lines_per_sec or 0))

        async def event_stream():
            try:
                client = await get_modal_client()
            except Exception as exc:
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': f'auth failed: {exc!s}'})}\n\n"
                )
                return
            if client is None:
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': 'No Modal credentials configured. Run training-gym setup.'})}\n\n"
                )
                return

            last_entry_id = ""
            window_start = time.monotonic()
            window_emitted = 0
            window_dropped = 0
            consecutive_errors = 0
            max_consecutive_errors = 10

            def _drain_drop_event() -> str | None:
                nonlocal window_dropped
                if not window_dropped:
                    return None
                payload = {"dropped": window_dropped}
                window_dropped = 0
                return f"event: dropped\ndata: {json.dumps(payload)}\n\n"

            while True:
                if await request.is_disconnected():
                    return
                req = api_pb2.AppGetLogsRequest(
                    app_id=app_id,
                    timeout=55,
                    last_entry_id=last_entry_id,
                )
                try:
                    async for log_batch in client.stub.AppGetLogs.unary_stream(req):
                        if await request.is_disconnected():
                            return
                        consecutive_errors = 0
                        if log_batch.entry_id:
                            last_entry_id = log_batch.entry_id
                        for log in log_batch.items:
                            if not log.data:
                                continue
                            if search_lower and search_lower not in log.data.lower():
                                continue

                            now = time.monotonic()
                            if now - window_start >= 1.0:
                                drop_event = _drain_drop_event()
                                if drop_event:
                                    yield drop_event
                                window_start = now
                                window_emitted = 0

                            if rate_cap and window_emitted >= rate_cap:
                                window_dropped += 1
                                continue

                            window_emitted += 1
                            payload: dict[str, str | float] = {
                                "task_id": log_batch.task_id,
                                "line": log.data,
                            }
                            ts = getattr(log, "timestamp", 0) or 0
                            if ts:
                                payload["ts"] = ts
                            yield f"data: {json.dumps(payload)}\n\n"
                        if log_batch.app_done:
                            drop_event = _drain_drop_event()
                            if drop_event:
                                yield drop_event
                            yield "event: done\ndata: {}\n\n"
                            return
                except asyncio.CancelledError:
                    return
                except Exception as exc:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        yield (
                            "event: error\n"
                            f"data: {json.dumps({'error': f'log stream failed after {consecutive_errors} retries: {exc!s}'})}\n\n"
                        )
                        return
                    backoff = min(2 ** (consecutive_errors - 1), 10)
                    yield (
                        "event: reconnect\n"
                        f"data: {json.dumps({'reason': str(exc)})}\n\n"
                    )
                    try:
                        await asyncio.sleep(backoff)
                    except asyncio.CancelledError:
                        return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Historical Modal logs ──────────────

    @web.get("/api/runs/{training_run_id}/logs")
    async def get_run_logs(
        training_run_id: str,
        since: str = "",
        until: str = "",
        max_lines: int = 100,
        search: str = "",
    ):
        """Historical log fetch for a run, backed by Modal's ``AppFetchLogs``.

        Returns the most recent ``max_lines`` entries within the ``[since, until]``
        window, oldest-first.

        Query params:
          - ``since`` / ``until``: window bounds as epoch seconds, ISO 8601, or a
            relative age (``30m`` / ``2h`` / ``1d`` / ``45s`` = "N ago").
            ``since`` is exclusive and ``until`` is inclusive. Defaults to the run's
            lifetime.
          - ``max_lines``: max entries to return (default 100). Throws if negative or too large.
            These are the newest entries in the window (ClickHouse caps a single
            fetch at 20000).
          - ``search``: case-insensitive substring filter.

        Returns a JSON object with the following fields:
          - ``logs``: a list of log entries
          - ``has_more``: whether there are more log entries to fetch
          - ``next_until``: the timestamp of the next log entry to fetch
        """
        from modal_proto import api_pb2

        if max_lines < 0:
            raise HTTPException(status_code=400, detail="max_lines must be positive")

        run = await _get_run_or_404(training_run_id)

        app_id = (run.modal_app_id or "").strip()
        if not app_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"TrainingRun {training_run_id!r} has no modal_app_id "
                    "yet — logs not available."
                ),
            )

        now = time.time()
        since_ts = _parse_log_time(since, now)
        until_ts = _parse_log_time(until, now)
        for value, parsed, name in (
            (since, since_ts, "since"),
            (until, until_ts, "until"),
        ):
            if value and parsed is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{name} must be epoch seconds, ISO 8601, "
                        "or a relative time such as 24h"
                    ),
                )
        since_ts, until_ts = _resolve_log_window(
            since_ts,
            until_ts,
            default_since=run.started_at or run.created_at or 0,
            default_until=run.ended_at or run.completed_at or 0,
            now=now,
        )

        try:
            client = await get_modal_client()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Modal auth: {exc!s}")
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="No Modal credentials configured. Run training-gym setup.",
            )

        req = api_pb2.AppFetchLogsRequest(
            app_id=app_id,
            since=_to_timestamp(since_ts),
            until=_to_timestamp(until_ts),
            limit=max_lines,
            search_text=search.strip(),
        )
        try:
            resp = await client.stub.AppFetchLogs(req)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"AppFetchLogs: {exc!s}")

        logs = _parse_log_batches(resp.batches)
        has_more, next_until = _compute_next_page(logs, max_lines)

        return JSONResponse(
            {
                "logs": logs,
                "has_more": has_more,
                "next_until": next_until,
            }
        )

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
            data = await run_in_threadpool(vol_get, MetadataStore.EVAL_RESULTS, eval_id)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"EvalResult {eval_id!r} not found",
            )
        if isinstance(data, dict):
            _apply_parsed(data.get("rows"))
        return JSONResponse(data)

    # ── Deployments ──────────────────────────────────────────────────────

    @web.get("/api/deployments")
    async def deployments():
        try:
            data = await get_cached_list("deployments", load_deployments)
        except Exception:
            data = []
        return JSONResponse(data)

    @web.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        return FileResponse(f"{STATIC_DIR}/favicon.svg", media_type="image/svg+xml")

    @web.get("/apple-touch-icon.png", include_in_schema=False)
    async def apple_touch_icon():
        return FileResponse(
            f"{STATIC_DIR}/apple-touch-icon.png", media_type="image/png"
        )

    # ── SPA fallback ─────────────────────────────────────────────────────

    @web.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        return FileResponse(f"{STATIC_DIR}/index.html")

    return web
