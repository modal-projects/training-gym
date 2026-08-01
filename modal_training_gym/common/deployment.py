"""Deployment config and model deployment handle.

``DeploymentConfig`` composes a ``ModelConfig`` with a serving recipe
(``VllmRecipe`` or ``SglangRecipe``) and exposes ``serve()`` to deploy
the model on Modal.

``ModelDeployment`` is the handle returned by ``serve()`` — it carries
the live endpoint URL and convenience methods for generation and eval.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

from modal.experimental import list_deployed_apps
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from modal_training_gym.common.checkpoint import (
    Checkpoint,
    CheckpointType,
    convert_checkpoint_to_hf,
)
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.ids import create_hash
from modal_training_gym.common.modal_urls import modal_app_dashboard_url
from modal_training_gym.common.models import ModelConfig
from modal_training_gym.common.proxy_auth import (
    PROXY_AUTH_HOSTS_ENV,
    _configured_pair,
    modal_proxy_auth_headers,
    proxy_auth_url_allowed,
)
from modal_training_gym.deploy_recipes.base import DeployRecipeType
from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe
from modal_training_gym.deploy_recipes.vllm_recipe import VllmRecipe
from modal_training_gym.utils.metadata import (
    MetadataStore,
    vol_get,
    vol_put,
    vol_upsert_summary_item,
)

DEPLOYMENTS_STORE_NAME = MetadataStore.DEPLOYMENTS.value


def _run_coro(coro):
    """Run a coroutine to completion, even from inside a running event loop.

    ``asyncio.run`` raises when an event loop is already running (e.g. in a
    Jupyter notebook), so in that case we run the coroutine on a dedicated
    worker thread with its own loop. If ``coro`` is already a resolved value
    (some Modal versions return the URL synchronously), it is returned as-is.
    """
    if not inspect.isawaitable(coro):
        return coro
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict = {}

    def _worker():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            result["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


# Shared with the dashboard clients (status reporters, CLI); see that module
# for resolution rules.
_modal_proxy_auth_headers = modal_proxy_auth_headers


def _raise_for_proxy_auth(status_code: int, url: str) -> None:
    """Turn a 401 into an actionable proxy-auth hint.

    SGLang endpoints are public by default (``unauthenticated=True``). When
    an endpoint was served with ``unauthenticated=False``, a 401 almost
    always means the ``MODAL_KEY`` / ``MODAL_SECRET`` proxy-auth token pair
    never reached it — unset, or withheld from a non-Modal URL by the egress
    guard — rather than a real authorization problem; surface which, instead
    of a bare ``HTTPError``/``TimeoutError``. No-op for any other status.
    """
    if status_code != 401:
        return
    key, secret = _configured_pair()
    if key and secret and not proxy_auth_url_allowed(url):
        detail = (
            "the configured MODAL_KEY / MODAL_SECRET pair was withheld from "
            f"this non-Modal URL; set {PROXY_AUTH_HOSTS_ENV}=<host> to send it"
        )
    elif not (key and secret):
        detail = (
            "the MODAL_KEY / MODAL_SECRET proxy-auth tokens are not set in "
            "this environment. Create a proxy-auth token pair at "
            "https://modal.com/settings/proxy-auth-tokens and export MODAL_KEY "
            "(wk-…) and MODAL_SECRET (ws-…) in the shell that runs the "
            "eval/serve. For calls issued from remote workers (e.g. a custom "
            "rm/reward function), also forward the pair into the worker via a "
            "modal.Secret"
        )
    else:
        detail = (
            "the MODAL_KEY / MODAL_SECRET tokens were sent but rejected "
            "(expired or wrong workspace)"
        )
    raise RuntimeError(
        f"401 Unauthorized from {url} — this endpoint is behind Modal proxy auth "
        f"and {detail}. SGLang endpoints are public by default; pass "
        "DeploymentConfig(unauthenticated=False) to require proxy auth."
    )


def _messages_to_openai(messages: list[dict]) -> list[dict]:
    """Serialize internal tool-call arguments without mutating caller messages."""
    wire_messages = []
    for message in messages:
        wire_message = dict(message)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            wire_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    wire_tool_calls.append(tool_call)
                    continue
                wire_tool_call = dict(tool_call)
                function = tool_call.get("function")
                if isinstance(function, dict):
                    wire_function = dict(function)
                    arguments = function.get("arguments")
                    if isinstance(arguments, dict):
                        wire_function["arguments"] = json.dumps(arguments)
                    wire_tool_call["function"] = wire_function
                wire_tool_calls.append(wire_tool_call)
            wire_message["tool_calls"] = wire_tool_calls
        wire_messages.append(wire_message)
    return wire_messages


@dataclass
class DeploymentConfig:
    """Deploy a model behind a serving engine.

    When ``recipe`` is ``None``, defaults to ``SglangRecipe()``.
    """

    model: ModelConfig
    checkpoint: Checkpoint | None = None
    recipe: VllmRecipe | SglangRecipe | None = None

    app_name: str | None = None
    served_model_name: str | None = None
    unauthenticated: bool = True

    def _checkpoints_volume_name(self) -> str | None:
        if self.checkpoint is not None and self.checkpoint.checkpoints_volume_name:
            return self.checkpoint.checkpoints_volume_name
        return getattr(self.model, "checkpoints_volume_name", None)

    def _checkpoints_mount_path(self) -> str | None:
        if self.checkpoint is not None and self.checkpoint.checkpoints_mount_path:
            return self.checkpoint.checkpoints_mount_path
        return getattr(self.model, "checkpoints_mount_path", None)

    def _resolved_model_path(self) -> str:
        if self.checkpoint is not None:
            checkpoint_path = getattr(self.checkpoint, "path", "")
            if checkpoint_path:
                return checkpoint_path

        model_path = self.model.model_path or self.model.model_name
        if not model_path:
            raise TrainingGymConfigError(
                f"{type(self.model).__name__} has no model path to serve. "
                "Set model_path or model_name."
            )
        return model_path

    def _new_deployment_id(
        self,
        *,
        recipe: VllmRecipe | SglangRecipe,
        model_path: str,
    ) -> str:
        return create_hash(
            self.model.model_name,
            self.checkpoint.path if self.checkpoint is not None else "",
            f"{type(recipe).__name__}:{recipe.recipe_type.value}",
            self.app_name or "",
            model_path,
        )

    # TODO: add _merge_recipe for deployment configs
    def serve(self) -> "ModelDeployment":
        """Build, deploy, and return a ``ModelDeployment`` handle."""

        recipe = self.recipe or SglangRecipe()
        model = self.model

        if (
            self.checkpoint is not None
            and self.checkpoint.checkpoint_type == CheckpointType.megatron
        ):
            self.checkpoint = convert_checkpoint_to_hf(
                checkpoint=self.checkpoint,
                model=model,
                recipe=recipe,
            )

        model_path = self._resolved_model_path()

        default_name_src = model.model_name or model_path
        default_slug = (
            default_name_src.rstrip("/").split("/")[-1].replace("_", "-").lower()
        )
        if not self.app_name:
            self.app_name = f"{default_slug}-serve"
        if not self.served_model_name:
            self.served_model_name = default_slug

        deployment_id = self._new_deployment_id(recipe=recipe, model_path=model_path)
        checkpoints_volume = self._checkpoints_volume_name()
        checkpoints_mount_path = self._checkpoints_mount_path()

        if isinstance(recipe, SglangRecipe):
            from modal_training_gym.deploy_recipes.sglang_recipe.serve_sglang import (
                build_sglang_serve_app,
            )

            app = build_sglang_serve_app(
                recipe=recipe,
                app_name=self.app_name,
                model_path=model_path,
                served_model_name=self.served_model_name,
                checkpoints_volume=checkpoints_volume,
                checkpoints_mount_path=checkpoints_mount_path,
                deployment_id=deployment_id,
                unauthenticated=self.unauthenticated,
            )
        elif isinstance(recipe, VllmRecipe):
            # unauthenticated is SGLang-only; vLLM's http_server has no
            # proxy-auth. Default True is a silent no-op; warn only
            # when the caller explicitly requests authenticated access.
            if self.unauthenticated is False:
                warnings.warn(
                    "DeploymentConfig(unauthenticated=False) is not supported for "
                    "VllmRecipe: modal.experimental.http_server has no proxy-auth, so the endpoint remains publicly reachable. Use "
                    "SglangRecipe if you need Modal proxy auth.",
                    stacklevel=2,
                )
            from modal_training_gym.deploy_recipes.vllm_recipe.serve_vllm import (
                build_vllm_serve_app,
            )

            app = build_vllm_serve_app(
                recipe=recipe,
                app_name=self.app_name,
                model_path=model_path,
                served_model_name=self.served_model_name,
                checkpoints_volume=checkpoints_volume,
                checkpoints_mount_path=checkpoints_mount_path,
                deployment_id=deployment_id,
            )
        else:
            raise TrainingGymConfigError(
                f"Unsupported deploy recipe: {type(recipe).__name__}"
            )

        env_name = recipe.environment_name
        strategy = recipe.deploy_strategy
        app.deploy(
            environment_name=env_name,
            strategy=strategy,
        )

        if recipe.recipe_type == DeployRecipeType.SGLANG:
            server_attr = "SGLangEndpoint"
        else:
            server_attr = "Server"
        server = getattr(app, server_attr, None)
        if server is None and hasattr(app, "registered_functions"):
            server = app.registered_functions.get(server_attr)
        if server is None:
            raise RuntimeError(
                f"Deployed {self.app_name!r} but could not resolve "
                f"{server_attr} server handle."
            )
        url = _run_coro(server.get_url())
        modal_app_id = app.app_id
        modal_app_url = modal_app_dashboard_url(modal_app_id)
        if not url:
            raise RuntimeError(
                f"Deployed {self.app_name!r} but no web URL was returned."
            )
        if not modal_app_id:
            raise RuntimeError(
                f"Deployed {self.app_name!r} but no Modal app id was returned."
            )
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            deployment_config=self,
            modal_app_id=modal_app_id,
            modal_app_url=modal_app_url or modal_app_dashboard_url(modal_app_id),
            url=url,
            status=DeploymentStatus.RUNNING.value,
        )
        deployment.save()
        return deployment


class DeploymentStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    READY = "ready"
    INITIALIZING = "initializing"
    INACTIVE = "inactive"


def update_deployment_status(
    deployment_id: str,
    status: str,
    *,
    seed: dict[str, Any] | None = None,
) -> bool:
    """Update a deployment's status in both individual record and summary.

    Returns ``True`` when the write succeeds. If the canonical record is
    missing, ``seed`` (e.g. a summary-only row) is used to create it.
    """
    try:
        payload = vol_get(MetadataStore.DEPLOYMENTS, deployment_id)
    except KeyError:
        if seed is None:
            return False
        payload = dict(seed)
        payload["deployment_id"] = deployment_id
    payload["status"] = status
    vol_put(MetadataStore.DEPLOYMENTS, deployment_id, payload)
    vol_upsert_summary_item(
        MetadataStore.DEPLOYMENTS_SUMMARY,
        payload,
        item_id_key="deployment_id",
        sort_key=lambda item: (
            str(item.get("deployment_config", {}).get("app_name", "")),
            str(item.get("deployment_id", "")),
        ),
        reverse=True,
    )
    return True


class _CrashloopDetector:
    """Crashloop heuristic over periodic container-count samples.

    Zero containers is tolerated for ``cold_start_grace`` before any container
    has been seen running (image pull, weight download, and server launch all
    happen in that window), but only for ``restart_grace`` once one has come
    up — a container that appeared and then vanished is a real restart cycle.
    """

    def __init__(self, *, cold_start_grace: int, restart_grace: int) -> None:
        self._cold_start_grace = cold_start_grace
        self._restart_grace = restart_grace
        self._zero_since: float | None = None
        self._seen_running = False

    def observe(self, containers: int, now: float) -> tuple[int, str] | None:
        """Record a container-count sample; returns ``(elapsed, phase)`` on crashloop.

        Negative counts are transient failures querying Modal app state and are
        ignored rather than counted as zero containers.
        """
        if containers < 0:
            return None
        if containers > 0:
            self._seen_running = True
            self._zero_since = None
            return None
        if self._zero_since is None:
            self._zero_since = now
        elapsed = int(now - self._zero_since)
        grace = self._restart_grace if self._seen_running else self._cold_start_grace
        if elapsed < grace:
            return None
        phase = (
            "came up and then died"
            if self._seen_running
            else f"never reached a running state within {grace}s"
        )
        return elapsed, phase


class ModelDeployment(BaseModel):
    """A deployed model endpoint.

    Returned by ``DeploymentConfig.serve()``. Carries the live endpoint
    URL, Modal app id, and convenience methods for generation and evaluation.
    """

    deployment_id: str

    model_config = ConfigDict(arbitrary_types_allowed=True)
    deployment_config: DeploymentConfig

    modal_app_id: str = ""
    modal_app_url: str = ""
    url: str
    status: str = DeploymentStatus.RUNNING.value

    @field_validator("deployment_config", mode="before")
    @classmethod
    def _parse_deployment_config(cls, value: object) -> DeploymentConfig | object:
        if not isinstance(value, dict):
            return value

        deployment_config = dict(value)
        model = deployment_config.get("model")
        if isinstance(model, dict):
            deployment_config["model"] = ModelConfig(**model)
        deployment_config.setdefault("unauthenticated", True)
        return DeploymentConfig(**deployment_config)

    @field_serializer("deployment_config")
    def _serialize_deployment_config(
        self, value: DeploymentConfig
    ) -> dict[str, object]:
        model = value.model
        return {
            "model": {
                "model_name": getattr(model, "model_name", ""),
                "model_path": getattr(model, "model_path", None),
                "checkpoints_volume_name": getattr(
                    model,
                    "checkpoints_volume_name",
                    None,
                ),
                "checkpoints_mount_path": getattr(
                    model,
                    "checkpoints_mount_path",
                    None,
                ),
            },
            "app_name": value.app_name,
            "served_model_name": value.served_model_name,
            "unauthenticated": value.unauthenticated,
        }

    # TODO(atoniolo76): A future PR should update all existing tutorials to
    # use this new function while getting rid of the old generate.
    def chat(
        self,
        messages: list[dict],
        ensure_ready: bool = True,
        max_attempts: int = 4,
        timeout: int = 120,
        **kwargs,
    ) -> dict:
        """Return one OpenAI-compatible chat-completion message while
        preserving structured fields like tool_calls and reasoning_content.
        """
        import time

        import requests

        if ensure_ready:
            self.wait_until_ready()
        body = {
            "model": self.deployment_config.served_model_name,
            "messages": _messages_to_openai(messages),
            **kwargs,
        }
        transient_status_codes = {429, 500, 502, 503, 504}

        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    f"{self.url}/v1/chat/completions",
                    json=body,
                    timeout=timeout,
                    headers=_modal_proxy_auth_headers(self.url),
                )
                if (
                    resp.status_code in transient_status_codes
                    and attempt < max_attempts
                ):
                    print(
                        f"Transient generation error {resp.status_code} from {self.url}; "
                        f"retrying ({attempt}/{max_attempts})..."
                    )
                    self.wait_until_ready(timeout=120)
                    time.sleep(min(2 * attempt, 5))
                    continue
                _raise_for_proxy_auth(resp.status_code, self.url)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= max_attempts:
                    raise
                print(
                    f"Transient generation transport error from {self.url}: {exc}; "
                    f"retrying ({attempt}/{max_attempts})..."
                )
                self.wait_until_ready(timeout=120)
                time.sleep(min(2 * attempt, 5))

        raise RuntimeError(
            f"Failed to generate from {self.url} after {max_attempts} attempts"
        )

    def generate(
        self,
        prompt: str | list[dict],
        ensure_ready: bool = True,
        **kwargs,
    ) -> str:
        messages = kwargs.pop("messages", None)
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        message = self.chat(
            messages,
            ensure_ready=ensure_ready,
            **kwargs,
        )
        content = message.get("content")
        if isinstance(content, str):
            return content
        if content is None:
            return message.get("reasoning_content", "")
        return str(content)

    def save(self) -> None:
        payload = self.model_dump(mode="json")
        vol_put(
            MetadataStore.DEPLOYMENTS,
            self.deployment_id,
            payload,
        )
        vol_upsert_summary_item(
            MetadataStore.DEPLOYMENTS_SUMMARY,
            payload,
            item_id_key="deployment_id",
            sort_key=lambda item: (
                str(item.get("deployment_config", {}).get("app_name", "")),
                str(item.get("deployment_id", "")),
            ),
            reverse=True,
        )

    def _start_log_tailer(self) -> "threading.Thread | None":
        """Spawn a daemon thread that streams deployed-app logs to stdout.

        Returns the thread (or None if we can't resolve the app).
        """
        import modal

        app_name = self.deployment_config.app_name
        env_name = (
            self.deployment_config.recipe.environment_name
            if self.deployment_config.recipe is not None
            else None
        )

        def _tail() -> None:
            try:
                app = modal.App.lookup(app_name, environment_name=env_name)
            except Exception:
                return

            async def _stream() -> None:
                async for line in app._logs():
                    print(line, end="", flush=True)

            try:
                asyncio.run(_stream())
            except Exception:
                pass

        t = threading.Thread(target=_tail, daemon=True)
        t.start()
        return t

    def wait_until_ready(self, timeout: int = 600) -> None:
        import time

        import requests

        app_name = self.deployment_config.app_name
        logs_hint = (
            f"`modal app logs {self.modal_app_id}`"
            if self.modal_app_id
            else f"`modal app logs {app_name}`"
        )
        print(f"Waiting for {app_name!r} — {self.modal_app_url}")

        log_thread = self._start_log_tailer()

        # A freshly-deployed SGLang container legitimately reports 0 running
        # containers for the whole cold-start window: pulling the (multi-GB)
        # image, downloading weights, and launching the server all happen before
        # it serves its first request. Modal allows the container `startup_timeout`
        # to reach readiness, so the crashloop heuristic must not fire inside that
        # window — otherwise a slow-but-healthy boot is misreported as
        # "crashlooping". Only flag a crashloop quickly once we've actually seen a
        # container come up and then disappear (a real restart cycle).
        recipe = self.deployment_config.recipe
        startup_timeout = recipe.startup_timeout if recipe is not None else 20 * 60
        # Cap the cold-start grace at the overall deadline — otherwise it always
        # exceeds `timeout` and the loop hits TimeoutError before the crashloop
        # check can fire, so cold-start crashloop detection never engages.
        detector = _CrashloopDetector(
            cold_start_grace=min(startup_timeout, timeout),
            restart_grace=60,
        )

        deadline = time.time() + timeout
        modal_poll_interval = 20
        last_modal_poll = 0.0
        last_probe = "no response yet"

        try:
            while time.time() < deadline:
                resp = None
                probe = last_probe
                try:
                    resp = requests.get(
                        f"{self.url}/v1/models",
                        timeout=10,
                        headers=_modal_proxy_auth_headers(self.url),
                    )
                    if resp.ok and resp.json().get("data"):
                        return
                    probe = f"HTTP {resp.status_code}"
                except requests.ConnectionError as exc:
                    probe = f"connection error ({type(exc).__name__})"
                except Exception as exc:
                    probe = f"{type(exc).__name__}: {exc}"

                if probe != last_probe:
                    last_probe = probe
                    print(f"[deployment] probe {self.url}/v1/models: {probe}")

                # A 401 is an auth problem, not a cold-start (loading returns
                # 502/503) — fail fast with a proxy-auth hint instead of polling
                # to the timeout. Checked outside the try so it isn't swallowed.
                if resp is not None:
                    _raise_for_proxy_auth(resp.status_code, self.url)

                now = time.time()
                if now - last_modal_poll >= modal_poll_interval:
                    last_modal_poll = now
                    containers = self._modal_container_count()
                    if containers is None:
                        raise RuntimeError(
                            f"Modal app {app_name!r} is not in a deployed state — "
                            f"the deploy likely failed or was stopped. Check {self.modal_app_url} "
                            f"and {logs_hint}."
                        )
                    crashloop = detector.observe(containers, now)
                    if crashloop is not None:
                        elapsed, phase = crashloop
                        raise RuntimeError(
                            f"Modal app {app_name!r} has had 0 running containers for "
                            f"~{elapsed}s ({phase}, last probe: {last_probe}). Containers "
                            f"are most likely crashlooping on startup (OOM, missing weights, "
                            f"bad config, etc.). Inspect logs with {logs_hint} or open "
                            f"{self.modal_app_url}."
                        )

                time.sleep(5)
        finally:
            if log_thread is not None and log_thread.is_alive():
                log_thread.join(timeout=2)

        raise TimeoutError(
            f"{self.url} not ready after {timeout}s (last probe: {last_probe}). "
            f"Inspect logs with {logs_hint} or open {self.modal_app_url}."
        )

    def _modal_container_count(self) -> int | None:
        """Container count for this deployment, or None if app isn't DEPLOYED."""
        try:
            apps = _run_coro(list_deployed_apps())
        except Exception as exc:
            print(f"[deployment] couldn't query Modal app state: {exc!r}")
            return -1
        for app in apps:
            if (
                app.app_id == self.modal_app_id
                or app.name == self.deployment_config.app_name
            ):
                return app.containers
        return None
