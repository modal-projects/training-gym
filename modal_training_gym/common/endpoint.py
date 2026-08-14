from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from typing import Any

import httpx
import modal

from modal_training_gym.common.checkpoint import Checkpoint, CheckpointType
from modal_training_gym.common.config import modal_proxy_auth_headers
from modal_training_gym.common.errors import TrainingGymConfigError
from modal_training_gym.common.openai_messages import _messages_to_openai
from modal_training_gym.model import ModelConfig


def _create_endpoint_and_wait_for_url(
    *,
    endpoint_name: str,
    model_name: str,
    checkpoint: Checkpoint | None,
    unauthenticated: bool,
    environment: str | None,
    routing_region: str | None,
    wait_timeout_sec: float,
) -> str:
    command = [
        sys.executable,
        "-m",
        "modal",
        "endpoint",
        "create",
        "--name",
        endpoint_name,
        "--model",
        model_name,
    ]

    if environment:
        command.extend(["--env", environment])
    if unauthenticated:
        command.append("--unauthenticated")
    if routing_region:
        command.extend(["--routing-region", routing_region])

    if checkpoint:
        command.extend(["--custom-volume-name", checkpoint.checkpoints_volume_name])
        command.extend(["--custom-volume-path", checkpoint.path_relative_to_volume])

    subprocess.run(command, check=True, timeout=120)

    server = modal.Server.from_name(
        f"ep-{endpoint_name}", "Server", environment_name=environment
    )
    deadline = time.monotonic() + wait_timeout_sec
    while time.monotonic() < deadline:
        try:
            raw = server.get_url()
        except modal.exception.NotFoundError:
            raw = None
        if raw:
            return raw.rstrip("/")
        time.sleep(1)
    else:
        raise TimeoutError(
            f"Timed out waiting for a URL for endpoint {endpoint_name!r}"
        )


class Endpoint:
    """A handle to a [Modal Endpoint](https://modal.com/docs/guide/endpoints).

    Use ``Endpoint.launch()`` to provision one, or construct this class
    directly to talk to an endpoint that already exists.

    An endpoint serves the OpenAI Chat Completions API, so you can use any
    OpenAI-compatible client in addition to ``chat()``.

    Endpoints require a Modal [proxy token](https://modal.com/docs/guide/webhook-proxy-auth)
    pair when launched with ``unauthenticated=False``. Export it as ``MODAL_KEY`` /
    ``MODAL_SECRET`` environment variables or save them with ``training-gym set-proxy-auth``.

    Endpoints outlive the process that launched them. List them with
    ``modal endpoint list`` and tear one down with ``modal endpoint stop <name>``.

    ## Attributes

    url : str
        Base URL of the endpoint.
    endpoint_name : str
        Modal endpoint name.
    model_name : str
        Base model ID sent in request bodies.
    requires_proxy_auth : bool
        Whether a proxy token is required to use the endpoint.
    """

    url: str
    endpoint_name: str
    model_name: str
    requires_proxy_auth: bool

    def __init__(
        self,
        url: str,
        *,
        endpoint_name: str,
        model_name: str,
        requires_proxy_auth: bool,
    ):
        self.endpoint_name = endpoint_name
        self.model_name = model_name
        self.url = url.rstrip("/")
        self.requires_proxy_auth = requires_proxy_auth

    @classmethod
    def launch(
        cls,
        model: ModelConfig | str,
        checkpoint: Checkpoint | None = None,
        *,
        endpoint_name: str | None = None,
        unauthenticated: bool = True,
        environment: str | None = None,
        routing_region: str | None = None,
        wait_timeout_sec: float = 300,
    ):
        """Provision a Modal endpoint for ``model`` and return a handle to it.

        Shells out to ``modal endpoint create``; see the [Modal Endpoints
        guide](https://modal.com/docs/guide/endpoints) for the full set of
        options and the catalog of supported model families.

        When ``endpoint_name`` is omitted, an endpoint name is derived for you.

        Endpoints require proxy auth if ``unauthenticated=False``.

        Returns once the endpoint has a URL, which may occur before it can serve
        traffic; call ``wait_until_ready()`` to wait for the model to become ready.
        Raises ``TimeoutError`` if no URL is published within ``wait_timeout_sec``.
        """
        if checkpoint and checkpoint.checkpoint_type is not CheckpointType.hf:
            raise TrainingGymConfigError(
                "Checkpoint must be in Hugging Face format. Convert it with "
                "`convert_checkpoint_to_hf()` first."
            )

        model_name = model if isinstance(model, str) else model.model_name

        if not endpoint_name:
            spec = {
                "model": model_name,
                "routing_region": routing_region,
                "unauthenticated": unauthenticated,
            }

            if checkpoint:
                spec.update(
                    {
                        "checkpoint_run": checkpoint.training_run_id,
                        "checkpoint_name": checkpoint.name,
                    }
                )

            digest = hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:12]
            endpoint_name = f"training-gym-{digest}"

        url = _create_endpoint_and_wait_for_url(
            endpoint_name=endpoint_name,
            model_name=model_name,
            checkpoint=checkpoint,
            unauthenticated=unauthenticated,
            environment=environment,
            routing_region=routing_region,
            wait_timeout_sec=wait_timeout_sec,
        )

        return cls(
            url,
            endpoint_name=endpoint_name,
            model_name=model_name,
            requires_proxy_auth=not unauthenticated,
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.requires_proxy_auth:
            headers = modal_proxy_auth_headers()
            if not headers:
                raise TrainingGymConfigError(
                    "Proxy authentication requires an HTTPS Modal URL and "
                    "MODAL_KEY and MODAL_SECRET."
                )
        return headers

    def wait_until_ready(self, timeout_sec: float = 30 * 60) -> None:
        """Block until the endpoint can serve traffic.

        Raises ``TimeoutError`` if the endpoint is still not ready by then, and
        ``RuntimeError`` if the endpoint rejects the proxy credentials.
        """
        last_error: Exception | None = None
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"{self.url}/v1/models",
                    headers=self._headers(),
                    timeout=30,
                    follow_redirects=False,
                )
                if response.status_code == 200:
                    return
                if response.status_code in {401, 403}:
                    raise RuntimeError(
                        f"Endpoint {self.endpoint_name} rejected proxy authentication. "
                        "Run `training-gym set-proxy-auth` and retry."
                    )
                if response.status_code not in {404, 429, 500, 502, 503, 504}:
                    response.raise_for_status()
                last_error = RuntimeError(
                    f"Endpoint {self.endpoint_name} readiness returned HTTP {response.status_code}"
                )
            except httpx.RequestError as exc:
                last_error = exc
            time.sleep(2)
        raise TimeoutError(
            f"Timed out waiting for endpoint {self.endpoint_name} at {self.url} to become ready"
        ) from last_error

    def chat(
        self,
        messages: list[dict[str, Any]],
        timeout: int = 120,
        max_attempts: int = 4,
        extra_parameters: dict[str, Any] | None = None,
    ):
        """POST one chat completion to ``/v1/chat/completions``.

        ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts, and
        ``extra_parameters`` carries any other body fields the OpenAI Chat
        Completions API accepts, such as ``temperature`` or ``max_tokens``.
        Returns the assistant message as a dict, preserving structured fields
        like ``tool_calls`` and ``reasoning_content``.

        Requests are retried up to ``max_attempts`` times with a short backoff,
        while ``timeout`` bounds each individual request. Raises ``RuntimeError``
        if the endpoint rejects the proxy credentials, and propagates the
        underlying ``httpx`` error on failure.
        """
        url = f"{self.url}/v1/chat/completions"
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": _messages_to_openai(messages),
        }
        if extra_parameters:
            body.update(extra_parameters)

        headers = self._headers()
        transient = {429, 500, 502, 503, 504}
        for attempt in range(1, max_attempts + 1):
            try:
                resp = httpx.post(
                    url,
                    json=body,
                    timeout=timeout,
                    headers=headers,
                    follow_redirects=False,
                )
                if resp.status_code in transient and attempt < max_attempts:
                    time.sleep(min(2 * attempt, 5))
                    continue
                if resp.status_code in {401, 403}:
                    raise RuntimeError(
                        f"HTTP {resp.status_code} from {url}. Proxy credentials were "
                        "rejected. Refresh them with `modal workspace proxy-tokens "
                        "create` and export MODAL_KEY and MODAL_SECRET."
                    )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]
            except httpx.RequestError:
                if attempt >= max_attempts:
                    raise
                time.sleep(min(2 * attempt, 5))

        raise RuntimeError(
            f"Chat completions exhausted {max_attempts} attempts at {url}"
        )
