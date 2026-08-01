"""Shared HTTP client for the deployed training-gym dashboard."""

from __future__ import annotations

import base64
import os
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Self
from urllib.parse import SplitResult, urlsplit

import httpx

from modal_training_gym.common.config import get_dashboard_url
from modal_training_gym.common.proxy_auth import modal_proxy_auth_headers

from .errors import CLIError, ExitCode


DASHBOARD_PASSWORD_ENV = "TRAINING_GYM_DASHBOARD_PASSWORD"
DEFAULT_TIMEOUT_SECONDS = 10.0
QueryParams = Mapping[str, str | int | float | bool | None]

_CREDENTIAL_HEADERS = ("Authorization", "Modal-Key", "Modal-Secret")


def _origin(parsed: SplitResult) -> tuple[str, str, int | None]:
    """(scheme, host, port) with default ports normalized to ``None``,
    matching how ``httpx.URL`` reports them."""
    port = parsed.port
    if (parsed.scheme, port) in {("http", 80), ("https", 443)}:
        port = None
    return (parsed.scheme, parsed.hostname or "", port)


def _credential_hook(
    origin: tuple[str, str, int | None],
    password: str,
    proxy_headers: Mapping[str, str],
) -> Callable[[httpx.Request], None]:
    """Attach the dashboard password and proxy-auth pair, on-origin only.

    A request hook rather than an ``httpx.Auth`` flow because it has to run on
    *every* redirect hop: httpx drops ``Authorization`` when the origin
    changes but copies custom headers along, so the pair needs its own
    boundary.
    """
    encoded = base64.b64encode(f"training-gym:{password}".encode()).decode("ascii")
    basic_header = f"Basic {encoded}" if password else ""

    def _apply(request: httpx.Request) -> None:
        if (request.url.scheme, request.url.host, request.url.port) == origin:
            if basic_header:
                request.headers["Authorization"] = basic_header
            for name, value in proxy_headers.items():
                request.headers[name] = value
        else:
            for name in _CREDENTIAL_HEADERS:
                if name in request.headers:
                    del request.headers[name]

    return _apply


class DashboardClient:
    """Transport shared by dashboard-backed commands."""

    def __init__(
        self,
        *,
        password: str | None = None,
    ) -> None:
        configured_url = (get_dashboard_url() or "").strip()
        if not configured_url:
            raise CLIError(
                "Dashboard URL is not configured.",
                error="dashboard_not_configured",
                exit_code=ExitCode.BACKEND,
                hint="training-gym setup",
            )

        parsed = urlsplit(configured_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CLIError(
                "Configured dashboard URL must use HTTP or HTTPS.",
                error="dashboard_configuration_invalid",
                exit_code=ExitCode.BACKEND,
                hint="training-gym setup",
            )

        dashboard_password = (
            os.environ.get(DASHBOARD_PASSWORD_ENV, "") if password is None else password
        )
        credential_hook = _credential_hook(
            _origin(parsed),
            dashboard_password,
            modal_proxy_auth_headers(configured_url),
        )
        self._client = httpx.Client(
            base_url=configured_url.rstrip("/") + "/",
            timeout=DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
            event_hooks={"request": [credential_hook]},
        )

    def get_json(
        self,
        path: str,
        *,
        params: QueryParams | None = None,
        not_found_error: CLIError | None = None,
    ) -> Any:
        """GET a dashboard-relative path and decode its JSON response."""
        parsed_path = urlsplit(path)
        if parsed_path.scheme or parsed_path.netloc:
            raise CLIError(
                "Dashboard request path must be relative.",
                error="invalid_dashboard_path",
            )

        query = (
            {key: value for key, value in params.items() if value is not None}
            if params
            else None
        )
        try:
            response = self._client.get(path.lstrip("/"), params=query)
        except httpx.TimeoutException as exc:
            raise CLIError(
                "Dashboard request timed out.",
                error="dashboard_timeout",
                exit_code=ExitCode.BACKEND,
            ) from exc
        except httpx.RequestError as exc:
            raise CLIError(
                "Could not connect to the dashboard.",
                error="dashboard_unreachable",
                exit_code=ExitCode.BACKEND,
                hint="training-gym setup",
            ) from exc

        if response.status_code == 404 and not_found_error is not None:
            raise not_found_error

        self._raise_for_status(response.status_code)
        try:
            return response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise CLIError(
                "Dashboard returned malformed JSON.",
                error="invalid_dashboard_response",
                exit_code=ExitCode.BACKEND,
            ) from exc

    @staticmethod
    def _raise_for_status(status_code: int) -> None:
        if status_code in {401, 403}:
            raise CLIError(
                "Dashboard authentication was rejected.",
                error="authentication_failed",
                exit_code=ExitCode.AUTH,
                hint=(
                    "training-gym set-password; for a dashboard deployed with "
                    "TRAINING_GYM_DASHBOARD_REQUIRES_PROXY_AUTH=1, configure the "
                    "Modal proxy-auth pair via training-gym setup or "
                    "MODAL_KEY/MODAL_SECRET"
                ),
            )
        if status_code == 404:
            raise CLIError(
                "The deployed dashboard does not support this resource.",
                error="dashboard_resource_not_found",
                exit_code=ExitCode.BACKEND,
                hint="training-gym setup",
            )
        if status_code >= 500:
            raise CLIError(
                f"Dashboard returned HTTP {status_code}.",
                error="dashboard_server_error",
                exit_code=ExitCode.BACKEND,
                status_code=status_code,
            )
        if status_code >= 400:
            raise CLIError(
                f"Dashboard returned HTTP {status_code}.",
                error="dashboard_request_failed",
                exit_code=ExitCode.BACKEND,
                status_code=status_code,
            )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
