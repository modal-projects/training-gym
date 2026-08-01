from __future__ import annotations

import base64

import httpx
import pytest

from modal_training_gym.cli import client as client_module
from modal_training_gym.cli.client import (
    DEFAULT_TIMEOUT_SECONDS,
    DashboardClient,
)
from modal_training_gym.cli.errors import CLIError, ExitCode
from modal_training_gym.common import config as config_module


@pytest.fixture(autouse=True)
def configured_dashboard_url(monkeypatch, tmp_path):
    monkeypatch.setattr(
        client_module,
        "get_dashboard_url",
        lambda: "https://example.test",
    )
    # Keep the client hermetic: never read the developer's real
    # ~/.training-gym.toml (it may hold a proxy-auth pair) or env pair.
    monkeypatch.setattr(config_module, "CONFIG_PATH", tmp_path / "training-gym.toml")
    monkeypatch.delenv("MODAL_KEY", raising=False)
    monkeypatch.delenv("MODAL_SECRET", raising=False)
    monkeypatch.delenv("TRAINING_GYM_PROXY_AUTH_HOSTS", raising=False)


@pytest.fixture
def mock_transport(monkeypatch):
    real_client = httpx.Client

    def install(handler):
        monkeypatch.setattr(
            client_module.httpx,
            "Client",
            lambda **kwargs: real_client(
                **kwargs,
                transport=httpx.MockTransport(handler),
            ),
        )

    return install


def test_uses_configured_url_and_encodes_query(monkeypatch, mock_transport):
    monkeypatch.setattr(
        client_module, "get_dashboard_url", lambda: "https://example.test/root/"
    )
    seen = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    mock_transport(respond)
    with DashboardClient() as client:
        result = client.get_json(
            "/api/items",
            params={"model": "a b", "limit": 2, "unused": None},
        )

    assert result == {"ok": True}
    assert str(seen[0].url) == ("https://example.test/root/api/items?model=a+b&limit=2")
    assert seen[0].extensions["timeout"]["read"] == DEFAULT_TIMEOUT_SECONDS


def test_sends_basic_auth_when_password_exists(monkeypatch, mock_transport):
    monkeypatch.setenv("TRAINING_GYM_DASHBOARD_PASSWORD", "secret")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    mock_transport(respond)
    with DashboardClient() as client:
        client.get_json("/api/items")

    expected = base64.b64encode(b"training-gym:secret").decode()
    assert requests[0].headers["authorization"] == f"Basic {expected}"


def test_does_not_forward_basic_auth_to_redirected_host(monkeypatch, mock_transport):
    monkeypatch.setenv("TRAINING_GYM_DASHBOARD_PASSWORD", "secret")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "example.test":
            return httpx.Response(
                302,
                headers={"location": "https://other.test/api/items"},
            )
        return httpx.Response(200, json={})

    mock_transport(respond)
    with DashboardClient() as client:
        client.get_json("/api/items")

    assert requests[0].headers["authorization"].startswith("Basic ")
    assert "authorization" not in requests[1].headers


@pytest.mark.parametrize(
    ("location", "still_credentialed"),
    [
        ("https://other.test/api/items", False),
        ("https://example.test/api/items/", True),
    ],
    ids=["cross-origin", "same-origin"],
)
def test_proxy_pair_follows_only_same_origin_redirects(
    monkeypatch, mock_transport, location, still_credentialed
):
    """httpx drops ``Authorization`` when a redirect changes origin but copies
    custom headers along unchanged, so the pair needs its own boundary — and
    that boundary must not be so eager it strips the dashboard's own trailing
    slash redirects."""
    monkeypatch.setenv("MODAL_KEY", "wk-key")
    monkeypatch.setenv("MODAL_SECRET", "ws-secret")
    # Allowlisted so the egress guard releases the pair to the mock host.
    monkeypatch.setenv("TRAINING_GYM_PROXY_AUTH_HOSTS", "example.test")
    monkeypatch.setenv("TRAINING_GYM_DASHBOARD_PASSWORD", "secret")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if str(request.url) == "https://example.test/api/items":
            return httpx.Response(302, headers={"location": location})
        return httpx.Response(200, json={})

    mock_transport(respond)
    with DashboardClient() as client:
        client.get_json("/api/items")

    assert requests[0].headers["modal-key"] == "wk-key"
    redirected = requests[1].headers
    assert ("modal-key" in redirected) is still_credentialed
    assert ("modal-secret" in redirected) is still_credentialed
    assert ("authorization" in redirected) is still_credentialed


def test_omits_auth_when_password_is_absent(monkeypatch, mock_transport):
    monkeypatch.delenv("TRAINING_GYM_DASHBOARD_PASSWORD", raising=False)
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    mock_transport(respond)
    with DashboardClient() as client:
        client.get_json("/api/items")

    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize(
    ("status_code", "error", "exit_code"),
    [
        (401, "authentication_failed", ExitCode.AUTH),
        (403, "authentication_failed", ExitCode.AUTH),
        (404, "dashboard_resource_not_found", ExitCode.BACKEND),
        (500, "dashboard_server_error", ExitCode.BACKEND),
        (503, "dashboard_server_error", ExitCode.BACKEND),
        (400, "dashboard_request_failed", ExitCode.BACKEND),
    ],
)
def test_maps_http_errors(status_code, error, exit_code, mock_transport):
    mock_transport(lambda _request: httpx.Response(status_code, text="ignored"))
    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.error == error
    assert exc_info.value.exit_code == exit_code


def test_uses_command_specific_not_found_error(mock_transport):
    mock_transport(lambda _request: httpx.Response(404, text="ignored"))
    not_found = CLIError(
        "Run not found.",
        error="run_not_found",
        exit_code=ExitCode.NOT_FOUND,
        run_id="run-1",
    )

    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/runs/run-1", not_found_error=not_found)

    assert exc_info.value is not_found
    assert exc_info.value.error == "run_not_found"
    assert exc_info.value.exit_code == ExitCode.NOT_FOUND


def test_maps_timeout_without_leaking_transport_details(mock_transport):
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret transport detail", request=request)

    mock_transport(timeout)
    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.error == "dashboard_timeout"
    assert exc_info.value.exit_code == ExitCode.BACKEND
    assert "secret transport detail" not in str(exc_info.value)


def test_maps_connection_failure(mock_transport):
    def disconnect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("host detail", request=request)

    mock_transport(disconnect)
    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.error == "dashboard_unreachable"
    assert exc_info.value.exit_code == ExitCode.BACKEND


def test_rejects_malformed_json(mock_transport):
    mock_transport(lambda _request: httpx.Response(200, text="<html>not json</html>"))

    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("/api/items")

    assert exc_info.value.error == "invalid_dashboard_response"
    assert exc_info.value.exit_code == ExitCode.BACKEND


@pytest.mark.parametrize(
    "url",
    [None, "", "not-a-url", "ftp://example.test"],
)
def test_rejects_missing_or_invalid_configuration(monkeypatch, url):
    monkeypatch.setattr(client_module, "get_dashboard_url", lambda: url)

    with pytest.raises(CLIError) as exc_info:
        DashboardClient()

    assert exc_info.value.exit_code == ExitCode.BACKEND


def test_rejects_absolute_request_path():
    with DashboardClient() as client:
        with pytest.raises(CLIError) as exc_info:
            client.get_json("https://other.test/api/items")

    assert exc_info.value.error == "invalid_dashboard_path"
    assert exc_info.value.exit_code == ExitCode.ERROR
