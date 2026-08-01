"""Ingestion-route authorization and the opt-in edge-auth flag."""

from __future__ import annotations

import importlib
import os

import modal
import pytest
from fastapi.testclient import TestClient

from modal_training_gym import _dashboard
from modal_training_gym.common import config
from modal_training_gym.common.framework import Framework
from modal_training_gym.common.run import TrainingRun
from modal_training_gym.utils import metadata
from modal_training_gym.utils.metadata import MetadataStore

RUN_ID = "run-auth-1"
TOKEN = "test-status-token-run-auth-1"


@pytest.fixture(autouse=True)
def isolated_config_file(monkeypatch, tmp_path):
    """The sticky flag reads/writes the developer's real config otherwise."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "training-gym.toml")


# Each ingestion POST reads training_run_id off a *different* pydantic model,
# so they are parametrized rather than sampled: an auth check wired to the
# wrong handler's variable is the mistake this catches.
ROUTES = [
    ("/api/framework-status", {"phase": "training"}),
    ("/api/training-rollouts", {"rollout_id": 1}),
    ("/api/advantage-distributions", {"rollout_id": 1}),
]
ROUTE_IDS = [route.rsplit("/", 1)[-1] for route, _ in ROUTES]


def _client(monkeypatch, tmp_path) -> TestClient:
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("ok")
    (static / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(_dashboard, "STATIC_DIR", str(static))
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    return TestClient(_dashboard.fastapi_app.local())


def _save_run(token: str | None = TOKEN) -> None:
    TrainingRun(
        training_run_id=RUN_ID,
        modal_app_id="ap-auth",
        framework=Framework.SLIME,
        config={"model": {"model_name": "Qwen/Qwen3-4B"}},
        created_at=100,
        started_at=100,
        updated_at=150,
    ).save()
    if token is not None:
        metadata.vol_put(
            MetadataStore.FRAMEWORK_STATUS_TOKENS, RUN_ID, {"token": token}
        )


@pytest.mark.parametrize(("route", "extra"), ROUTES, ids=ROUTE_IDS)
def test_every_rejection_is_indistinguishable(
    fake_volume, monkeypatch, tmp_path, route, extra
):
    """Rejections must not reveal whether a run id exists.

    The handlers used to look the run up (404) before checking the token
    (403), so an anonymous caller could enumerate real training-run ids by
    the status code alone. Asserting the responses are *equal* — rather than
    each being 403 — is what pins the property: it fails just as loudly if a
    future change makes unknown runs 404 again, or gives any one failure
    mode a distinguishing body.
    """
    _save_run()
    rejections = {
        "unknown run, no credentials": ("ghost-run", {}),
        "unknown run, valid-looking token": ("ghost-run", f"Bearer {TOKEN}"),
        "real run, no credentials": (RUN_ID, {}),
        "real run, wrong token": (RUN_ID, "Bearer wrong-token"),
        "real run, Basic instead of Bearer": (RUN_ID, "Basic dHJhaW5pbmctZ3ltOnB3"),
        "real run, bare Bearer": (RUN_ID, "Bearer"),
    }

    seen = {}
    with _client(monkeypatch, tmp_path) as client:
        for label, (run_id, auth) in rejections.items():
            response = client.post(
                route,
                json={"training_run_id": run_id, **extra},
                headers={"Authorization": auth} if isinstance(auth, str) else {},
            )
            seen[label] = (response.status_code, response.json())

    refused = (403, {"detail": "Invalid status token"})
    assert seen == dict.fromkeys(rejections, refused)


@pytest.mark.parametrize(("route", "extra"), ROUTES, ids=ROUTE_IDS)
def test_correct_token_still_reaches_the_handler(
    fake_volume, monkeypatch, tmp_path, route, extra
):
    """The ingestion path every training container depends on."""
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            route,
            json={"training_run_id": RUN_ID, **extra},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("stored", [None, ""], ids=["no-record", "empty-token"])
def test_run_with_no_token_on_record_refuses_the_dummy(
    fake_volume, monkeypatch, tmp_path, stored
):
    """The dummy the no-token path compares against must never authenticate.

    That path exists so a missing token record costs the same comparison as
    a wrong token; the constant is public, so a refactor that let it match
    would hand every tokenless run a skeleton key.
    """
    _save_run(token=stored)
    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/framework-status",
            json={"training_run_id": RUN_ID, "phase": "training"},
            headers={"Authorization": f"Bearer {_dashboard._MISSING_TOKEN_DUMMY}"},
        )

    assert response.status_code == 403


def test_payload_validation_still_runs_after_auth(fake_volume, monkeypatch, tmp_path):
    """Authenticating first must not swallow the handler's own 400s."""
    _save_run()
    with _client(monkeypatch, tmp_path) as client:
        response = client.post(
            "/api/framework-status",
            json={"training_run_id": RUN_ID, "phase": "not-a-phase"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 400


# ── TRAINING_GYM_DASHBOARD_REQUIRES_PROXY_AUTH ────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("1", True),
        (" TRUE ", True),
        # A truthy-looking value that parsed as "off" would deploy an open
        # dashboard while the operator believed otherwise — so it must raise.
        ("yes", ValueError),
        ("on", ValueError),
        ("2", ValueError),
    ],
)
def test_flag_parses_strictly(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv(_dashboard.REQUIRES_PROXY_AUTH_ENV, raising=False)
    else:
        monkeypatch.setenv(_dashboard.REQUIRES_PROXY_AUTH_ENV, raw)

    if expected is ValueError:
        with pytest.raises(ValueError):
            _dashboard._requires_proxy_auth()
    else:
        assert _dashboard._requires_proxy_auth() is expected


@pytest.mark.parametrize(("raw", "expected"), [("1", True), (None, False)])
def test_flag_reaches_the_asgi_decorator(monkeypatch, tmp_path, raw, expected):
    """The flag is only worth anything if it lands on the decorator.

    Parsing it correctly and then not passing it through would leave a
    deployment silently unprotected — the one failure this whole knob exists
    to prevent — and no assertion on ``_requires_proxy_auth`` alone can see
    it. Re-imports the module (which is where the decorator runs) with
    ``modal.asgi_app`` recording its keyword arguments.
    """
    # Module import provisions a Modal Secret from ~/.modal.toml when it can
    # find credentials; point that lookup at nothing so the reload stays local.
    monkeypatch.setattr(config, "MODAL_CONFIG_PATH", tmp_path / "absent.toml")
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    if raw is None:
        monkeypatch.delenv(_dashboard.REQUIRES_PROXY_AUTH_ENV, raising=False)
    else:
        monkeypatch.setenv(_dashboard.REQUIRES_PROXY_AUTH_ENV, raw)

    recorded: dict[str, object] = {}
    real_asgi_app = modal.asgi_app

    def recording_asgi_app(**kwargs):
        recorded.update(kwargs)
        return real_asgi_app(**kwargs)

    monkeypatch.setattr(modal, "asgi_app", recording_asgi_app)
    try:
        importlib.reload(_dashboard)
    finally:
        monkeypatch.setattr(modal, "asgi_app", real_asgi_app)
        # Other modules hold a reference to this module object; reload once
        # more, unpatched, so they see it in its default state (the sticky
        # persisted choice must go too).
        os.environ.pop(_dashboard.REQUIRES_PROXY_AUTH_ENV, None)
        config.CONFIG_PATH.unlink(missing_ok=True)
        importlib.reload(_dashboard)

    assert recorded == {"requires_proxy_auth": expected}


def test_unset_flag_keeps_the_persisted_choice(monkeypatch):
    """A redeploy from a shell that doesn't export the flag must keep the
    persisted choice instead of silently reopening the dashboard; downgrading
    takes an explicit 0, which is just as sticky."""
    monkeypatch.setenv(_dashboard.REQUIRES_PROXY_AUTH_ENV, "1")
    assert _dashboard._requires_proxy_auth() is True

    monkeypatch.delenv(_dashboard.REQUIRES_PROXY_AUTH_ENV)
    assert _dashboard._requires_proxy_auth() is True

    monkeypatch.setenv(_dashboard.REQUIRES_PROXY_AUTH_ENV, "0")
    assert _dashboard._requires_proxy_auth() is False

    monkeypatch.delenv(_dashboard.REQUIRES_PROXY_AUTH_ENV)
    assert _dashboard._requires_proxy_auth() is False


def test_containers_never_touch_the_config_file(monkeypatch):
    """In-container re-imports must neither create a config file nor
    resurrect a persisted choice; the edge already has its config."""
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")

    monkeypatch.setenv(_dashboard.REQUIRES_PROXY_AUTH_ENV, "1")
    assert _dashboard._requires_proxy_auth() is True
    assert not config.CONFIG_PATH.exists()

    config.save_dashboard_requires_proxy_auth(True)
    monkeypatch.delenv(_dashboard.REQUIRES_PROXY_AUTH_ENV)
    assert _dashboard._requires_proxy_auth() is False
