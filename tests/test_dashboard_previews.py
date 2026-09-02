"""The dashboard preview's `/api` proxy target and per-PR backend naming."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "previews"))

from dashboard_api import app_name  # noqa: E402
from frontend_previews import (  # noqa: E402
    DEPLOYED_DASHBOARD_HOST,
    render_dashboard_conf,
)

TEMPLATE = (REPO_ROOT / "scripts/previews/nginx/dashboard.conf").read_text()


def test_conf_proxies_api_to_the_prs_own_backend():
    conf = render_dashboard_conf(
        TEMPLATE, "https://ws--training-gym-dashboard-pr-489-fastapi-app.modal.run"
    )

    host = "ws--training-gym-dashboard-pr-489-fastapi-app.modal.run"
    assert f"proxy_pass https://{host};" in conf
    assert f"proxy_set_header Host {host};" in conf
    assert "__API_HOST__" not in conf


def test_conf_without_a_pr_backend_keeps_using_the_deployed_dashboard():
    conf = render_dashboard_conf(TEMPLATE, None)

    assert f"proxy_pass https://{DEPLOYED_DASHBOARD_HOST};" in conf
    assert "__API_HOST__" not in conf


def test_conf_rejects_an_api_url_it_cannot_read_a_host_from():
    with pytest.raises(ValueError):
        render_dashboard_conf(TEMPLATE, "not-a-url")


def test_backend_app_name_is_per_pr():
    assert app_name(489) == "training-gym-dashboard-pr-489"
    assert app_name(489) != app_name(490)
