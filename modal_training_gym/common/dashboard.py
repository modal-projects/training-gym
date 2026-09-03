"""Identity of the deployed training-gym dashboard app.

Lives in ``common`` so library code — ``common.config``, the launchers — can
resolve the live dashboard without importing ``cli``, which imports ``common``.
"""

from __future__ import annotations

import os
from pathlib import Path

DASHBOARD_APP_NAME = "training-gym-dashboard"
DASHBOARD_WEB_FUNCTION = "fastapi_app"

# Set on a per-PR dashboard deploy (scripts/previews/dashboard_api.py) so it
# serves the API without the scheduled jobs that rewrite shared metadata.
DASHBOARD_PREVIEW_ENV_KEY = "TRAINING_GYM_DASHBOARD_PREVIEW"
DASHBOARD_VERSION_ENV_KEY = "DASHBOARD_VERSION"

# Bump when the deployed dashboard frontend or backend changes.
DASHBOARD_VERSION = 1

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def dashboard_frontend_dir() -> Path:
    """The frontend sources a dashboard image is built from.

    A repo checkout has them at ``dashboards/frontend``; a wheel ships a copy
    at ``modal_training_gym/_frontend`` (see ``pyproject.toml``).
    """
    checkout = _PACKAGE_DIR.parent / "dashboards" / "frontend"
    return checkout if checkout.is_dir() else _PACKAGE_DIR / "_frontend"


def current_dashboard_version() -> str:
    if baked := os.environ.get(DASHBOARD_VERSION_ENV_KEY):
        return baked
    return str(DASHBOARD_VERSION)


def is_dashboard_upgrade(incoming: str, deployed: str | None) -> bool:
    if not deployed:
        return True
    try:
        return int(incoming) > int(deployed)
    except ValueError:
        return True


class DashboardLookupUnknown(Exception):
    """Modal lookup of the dashboard app failed before not-found could be observed."""


def deployed_dashboard_url() -> str | None:
    """Return the live dashboard web URL if its app is deployed, else ``None``.

    Raises ``DashboardLookupUnknown`` when lookup fails before that can
    be observed (timeout, auth, network, unexpected Modal errors).
    """
    import modal
    from modal.exception import NotFoundError

    try:
        fn = modal.Function.from_name(DASHBOARD_APP_NAME, DASHBOARD_WEB_FUNCTION)
        fn.hydrate()
        url = fn.get_web_url()
    except NotFoundError:
        return None
    except Exception as exc:
        raise DashboardLookupUnknown from exc
    if not url:
        raise DashboardLookupUnknown
    return url
