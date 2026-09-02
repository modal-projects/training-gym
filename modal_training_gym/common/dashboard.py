"""Identity of the deployed training-gym dashboard app.

Lives in ``common`` so library code — ``common.config``, the launchers — can
resolve the live dashboard without importing ``cli``, which imports ``common``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

DASHBOARD_APP_NAME = "training-gym-dashboard"
DASHBOARD_WEB_FUNCTION = "fastapi_app"

# Set on a per-PR dashboard deploy (scripts/previews/dashboard_api.py) so it
# serves the API without the scheduled jobs that rewrite shared metadata.
DASHBOARD_PREVIEW_ENV_KEY = "TRAINING_GYM_DASHBOARD_PREVIEW"

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_PRUNED_DIRS = frozenset({"__pycache__", "node_modules", "dist"})


def dashboard_frontend_dir() -> Path:
    """The frontend sources a dashboard image is built from.

    A repo checkout has them at ``dashboards/frontend``; a wheel ships a copy
    at ``modal_training_gym/_frontend`` (see ``pyproject.toml``).
    """
    checkout = _PACKAGE_DIR.parent / "dashboards" / "frontend"
    return checkout if checkout.is_dir() else _PACKAGE_DIR / "_frontend"


def current_dashboard_fingerprint() -> str:
    hasher = hashlib.sha256()
    for root, suffixes in ((_PACKAGE_DIR, (".py",)), (dashboard_frontend_dir(), None)):
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in _PRUNED_DIRS)
            for name in sorted(filenames):
                if suffixes is not None and not name.endswith(suffixes):
                    continue
                path = Path(dirpath) / name
                hasher.update(path.relative_to(root).as_posix().encode())
                hasher.update(path.read_bytes())
    return hasher.hexdigest()


def deployed_dashboard_url() -> str | None:
    """Return the live dashboard web URL if its app is deployed, else ``None``.

    Authoritative check against Modal (not the local toml): looks up the
    deployed ``fastapi_app`` function and returns its web URL. Any lookup
    failure — not deployed, no credentials, network blip — yields ``None``.
    """
    import modal

    try:
        fn = modal.Function.from_name(DASHBOARD_APP_NAME, DASHBOARD_WEB_FUNCTION)
        fn.hydrate()
        return fn.get_web_url()
    except Exception:
        return None
