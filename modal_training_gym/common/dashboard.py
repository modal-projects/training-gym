"""Identity of the deployed training-gym dashboard app.

Lives in ``common`` so library code — ``common.config``, the launchers — can
resolve the live dashboard without importing ``cli``, which imports ``common``.
"""

from __future__ import annotations

DASHBOARD_APP_NAME = "training-gym-dashboard"
DASHBOARD_WEB_FUNCTION = "fastapi_app"

# Set on a per-PR dashboard deploy (scripts/previews/dashboard_api.py) so it
# serves the API without the scheduled jobs that rewrite shared metadata.
DASHBOARD_PREVIEW_ENV_KEY = "TRAINING_GYM_DASHBOARD_PREVIEW"


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
