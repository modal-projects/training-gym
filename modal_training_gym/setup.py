"""Deploy the training-gym dashboard to Modal.

Usage (Python):
    import modal_training_gym
    modal_training_gym.setup()

Usage (CLI):
    training-gym setup

What this does:
1. Provisions a ``_training-gym-modal-creds`` Modal Secret containing
   ``MODAL_TOKEN_ID`` + ``MODAL_TOKEN_SECRET``. The deployed dashboard uses
   these credentials to stream logs from *other* Modal apps (the user's
   training runs) into the dashboard UI — container-default creds aren't
   scoped to read cross-app logs, so a workspace token is required.

   The underscore prefix marks the secret as auto-managed (hidden from the
   normal Secrets list in the UI). Credentials are auto-resolved from
   ``MODAL_TOKEN_*`` env vars or the active profile in ``~/.modal.toml``.

2. Deploys the dashboard's ASGI app.
3. Persists the FastAPI web URL to ``~/.training-gym.toml`` so other
   clients (e.g. the slime launcher) can find the dashboard.
"""

from __future__ import annotations

from modal_training_gym._dashboard import (
    MODAL_CREDS_SECRET_NAME,
    ensure_creds_secret,
)

DASHBOARD_APP_NAME = "training-gym-dashboard"
DASHBOARD_WEB_FUNCTION = "fastapi_app"


def setup(interactive: bool = True) -> str:
    """Deploy the training-gym dashboard, persist its URL, and return it.

    ``interactive=False`` resolves Modal credentials silently (from env vars
    or ``~/.modal.toml``) and never prompts — used by the auto-deploy path in
    ``TrainConfig.train()`` so a training run is never blocked on stdin.
    """
    import modal

    from modal_training_gym._dashboard import app, fastapi_app
    from modal_training_gym.common.config import CONFIG_PATH, save_dashboard_url

    if not ensure_creds_secret(interactive=interactive):
        print(
            f"WARNING: continuing without the {MODAL_CREDS_SECRET_NAME!r} "
            "Modal Secret — the dashboard will not be able to stream Modal "
            "app logs."
        )

    with modal.enable_output():
        app.deploy()

    web_url = fastapi_app.get_web_url()
    save_dashboard_url(web_url)
    print(f"\nDashboard deployed: {web_url}")
    print(f"Saved dashboard URL to {CONFIG_PATH}")
    return web_url


def deployed_dashboard_url() -> str | None:
    """Return the live dashboard web URL if its app is deployed, else ``None``.

    Authoritative check against Modal (not the local toml): looks up the
    deployed ``fastapi_app`` function and returns its web URL. Any lookup
    failure — not deployed, no credentials, network blip — yields ``None``.
    """
    import modal
    from modal.exception import NotFoundError

    try:
        fn = modal.Function.from_name(DASHBOARD_APP_NAME, DASHBOARD_WEB_FUNCTION)
        fn.hydrate()
        return fn.get_web_url()
    except NotFoundError:
        return None
    except Exception:
        return None


def ensure_dashboard_deployed() -> str | None:
    """Deploy the dashboard if it isn't already; return its web URL (or ``None``).

    Idempotent: if the app is already deployed we only reconcile the cached URL
    in ``~/.training-gym.toml`` and return; we never redeploy.

    Best-effort and guaranteed not to raise: this is called from the hot path
    of ``train()`` and ``evaluate()``, where dashboard provisioning is a
    convenience, not a precondition. Any failure — Modal deploy errors
    (network, auth, image build, outage), a read-only/full disk on the toml
    write, or an import error — is swallowed with a warning so the run itself
    is never aborted. Callers therefore don't need their own try/except.
    """
    try:
        from modal_training_gym.common.config import (
            get_dashboard_url,
            save_dashboard_url,
        )

        web_url = deployed_dashboard_url()
        if web_url:
            # Already deployed — keep the local toml in sync with the live URL.
            if get_dashboard_url() != web_url:
                save_dashboard_url(web_url)
            return web_url

        print(
            f"Training-gym dashboard ({DASHBOARD_APP_NAME!r}) is not deployed — "
            "deploying it now (this happens once)."
        )
        return setup(interactive=False)
    except Exception as exc:
        print(
            f"WARNING: could not ensure the training-gym dashboard is deployed: "
            f"{exc}. Continuing without dashboard status reporting; run "
            "`training-gym setup` to deploy it manually."
        )
        return None
