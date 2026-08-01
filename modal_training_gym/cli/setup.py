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

2. Deploys the dashboard's ASGI app and orphan reconciler.
3. Persists the FastAPI web URL to ``~/.training-gym.toml`` so other
   clients (e.g. the slime launcher) can find the dashboard.
"""

from __future__ import annotations
import webbrowser

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
    from modal_training_gym.common.config import (
        CONFIG_PATH,
        get_dashboard_requires_proxy_auth,
        save_dashboard_url,
    )

    ensure_proxy_auth(interactive=interactive)

    # The decorator resolved the sticky flag at import; say which way it went,
    # so neither a redeploy that stays closed despite an unset env var nor a
    # machine without the persisted choice deploying open is ever a surprise.
    if get_dashboard_requires_proxy_auth():
        print(
            "Dashboard edge auth: on — Modal rejects anonymous requests at its "
            "edge. Sticky across redeploys; export "
            "TRAINING_GYM_DASHBOARD_REQUIRES_PROXY_AUTH=0 to deploy open."
        )
    else:
        print(
            "Dashboard edge auth: off — Modal's edge forwards anonymous "
            "requests to the dashboard. Export "
            "TRAINING_GYM_DASHBOARD_REQUIRES_PROXY_AUTH=1 to reject them there."
        )

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


def ensure_proxy_auth(interactive: bool = True, force: bool = False) -> bool:
    """Prompt for and persist Modal proxy-auth tokens in ``~/.training-gym.toml``.

    Authenticated served endpoints (``unauthenticated=False``) need Modal proxy auth
    and need a ``MODAL_KEY`` / ``MODAL_SECRET`` token pair. When ``interactive``
    we ask whether the user has created a pair and, if so, read and save it.
    With ``force`` we offer to replace an already-saved pair (e.g. when the old
    one was minted in the wrong workspace). Returns ``True`` when both tokens are
    available afterwards.
    """
    from getpass import getpass

    from modal_training_gym.common.config import (
        CONFIG_PATH,
        get_proxy_auth,
        save_proxy_auth,
    )

    key, secret = get_proxy_auth()
    if key and secret and not force:
        return True
    if not interactive:
        return bool(key and secret)

    if key and secret:
        print(
            f"\nA proxy-auth pair is already saved in {CONFIG_PATH} "
            f"(MODAL_KEY {key[:6]}…). Proxy-auth tokens are workspace-scoped — "
            "if endpoints return 401 the pair was likely minted in the wrong "
            "workspace."
        )
        answer = input("Replace it? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            return True
    else:
        print(
            "\nAuthenticated served endpoints (DeploymentConfig(unauthenticated=False)) need "
            "a Modal proxy-auth token pair (MODAL_KEY / MODAL_SECRET)."
        )
        answer = (
            input(
                "Have you created a proxy-auth token pair at "
                "https://modal.com/settings/proxy-auth-tokens? [y/N] "
            )
            .strip()
            .lower()
        )
        if answer not in ("y", "yes"):
            print(
                "Skipping proxy-auth setup. Create a pair at "
                "https://modal.com/settings/proxy-auth-tokens and re-run "
                "`training-gym set-proxy-auth`, or export MODAL_KEY / "
                "MODAL_SECRET yourself."
            )
            return False

    key = input("MODAL_KEY (wk-…): ").strip()
    secret = getpass("MODAL_SECRET (ws-…): ").strip()
    if not (key and secret):
        print("Both MODAL_KEY and MODAL_SECRET are required — skipping.")
        return False
    if not key.startswith("wk-") or not secret.startswith("ws-"):
        print(
            "WARNING: expected MODAL_KEY to start with 'wk-' and MODAL_SECRET "
            "with 'ws-'. Saving anyway."
        )

    save_proxy_auth(key, secret)
    print(f"Saved proxy-auth tokens to {CONFIG_PATH}")
    print(
        "\nThis token cannot access any environments by default. You must "
        "configure which environments it is valid for before it can be used.\n"
        "Set this at https://modal.com/settings/proxy-auth-tokens — make sure "
        "the environment your endpoint runs in is enabled for "
        "this token."
    )
    return True


def set_proxy_auth() -> bool:
    """Interactively (re)set the saved proxy-auth token pair.

    Thin wrapper over :func:`ensure_proxy_auth` with ``force=True`` so an
    existing pair is replaced — exposed as ``training-gym set-proxy-auth``.
    """
    return ensure_proxy_auth(interactive=True, force=True)


def set_password(password: str | None = None) -> None:
    """Set or clear the dashboard password, then redeploy so it takes effect.

    Pass an empty string to disable auth. When ``password`` is ``None`` we
    prompt for it (hidden input). The deployed app reads the value from its
    environment at startup, so we redeploy after updating the Secret.
    """
    from getpass import getpass

    from modal_training_gym._dashboard import set_dashboard_password

    if password is None:
        password = getpass("Dashboard password (leave empty to disable auth): ").strip()
        if password:
            confirm = getpass("Confirm password: ").strip()
            if confirm != password:
                print("Passwords don't match — aborting.")
                return

    set_dashboard_password(password)
    if password:
        print("Dashboard password set. Redeploying so it takes effect...")
    else:
        print("Dashboard password cleared (open access). Redeploying...")

    setup(interactive=False)


def open_dashboard() -> str | None:
    """Open the deployed dashboard in the default browser; return its URL.

    Resolves the live URL from Modal (authoritative) and keeps the cached
    ``~/.training-gym.toml`` value in sync, falling back to that cache if the
    Modal lookup fails. Prints guidance and returns ``None`` when nothing is
    deployed.
    """
    from modal_training_gym.common.config import get_dashboard_url, save_dashboard_url

    web_url = deployed_dashboard_url()
    if web_url:
        if get_dashboard_url() != web_url:
            save_dashboard_url(web_url)
    else:
        web_url = get_dashboard_url()

    if not web_url:
        print(
            "No deployed training-gym dashboard found. "
            "Run `training-gym setup` to deploy it first."
        )
        return None

    print(f"Opening dashboard: {web_url}")
    webbrowser.open(web_url)
    return web_url


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
