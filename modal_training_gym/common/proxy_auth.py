"""Modal proxy-auth plumbing shared by the dashboard HTTP clients.

An endpoint deployed with ``requires_proxy_auth=True`` is enforced at Modal's
edge: without a workspace ``wk-``/``ws-`` pair in ``Modal-Key``/``Modal-Secret``
the request never reaches the app. The serving path already speaks this via
``DeploymentConfig(unauthenticated=False)``; this gives the status reporters
and the CLI the same ability, and is inert when no pair is configured.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

PROXY_AUTH_KEY_ENV = "MODAL_KEY"
PROXY_AUTH_SECRET_ENV = "MODAL_SECRET"

# Comma-separated extra hostnames the pair may be sent to (e.g. a custom
# domain in front of Modal); non-Modal hosts not named here never see it.
PROXY_AUTH_HOSTS_ENV = "TRAINING_GYM_PROXY_AUTH_HOSTS"

# Modal edge auth only exists on Modal-served hostnames.
_MODAL_HOST_SUFFIXES = (".modal.run", ".modal.direct")

# Where plaintext is not an exposure: a loopback request never reaches a
# network. Also what the tests' local dashboard server speaks.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _extra_hosts() -> frozenset[str]:
    raw = os.environ.get(PROXY_AUTH_HOSTS_ENV, "")
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


def proxy_auth_url_allowed(url: str) -> bool:
    """Whether the workspace pair may be sent to ``url``.

    The pair is a workspace-wide credential and the dashboard URL is
    operator-supplied configuration, so it goes only where Modal edge auth
    can exist: an ``https`` origin on a Modal-served hostname, or on a host
    named in ``TRAINING_GYM_PROXY_AUTH_HOSTS``. Naming a host opts it into
    receiving the pair, not into sending it in the clear.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if not host:
        return False
    if parts.scheme != "https" and host not in _LOOPBACK_HOSTS:
        return False
    if host in _extra_hosts():
        return True
    return host.endswith(_MODAL_HOST_SUFFIXES)


def _configured_pair() -> tuple[str, str]:
    from modal_training_gym.common.config import load_proxy_auth

    load_proxy_auth()
    return (
        os.environ.get(PROXY_AUTH_KEY_ENV, "").strip(),
        os.environ.get(PROXY_AUTH_SECRET_ENV, "").strip(),
    )


def modal_proxy_auth_headers(url: str) -> dict[str, str]:
    """Proxy-auth headers for ``url``, from the environment or
    ``~/.training-gym.toml``.

    Containers only ever see the env vars, forwarded by
    :func:`modal_training_gym.common.proxy_auth_secrets`; the file is how a
    laptop persists the pair via ``training-gym setup``. A partial pair — or
    a URL that fails :func:`proxy_auth_url_allowed` — returns nothing.
    """
    key, secret = _configured_pair()
    if key and secret and proxy_auth_url_allowed(url):
        return {"Modal-Key": key, "Modal-Secret": secret}
    return {}


class _RefuseRedirects(HTTPRedirectHandler):
    """urllib replays the original headers — run token and proxy pair — at
    whatever origin a redirect names. The ingestion endpoints never redirect,
    so treat a 3xx as the misconfiguration it is."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


NO_REDIRECT_OPENER = build_opener(_RefuseRedirects)


_AUTH_WARN_INTERVAL_SECONDS = 60.0
_auth_warn_lock = threading.Lock()
# Per warning kind, so an auth warning cannot silence a redirect warning.
_next_warn: dict[str, float] = {}


def _warn_rate_limited(kind: str) -> bool:
    now = time.monotonic()
    with _auth_warn_lock:
        if now < _next_warn.get(kind, 0.0):
            return True
        _next_warn[kind] = now + _AUTH_WARN_INTERVAL_SECONDS
    return False


def _loggable(url: str) -> str:
    """``scheme://host/path`` — enough to identify the endpoint, without the
    userinfo or query string a hand-configured status URL might sign.

    ``urlsplit`` accepts a non-numeric port and ``SplitResult.port`` raises for
    it lazily, so the port has to be read inside the guard: this runs from the
    reporter's ``except HTTPError`` path, where an exception would unwind the
    worker thread and silently end reporting for the rest of the run.
    """
    try:
        parts = urlsplit(url)
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "<unparseable url>"
    if not parts.scheme and not parts.hostname:
        return "<unparseable url>"
    return f"{parts.scheme}://{parts.hostname or ''}{port}{parts.path}"


def warn_auth_rejected(status: int, url: str) -> None:
    """Rate-limited stderr warning for a 401/403 from the dashboard.

    Reports are fire-and-forget, so a bad run token — or a proxy-authed
    dashboard reached without the pair — would otherwise erase dashboard
    updates in silence. Never logs a credential.
    """
    if _warn_rate_limited("auth"):
        return
    key, secret = _configured_pair()
    if key and secret and not proxy_auth_url_allowed(url):
        hint = (
            "The configured Modal proxy-auth pair was withheld from this "
            f"non-Modal URL; set {PROXY_AUTH_HOSTS_ENV}=<host> to send it."
        )
    else:
        hint = (
            "Check the run's status token and, if the dashboard requires "
            "Modal proxy auth, MODAL_KEY/MODAL_SECRET."
        )
    print(
        f"training-gym reporter: dashboard rejected a report (HTTP {status} "
        f"from {_loggable(url)}). {hint} Reports are "
        "best-effort; training is unaffected.",
        file=sys.stderr,
    )


def warn_redirect_refused(status: int, url: str) -> None:
    """Rate-limited stderr warning for a 3xx from the dashboard.

    Redirects are refused (see :class:`_RefuseRedirects`), so a redirecting
    dashboard URL — e.g. one saved as ``http://`` — silently drops every
    report; that configuration error is worth one loud line.
    """
    if _warn_rate_limited("redirect"):
        return
    print(
        f"training-gym reporter: refusing a redirect (HTTP {status} from "
        f"{_loggable(url)}); reports are being dropped. Configure the final "
        "https dashboard URL. Training is unaffected.",
        file=sys.stderr,
    )
