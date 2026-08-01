"""User-local config persisted at ``~/.training-gym.toml``.

Populated by ``training-gym setup``; read by the slime launcher (and any other
caller) to look up where to POST phase reports and other client-side defaults.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any


CONFIG_PATH = Path.home() / ".training-gym.toml"
MODAL_CONFIG_PATH = Path(
    os.environ.get("MODAL_CONFIG_PATH") or os.path.expanduser("~/.modal.toml")
)


def load_config() -> dict[str, Any]:
    """Return the parsed ``~/.training-gym.toml``, or ``{}`` if missing."""
    if not CONFIG_PATH.is_file():
        return {}
    try:
        with CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _write_config(config: dict[str, Any]) -> None:
    """Write atomically, owner-only: the file can hold the proxy-auth pair.

    The temporary name is unique per call. A fixed one is not safe here:
    ``save_dashboard_url`` and ``save_proxy_auth`` can run concurrently (two
    parallel deploys, or ``setup`` beside one), and a shared temp path lets one
    writer rename the other's half-written file into place — or delete it out
    from under the other's ``os.replace``.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=CONFIG_PATH.parent, prefix=CONFIG_PATH.name + ".", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        # mkstemp already creates 0600; the mode matters because this file can
        # hold a workspace credential.
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_render(config))
        os.replace(tmp_path, CONFIG_PATH)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_dashboard_url(url: str) -> None:
    """Persist the deployed dashboard URL under ``[dashboard].url``."""
    config = load_config()
    dashboard = config.get("dashboard")
    if not isinstance(dashboard, dict):
        dashboard = {}
    dashboard["url"] = url
    config["dashboard"] = dashboard
    _write_config(config)


def get_dashboard_requires_proxy_auth() -> bool:
    """Return the persisted dashboard edge-auth choice (default ``False``)."""
    dashboard = load_config().get("dashboard")
    if isinstance(dashboard, dict):
        return dashboard.get("requires_proxy_auth") is True
    return False


def save_dashboard_requires_proxy_auth(value: bool) -> None:
    """Persist the dashboard edge-auth choice under
    ``[dashboard].requires_proxy_auth``."""
    config = load_config()
    dashboard = config.get("dashboard")
    if not isinstance(dashboard, dict):
        dashboard = {}
    dashboard["requires_proxy_auth"] = bool(value)
    config["dashboard"] = dashboard
    _write_config(config)


def get_dashboard_url() -> str | None:
    """Return the saved dashboard base URL, or ``None``."""
    dashboard = load_config().get("dashboard")
    if isinstance(dashboard, dict):
        url = dashboard.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


PROXY_AUTH_SECTION = "proxy_auth"


def get_proxy_auth() -> tuple[str, str]:
    """Return the ``(MODAL_KEY, MODAL_SECRET)`` pair saved under ``[proxy_auth]``.

    Returns empty strings for any value that is missing or blank.
    """
    section = load_config().get(PROXY_AUTH_SECTION)
    if isinstance(section, dict):
        key = str(section.get("key") or "").strip()
        secret = str(section.get("secret") or "").strip()
        return key, secret
    return "", ""


def save_proxy_auth(key: str, secret: str) -> None:
    """Persist the proxy-auth token pair under ``[proxy_auth]``."""
    config = load_config()
    config[PROXY_AUTH_SECTION] = {"key": key.strip(), "secret": secret.strip()}
    _write_config(config)


def load_proxy_auth() -> bool:
    """Populate ``MODAL_KEY`` / ``MODAL_SECRET`` from ``~/.training-gym.toml``.

    Dotenv-style: real environment variables always win and are never
    overwritten; only unset ones are filled in from the saved config. Returns
    ``True`` when both end up set in the environment.
    """
    have_key = bool(os.environ.get("MODAL_KEY", "").strip())
    have_secret = bool(os.environ.get("MODAL_SECRET", "").strip())
    if have_key and have_secret:
        return True

    key, secret = get_proxy_auth()
    if key and not have_key:
        os.environ["MODAL_KEY"] = key
    if secret and not have_secret:
        os.environ["MODAL_SECRET"] = secret
    return bool(
        os.environ.get("MODAL_KEY", "").strip()
        and os.environ.get("MODAL_SECRET", "").strip()
    )


def get_framework_status_url() -> str | None:
    """Return the saved framework-status endpoint URL, or ``None``."""
    base = get_dashboard_url()
    if not base:
        return None
    return base.rstrip("/") + "/api/framework-status"


def _render(config: dict[str, Any]) -> str:
    """Minimal TOML writer for the shapes we persist (string-valued tables)."""
    lines: list[str] = []
    for section, entries in config.items():
        if not isinstance(entries, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in entries.items():
            lines.append(f"{key} = {_format_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


# ── Modal credential resolution ──────────────────────────────────────────


def read_modal_toml_creds() -> tuple[str, str, str]:
    """Resolve ``(token_id, token_secret, profile_name)`` from ``~/.modal.toml``.

    Follows Modal's own profile-selection rules: the ``MODAL_PROFILE`` env
    var wins if set; otherwise the profile flagged ``active = true``;
    otherwise the ``[default]`` profile; otherwise the first profile in the
    file. Returns empty strings if no credentials can be found.
    """
    if not MODAL_CONFIG_PATH.is_file():
        return "", "", ""

    try:
        with MODAL_CONFIG_PATH.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return "", "", ""

    profiles = {
        name: section for name, section in data.items() if isinstance(section, dict)
    }
    if not profiles:
        return "", "", ""

    candidate_names: list[str] = []
    env_profile = os.environ.get("MODAL_PROFILE", "").strip()
    if env_profile:
        candidate_names.append(env_profile)
    candidate_names.extend(
        name for name, sec in profiles.items() if sec.get("active") is True
    )
    if "default" in profiles:
        candidate_names.append("default")
    candidate_names.extend(profiles.keys())

    seen: set[str] = set()
    for name in candidate_names:
        if name in seen or name not in profiles:
            continue
        seen.add(name)
        section = profiles[name]
        token_id = str(section.get("token_id") or "").strip()
        token_secret = str(section.get("token_secret") or "").strip()
        if token_id and token_secret:
            return token_id, token_secret, name
    return "", "", ""


def resolve_modal_creds() -> tuple[str, str, str]:
    """Resolve Modal credentials with a source label for logging.

    Order: ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET`` env vars → active
    profile in ``~/.modal.toml``.
    """
    env_id = (os.environ.get("MODAL_TOKEN_ID") or "").strip()
    env_secret = (os.environ.get("MODAL_TOKEN_SECRET") or "").strip()
    if env_id and env_secret:
        return env_id, env_secret, "environment"

    toml_id, toml_secret, profile_name = read_modal_toml_creds()
    if toml_id and toml_secret:
        return toml_id, toml_secret, f"~/.modal.toml profile [{profile_name}]"
    return "", "", ""
