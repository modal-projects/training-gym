"""The launch-side helpers that have no stitch cookbook equivalent.

Everything the cookbook already does — config resolution, the train command, Ray
bring-up, the Megatron patches — is imported from it in the trainer. What is left
is reaching a pool's Flash gateway, part of it from the *client*, where the
cookbook is not importable.
"""

from __future__ import annotations

import time
import urllib.request
from typing import Any

import modal


def flash_gateway_url(server_cls: Any) -> str:
    """The rollout pool's Flash gateway URL, from the app's own ``Server`` handle.

    Called with the class object the app was built with: that handle is hydrated
    for the container's own app, so it resolves in an ephemeral run — unlike
    ``Cls.from_name``/``ModalFlashPool``, which need a deployed app.
    """
    urls = server_cls._experimental_get_flash_urls()
    if not urls:
        raise RuntimeError(f"no Flash gateway URL for {server_cls}")
    return str(urls[0]).rstrip("/")


def deployed_gateway_url(app_name: str, cls_name: str = "Server") -> str:
    """The pool's Flash gateway URL looked up by app name — deployed pools only."""
    try:
        cls = modal.Cls.from_name(app_name, cls_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"cannot resolve {app_name}.{cls_name} — is the pool deployed?"
        ) from exc
    return flash_gateway_url(cls)


def await_gateway_ready(
    gateway: str,
    *,
    timeout_seconds: float = 20 * 60,
    interval_seconds: float = 15.0,
) -> bool:
    """Block until the pool's gateway answers ``/health`` 200.

    Flash holds requests through a cold-starting pool, so this only matters for
    the trainer's first rollout meeting engines that are still loading. On
    timeout it warns and returns ``False``; the caller proceeds because the
    trainer's rollout requests retry.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{gateway}/health", timeout=10).close()
            return True
        except Exception:  # noqa: BLE001
            time.sleep(interval_seconds)
    print(f"WARNING: pool at {gateway} not ready after {timeout_seconds:.0f}s")
    return False
