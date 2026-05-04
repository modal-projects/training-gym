"""Thin HTTP client for the training-gym central server."""

from __future__ import annotations

from enum import Enum
from typing import Any

import requests

# TODO(joy): allow an env variable or a config file to override where the server is hosted
SERVER_URL = "https://gymserver.modal.dev"


def _json_safe(obj: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable form."""
    if isinstance(obj, Enum):
        return obj.value
    if callable(obj) and not isinstance(obj, (type, bool)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_json_safe(v) for v in obj)
    return obj


def _post(path: str, data: dict[str, Any]) -> None:
    resp = requests.post(f"{SERVER_URL}{path}", json=_json_safe(data))
    resp.raise_for_status()


def _get(path: str) -> Any:
    resp = requests.get(f"{SERVER_URL}{path}")
    resp.raise_for_status()
    return resp.json()
