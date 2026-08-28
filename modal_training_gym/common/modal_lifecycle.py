"""Modal app lifecycle helpers."""

from __future__ import annotations

import re

_LIVE_APP_STATES: frozenset[int] | None = None
_TERMINAL_APP_STATES: frozenset[int] | None = None
_APP_ID_PATTERN = re.compile(r"ap-[A-Za-z0-9]+")


def live_app_states() -> frozenset[int]:
    global _LIVE_APP_STATES
    if _LIVE_APP_STATES is None:
        from modal_proto import api_pb2

        _LIVE_APP_STATES = frozenset(
            {
                api_pb2.APP_STATE_EPHEMERAL,
                api_pb2.APP_STATE_DETACHED,
                api_pb2.APP_STATE_DETACHED_DISCONNECTED,
                api_pb2.APP_STATE_INITIALIZING,
                api_pb2.APP_STATE_DEPLOYED,
                api_pb2.APP_STATE_DERIVED,
                # STOPPING is not terminal: tasks can still be winding down.
                api_pb2.APP_STATE_STOPPING,
            }
        )
    return _LIVE_APP_STATES


def terminal_app_states() -> frozenset[int]:
    """Return states that prove an app can no longer be running tasks."""

    global _TERMINAL_APP_STATES
    if _TERMINAL_APP_STATES is None:
        from modal_proto import api_pb2

        _TERMINAL_APP_STATES = frozenset(
            {
                api_pb2.APP_STATE_STOPPED,
                api_pb2.APP_STATE_DISABLED,
            }
        )
    return _TERMINAL_APP_STATES


def get_app_lifecycle_state(app_id: str) -> int | None:
    """Return Modal ``app_state`` enum value, or ``None`` on error."""
    if not app_id:
        return None
    try:
        from modal._utils.async_utils import synchronizer
        from modal.client import _Client
        from modal_proto import api_pb2

        async def _lifecycle_state() -> int:
            client = await _Client.from_env()
            resp = await client.stub.AppGetLifecycle(
                api_pb2.AppGetLifecycleRequest(app_id=app_id)
            )
            return resp.lifecycle.app_state

        return synchronizer.create_blocking(_lifecycle_state)()
    except Exception:
        return None


def app_live_status(app_id: str) -> bool | None:
    """Return whether the Modal app is live, or ``None`` if state is unknown."""
    state = get_app_lifecycle_state(app_id)
    if state is None:
        return None
    if state in live_app_states():
        return True
    if state in terminal_app_states():
        return False
    # APP_STATE_UNSPECIFIED and future enum values are unknown evidence, not
    # proof of teardown.
    return None


def app_is_live(app_id: str) -> bool:
    """Best-effort check whether a Modal app is still alive on the server."""
    status = app_live_status(app_id)
    if status is None:
        return False
    return status


def stop_app(app_id: str) -> None:
    """Stop a detached Modal app (best effort). Never raises."""
    if not app_id:
        return
    try:
        request_stop_app(app_id)
    except Exception as exc:  # noqa: BLE001 — auto-stop is best-effort
        print(f"WARNING: could not auto-stop app {app_id}: {exc}")


def request_stop_app(app_id: str) -> None:
    """Request an app stop and surface transport/server failures to the caller.

    This strict sibling of :func:`stop_app` is for watchdogs and other controls
    that must not confuse "the request was attempted" with "the request reached
    Modal".  Terminal confirmation remains a separate lifecycle-state check.
    """

    if not isinstance(app_id, str) or _APP_ID_PATTERN.fullmatch(app_id) is None:
        raise ValueError("an exact Modal app ID is required")
    from modal._utils.async_utils import synchronizer
    from modal.client import _Client
    from modal_proto import api_pb2

    async def _request_stop() -> None:
        client = await _Client.from_env()
        await client.stub.AppStop(
            api_pb2.AppStopRequest(
                app_id=app_id,
                source=api_pb2.APP_STOP_SOURCE_PYTHON_CLIENT,
            )
        )

    synchronizer.create_blocking(_request_stop)()
