"""Trackio run configuration."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode


TRACKIO_APP_NAME = "training-gym-dashboard"
TRACKIO_WEB_FUNCTION = "trackio_dashboard"
TRACKIO_VOLUME_NAME = "training-gym-trackio"
TRACKIO_SECRET_NAME = "_training-gym-trackio"
TRACKIO_MOUNT_PATH = "/trackio"
TRACKIO_DATA_PATH = f"{TRACKIO_MOUNT_PATH}/data"


def deployed_trackio_url() -> str | None:
    """Return the permanent Trackio endpoint URL when it is deployed."""
    import modal

    try:
        fn = modal.Function.from_name(TRACKIO_APP_NAME, TRACKIO_WEB_FUNCTION)
        fn.hydrate()
        return fn.get_web_url()
    except Exception:
        return None


def trackio_project_url(base_url: str | None, project: str) -> str | None:
    """Build a dashboard URL filtered to ``project``."""
    if not base_url:
        return None
    query = urlencode({"project": project}) if project else ""
    return f"{base_url.rstrip('/')}/?{query}" if query else base_url.rstrip("/") + "/"


def trackio_run_name(config: TrackioConfig, training_run_id: str) -> str:
    """Return the display name for a stable Training Gym-backed Trackio run."""
    return config.run_name or training_run_id


@dataclass
class TrackioConfig:
    """Configuration for the permanent Trackio server used by Slime.

    Parameters
    ----------
    project:
        Trackio project name.
    run_name:
        Optional display name. The Training Gym run id is used when omitted.
        This does not affect the stable Trackio run id used to combine metrics
        from distributed workers and retries.
    """

    project: str
    run_name: str = ""
