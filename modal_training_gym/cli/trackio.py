from __future__ import annotations

import webbrowser

import click

from .commands import _TrainingGymGroup


def setup_trackio() -> str:
    from modal_training_gym.common.trackio import TrackioConfig

    config = TrackioConfig.deploy_to_modal()
    web_url = config.dashboard_url or config.server_url
    print(f"\nTrackio dashboard deployed: {web_url}")
    return web_url


def open_trackio() -> str | None:
    from modal_training_gym.common.trackio import deployed_trackio_url

    web_url = deployed_trackio_url()
    if not web_url:
        print(
            "No deployed Trackio dashboard found. "
            "Run `training-gym trackio setup` to deploy it first."
        )
        return None

    print(f"Opening Trackio dashboard: {web_url}")
    webbrowser.open(web_url)
    return web_url


@click.group("trackio", cls=_TrainingGymGroup)
def trackio_group() -> None:
    """Deploy and open the Trackio dashboard."""


@trackio_group.command("setup")
def setup_command() -> None:
    """Deploy the Trackio dashboard."""
    setup_trackio()


@trackio_group.command("open")
def open_command() -> None:
    """Open the Trackio dashboard."""
    open_trackio()
