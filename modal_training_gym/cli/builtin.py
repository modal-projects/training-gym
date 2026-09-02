"""Click wrappers for the CLI's existing top-level commands."""

from __future__ import annotations

import click

from .commands import _TrainingGymCommand


@click.command("setup", cls=_TrainingGymCommand)
@click.option(
    "--proxy-auth",
    is_flag=True,
    help="Require Modal proxy authentication for the dashboard.",
)
@click.option(
    "--no-proxy-auth",
    is_flag=True,
    help="Deploy the dashboard without Modal proxy authentication.",
)
def setup_command(proxy_auth: bool, no_proxy_auth: bool) -> None:
    """Deploy the dashboard."""
    if proxy_auth and no_proxy_auth:
        raise click.UsageError(
            "--proxy-auth and --no-proxy-auth cannot be used together."
        )

    from .setup import setup
    from modal_training_gym.common.config import get_dashboard_proxy_auth

    if not proxy_auth and not no_proxy_auth:
        if get_dashboard_proxy_auth() is True:
            raise click.UsageError(
                "The deployed dashboard uses proxy auth. "
                "Pass either --proxy-auth or --no-proxy-auth explicitly."
            )

    setup(require_proxy_auth=proxy_auth)


@click.command("open", cls=_TrainingGymCommand)
def open_command() -> None:
    """Open the dashboard."""
    from .setup import open_dashboard

    open_dashboard()


@click.command("set-proxy-auth", cls=_TrainingGymCommand)
def set_proxy_auth_command() -> None:
    """Store Modal proxy-auth credentials.

    Used by authenticated custom deployments when `MODAL_KEY` and
    `MODAL_SECRET` are not set in the environment. Re-running replaces the saved credentials.
    """
    from .setup import set_proxy_auth

    set_proxy_auth()


@click.command("set-password", cls=_TrainingGymCommand)
@click.option(
    "--password",
    default=None,
    metavar="PASSWORD",
    help="Set a password, prompt when omitted, or disable authentication with an empty value.",
)
def set_password_command(password: str | None) -> None:
    """Set or clear dashboard Basic Auth, then redeploy."""
    from .setup import set_password

    set_password(password=password)


@click.command("cleanup", cls=_TrainingGymCommand)
@click.option(
    "--older-than-days",
    type=int,
    default=7,
    show_default=True,
    metavar="DAYS",
    help="Delete failed or cancelled runs older than this many days.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without deleting.",
)
def cleanup_command(older_than_days: int, dry_run: bool) -> None:
    """Delete old failed or cancelled run metadata."""
    from .cleanup import cleanup

    cleanup(older_than_days=older_than_days, dry_run=dry_run)
