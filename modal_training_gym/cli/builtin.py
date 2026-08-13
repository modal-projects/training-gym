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
    """Deploy the training-gym dashboard to Modal."""
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
    """Open the deployed dashboard in your browser."""
    from .setup import open_dashboard

    open_dashboard()


@click.command("set-proxy-auth", cls=_TrainingGymCommand)
def set_proxy_auth_command() -> None:
    """Set/replace the Modal proxy-auth credentials for authenticated served endpoints.

    You only need this after deploying a model with `unauthenticated=False`.
    Re-run this command to change the saved credentials.
    """
    from .setup import set_proxy_auth

    set_proxy_auth()


@click.command("set-password", cls=_TrainingGymCommand)
@click.option(
    "--password",
    default=None,
    metavar="PASSWORD",
    help="Password to set (prompted securely if omitted; empty disables auth).",
)
def set_password_command(password: str | None) -> None:
    """Set or clear the dashboard password (Basic Auth) and redeploy."""
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
    """Delete metadata for old failed or cancelled runs."""
    from .cleanup import cleanup

    cleanup(older_than_days=older_than_days, dry_run=dry_run)
