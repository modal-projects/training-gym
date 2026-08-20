"""Deploy the training-gym dashboard non-interactively (used by CI).

`modal deploy dashboards/app.py` always registers the ASGI app without proxy
auth, so the deploy goes through the CLI's `setup()`, which honours the
requested mode and provisions the `_training-gym-modal-creds` Secret the
dashboard uses to stream logs from training apps.

Usage:
    python scripts/deploy_dashboard.py --proxy-auth
    python scripts/deploy_dashboard.py --no-proxy-auth
"""

from __future__ import annotations

import argparse

from modal_training_gym.cli.setup import setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--proxy-auth",
        dest="proxy_auth",
        action="store_true",
        help="Require Modal proxy authentication for the dashboard.",
    )
    group.add_argument(
        "--no-proxy-auth",
        dest="proxy_auth",
        action="store_false",
        help="Deploy the dashboard without Modal proxy authentication.",
    )
    args = parser.parse_args()

    setup(require_proxy_auth=args.proxy_auth, interactive=False)


if __name__ == "__main__":
    main()
