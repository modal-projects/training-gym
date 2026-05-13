"""Deploy the training-gym dashboard to Modal.

Usage (Python):
    import modal_training_gym
    modal_training_gym.setup()

Usage (CLI):
    training-gym setup
"""

from __future__ import annotations


def setup() -> str:
    """Deploy the training-gym dashboard and return its URL."""
    import modal

    from modal_training_gym._dashboard import app

    with modal.enable_output():
        deployed = app.deploy()

    url = deployed.get_dashboard_url()
    print(f"\nDashboard deployed: {url}")
    return url
