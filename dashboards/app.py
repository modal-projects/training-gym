"""Training-gym dashboard: Svelte SPA + JSON data APIs.

Deploy with:

    uv run modal deploy dashboards/app.py

Or programmatically:

    import modal_training_gym
    modal_training_gym.setup()
"""

from modal_training_gym._dashboard import app, fastapi_app  # noqa: F401
