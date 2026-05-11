# Dashboard

Self-hosted observability dashboard for Training Gym. Aggregates training
runs, deployments, and eval results from the `training-gym-metadata` Modal
Volume into a single Svelte SPA served by a Modal ASGI endpoint.

Deploy your own copy:

```bash
modal deploy dashboards/app.py
```

Modal prints the URL where the dashboard is served.
