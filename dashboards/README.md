# Dashboard

Self-hosted observability dashboard for Training Gym. Aggregates training
runs, deployments, and eval results from the `training-gym-metadata` Modal
Volume into a single Svelte SPA served by a Modal ASGI endpoint.

Deploy your own copy:

```bash
modal deploy dashboards/app.py
```

Modal prints the URL where the dashboard is served.

## Pointing the metric links at another tracker

Each run links out to its metric curves on wandb.ai. If your training
containers log somewhere else — a self-hosted W&B server, or a wandb-compatible
shim in front of another backend — create an optional `training-gym-tracker`
Secret and redeploy:

```bash
modal secret create training-gym-tracker \
  TRAINING_GYM_TRACKER_LABEL=metrics \
  TRAINING_GYM_TRACKER_RUN_URL_TEMPLATE='https://metrics.example.com/?project={project}&run={run_id}' \
  TRAINING_GYM_TRACKER_PROJECT_URL_TEMPLATE='https://metrics.example.com/?project={project}'
```

Templates substitute `{entity}`, `{project}`, `{group}` and `{run_id}` from the
run record, percent-encoded, and render only when every field they reference is
non-empty — so the project template is the fallback for a run whose id isn't
known yet, and a backend with no notion of an entity can just leave `{entity}`
out. Setting either template switches both away from the wandb.ai defaults.

Links are derived when the dashboard reads a run, not when it writes one, so
this also relabels and relinks runs that were recorded before you set it. See
`modal_training_gym/common/tracker.py`.
