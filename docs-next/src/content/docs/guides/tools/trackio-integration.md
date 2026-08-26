---
order: 3
---

# Trackio integration

[Trackio](https://huggingface.co/docs/trackio/index) is a lightweight,
W&B-compatible experiment tracker from Hugging Face. Training Gym installs it
in the training image and routes the framework's existing metric calls to it
when a recipe uses `TrackioConfig`.

Trackio can be hosted on Modal or on [Hugging Face
Spaces](https://huggingface.co/docs/hub/spaces). Both modes produce dashboard
links for Training Gym runs.

## Modal

Deploy a persistent Trackio server and use the returned configuration:

```python
metrics = TrackioConfig.deploy_to_modal(project="my-rl-project")
```

The first call creates a Modal app, a persistent Volume for Trackio data, and
an auto-managed Modal Secret for the write token. Later calls reuse them. The
dashboard is publicly readable; writes require the token, and Training Gym
does not include it in dashboard links.

Pass `metrics` to the recipe as shown below. Use `app_name`, `volume_name`, or
`modal_secret_name` when you need separate Trackio deployments.

## Hugging Face Spaces

Create a Modal Secret named `huggingface-secret` with an `HF_TOKEN` that can
create or write to the Space:

```bash
modal secret create huggingface-secret HF_TOKEN=<your-token>
```

Then pass a complete `owner/name` Space ID so the observability dashboard can
link directly to it:

```python
from modal_training_gym import Qwen3_5_4b_Recipe, TrackioConfig, TrainConfig
from modal_training_gym.common.models import Qwen3_5_4B

training_run = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4b_Recipe(
        metrics=TrackioConfig(
            project="my-rl-project",
            group="baseline",
            space_id="my-org/training-metrics",
        ),
    ),
)
training_run.train()
```

Trackio creates or reuses the Space and its default Hugging Face Bucket. Set
`bucket_id` only when you need to select a specific Bucket.

## Other self-hosted servers

For a self-hosted Trackio server, configure its base URL and put the write
token in a Modal Secret as `TRACKIO_WRITE_TOKEN`:

```python
TrackioConfig(
    project="my-rl-project",
    server_url="https://metrics.example.com",
    modal_secret_name="trackio-secret",
)
```

Training Gym strips URL credentials and query parameters from dashboard
metadata. Use `dashboard_url` when the browser-facing dashboard URL differs
from the ingestion URL.

See the [`TrackioConfig` reference](/reference/core/trackioconfig/) for all
fields and the Trackio documentation for
[remote logging](https://huggingface.co/docs/trackio/track#remote-logging-hugging-face-space-or-self-hosted-server).
