---
title: Trackio integration
description: Stream training metrics to a Trackio Space or self-hosted dashboard.
sidebar:
  order: 3
---

[Trackio](https://huggingface.co/docs/trackio/index) is a lightweight,
W&B-compatible experiment tracker from Hugging Face. Training Gym installs it
in the training image and routes the framework's existing metric calls to it
when a recipe uses `TrackioConfig`.

## Hugging Face Space

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

## Self-hosted server

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
