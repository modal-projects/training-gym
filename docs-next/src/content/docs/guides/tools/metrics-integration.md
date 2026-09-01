---
order: 2
---

# Metrics integration

Every training recipe can stream reward, KL, response lengths, learning rate,
and the rest of the framework's metrics to either [Weights &
Biases](https://wandb.ai) or
[Trackio](https://huggingface.co/docs/trackio/index). Logging is opt-in per
run: pass a `WandbConfig` or `TrackioConfig` to the recipe's `metrics` field.

The [observability dashboard](/guides/tools/observability-dashboard) records
the selected provider and links each run to its metrics dashboard.

## Weights & Biases

### Prerequisites

Training containers read the W&B API key from a Modal Secret named
`wandb-secret` with a `WANDB_API_KEY` entry. Create it once per workspace at
[modal.com/secrets](https://modal.com/secrets) or from the CLI:

```bash
modal secret create wandb-secret WANDB_API_KEY=<your-api-key>
```

To use a differently named secret, set
`WandbConfig(modal_wandb_secret_name="my-secret")`.

### Enable W&B logging

Pass `metrics=WandbConfig(project="...")` to any training recipe:

```python
from modal_training_gym import Qwen3_5_4b_Recipe, TrainConfig, WandbConfig
from modal_training_gym.common.models import Qwen3_5_4B

training_run = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4b_Recipe(
        num_rollout=40,
        metrics=WandbConfig(
            project="my-rl-project",
            group="lr-sweep",
        ),
    ),
)
training_run.train()
```

The most commonly used fields are:

| Field | What it does |
|---|---|
| `project` | W&B project the run logs to. |
| `entity` | W&B team or user. The API key's default entity is used when omitted. |
| `group` | Group tag for related runs, such as variants in a sweep. |
| `disable_random_suffix` | Preserves stable run names when enabled. Defaults to `True`. |
| `modal_wandb_secret_name` | Modal Secret containing `WANDB_API_KEY`. Defaults to `"wandb-secret"`. |

See the [`WandbConfig` reference](/reference/core/wandbconfig) for all fields.

### Preflight and run identity

Before GPU work begins, the launcher logs in, resolves the entity, and verifies
that the key can write to the configured project by creating and deleting a
probe run. A missing, expired, or unauthorized key therefore fails before the
cluster loads the model.

The Training Gym run ID is also the W&B run ID. If a run is preempted and
retried, attempt *N* logs to `<run-id>-aN` with resume enabled. When launching
variants with `TrainingGroup`, use a shared `group` so W&B can overlay their
curves. Sweep grids can override fields such as `recipe.metrics.group`.

## Trackio

Trackio is a lightweight, W&B-compatible experiment tracker from Hugging Face.
Training Gym installs it in the training image and routes the framework's
existing metric calls to it when a recipe uses `TrackioConfig`.

Trackio can use a server deployed on Modal, a Hugging Face Space, or another
self-hosted server.

### Deploy on Modal

Deploy a persistent Trackio server and use the returned configuration:

```python
metrics = TrackioConfig.deploy_to_modal(project="my-rl-project")
```

The first call creates a Modal app, a persistent Volume for Trackio data, and
an auto-managed Modal Secret for the write token. Later calls reuse them. Writes
require the token, and Training Gym does not include it in dashboard links.

Reads are open unless a dashboard password is set with
[`training-gym set-password`](/guides/tools/observability-dashboard), which puts
the Trackio dashboard behind the same HTTP Basic Auth as the observability
dashboard. Training containers keep ingesting with the write token. The
password is read at container startup, so run `deploy_to_modal()` again after
changing it.

Pass `metrics` to the recipe. Use `app_name`, `volume_name`, or
`modal_secret_name` when separate Trackio deployments are required.

### Host on Hugging Face Spaces

Create a Modal Secret named `huggingface-secret` with an `HF_TOKEN` that can
create or write to the Space:

```bash
modal secret create huggingface-secret HF_TOKEN=<your-token>
```

Then pass a complete `owner/name` Space ID:

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
`bucket_id` only when a specific Bucket is required.

### Use another self-hosted server

Configure the server URL and put its write token in a Modal Secret as
`TRACKIO_WRITE_TOKEN`:

```python
TrackioConfig(
    project="my-rl-project",
    server_url="https://metrics.example.com",
    modal_secret_name="trackio-secret",
)
```

Training Gym strips URL credentials and sensitive query parameters from
dashboard metadata. Set `dashboard_url` when the browser-facing URL differs
from the ingestion URL.

See the [`TrackioConfig` reference](/reference/core/trackioconfig) for all
fields and the Trackio documentation for
[remote logging](https://huggingface.co/docs/trackio/track#remote-logging-hugging-face-space-or-self-hosted-server).

## Disable metrics logging

Omit `metrics` from the recipe to disable external metrics logging. No provider
secret or preflight is required, and the observability dashboard still
captures the run's Training Gym metadata.
