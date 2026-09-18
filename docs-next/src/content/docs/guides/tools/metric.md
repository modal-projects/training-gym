---
order: 2
---

# Logging metrics

The [observability dashboard](https://gym.modal.dev/guides/dashboard) captures the most important plots and metadata you'd care about during training, and its **Metrics** tab charts every scalar the [underlying framework](https://miles.radixark.com/docs) logs. When you also want those numbers in an external tracker, use our [Weights & Biases](https://wandb.ai) or [Trackio](#trackio) integration; the dashboard keeps mirroring them either way.

## Dashboard only

The frameworks log through the `wandb` API. With `DashboardMetricConfig`, Training Gym swaps in a W&B-shaped shim inside the training container so those calls land in the dashboard instead. No account, API key, or extra server is involved:

```python
from modal_training_gym import DashboardMetricConfig, Qwen3_5_4B, Qwen3_5_4B_Recipe, TrainConfig

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4B_Recipe(
        # ...
        metrics=DashboardMetricConfig(project="my-rl-project"),
    ),
)

run = config.launch()
```

Open the run in the dashboard and switch to the Metrics tab. Keys are grouped by prefix the way W&B groups panels (`train/`, `rollout/`, `perf/`, ...), the search box narrows every group at once, and brushing one chart zooms all of them to the same step range. Charts refresh every few seconds while the run trains and the full history stays available after it finishes.

Only finite scalars are mirrored: nested dicts are flattened to `a/b`, while images, tables, histograms, strings, and booleans are dropped. Points are batched in the container and persisted to the metadata Volume in chunks, so logging every step costs nothing noticeable.

## Weights & Biases

When you need everything W&B offers (media, tables, cross-project reports), pass a `WandbConfig`. The framework logs to W&B as usual and, by default, the same scalars are mirrored to the dashboard's Metrics tab. Set `mirror_to_dashboard=False` to turn the mirror off.

First, you'll need to create a [Modal Secret](https://modal.com/docs/guide/secrets) with your API key:

```bash
modal secret create wandb-secret WANDB_API_KEY=<your-api-key>
```

Then, just pass it in your [training recipe](https://gym.modal.dev/guides/recipe):

```python
from modal_training_gym import Qwen3_5_4B, Qwen3_5_4B_Recipe, TrainConfig, WandbConfig

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4B_Recipe(
        # ...
        metrics=WandbConfig(
            project="my-rl-project",
            group="lr-sweep",  # optional: organize related runs
        ),
    ),
)

run = config.launch()
```

See the [reference page](https://gym.modal.dev/reference/wandbconfig) for the full list of parameters.

When launching a [hyperparameter sweep](https://gym.modal.dev/tutorials/param_sweep), the `group` parameter is especially useful to overlay multiple runs' reward curves.

## Trackio

[Trackio](https://huggingface.co/docs/trackio) is a lightweight, W&B-compatible tracker from Hugging Face. Training Gym installs it in the training image and routes the framework's existing metric calls to it whenever a recipe uses `TrackioConfig`. Like `WandbConfig`, it mirrors scalars to the dashboard's Metrics tab unless you pass `mirror_to_dashboard=False`.

There are two ways to visualize your metrics if you are using Trackio: 1) deploy on Modal, and 2) deploy on a Hugging Face Space.

### Deploy on Modal

You can host a Trackio server on Modal:

```python
from modal_training_gym import TrackioConfig

metrics = TrackioConfig.deploy_to_modal(project="my-rl-project")
```

The first call creates a Modal app, a Volume for Trackio's data, and a Secret holding a write token; later calls reuse them. Pass `metrics` to your recipe exactly like `WandbConfig`.

Reads to Trackio are open unless you've set a [dashboard password](https://gym.modal.dev/guides/dashboard) with `training-gym set-password`:

```bash
training-gym set-password
```

Training containers keep logging either way, since they authenticate with the write token instead. The password is read at container startup, so rerun `deploy_to_modal()` after changing it.

### Deploy on a Hugging Face Space

Point `TrackioConfig` at a Hugging Face Space with `space_id="my-org/training-metrics"`, or at your own server with `server_url` plus a Modal Secret holding `TRACKIO_WRITE_TOKEN`. See the [reference page](https://gym.modal.dev/reference/core/trackioconfig) for all parameters.
