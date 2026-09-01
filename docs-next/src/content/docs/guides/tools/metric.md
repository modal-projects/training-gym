---
order: 2
---

# Logging metrics

The [observability dashboard](https://gym.modal.dev/guides/dashboard) captures the most important plots and metadata you'd care about during training. However, when you need access to everything logged by the [underlying framework](https://miles.radixark.com/docs), you can use our [Weights & Biases](https://wandb.ai) integration.

First, you'll need to create a [Modal Secret](https://modal.com/docs/guide/secrets) with your API key:

```bash
modal secret create wandb-secret WANDB_API_KEY=<your-api-key>
```

Then, just pass it in your [training recipe](https://gym.modal.dev/guides/recipe):

```python
from modal_training_gym import Qwen3_5_4B, Qwen3_5_4b_Recipe, TrainConfig, WandbConfig

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4b_Recipe(
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

[Trackio](https://huggingface.co/docs/trackio) is a lightweight, W&B-compatible tracker from Hugging Face. Training Gym installs it in the training image and routes the framework's existing metric calls to it whenever a recipe uses `TrackioConfig`.

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
