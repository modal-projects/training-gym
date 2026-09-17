---
order: 2
---

# Logging metrics

The [observability dashboard](https://gym.modal.dev/guides/dashboard) captures the most important plots and metadata you'd care about during training. However, when you need access to everything logged by the [underlying framework](https://miles.radixark.com/docs), you can use our [Weights & Biases](https://wandb.ai) or [Trackio](https://huggingface.co/docs/trackio) integrations.

## Weights & Biases

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

Trackio is self-hostable, so you can deploy it on Modal, via a Hugging Face Space, or locally.

### Deploy on Modal

Host a Trackio server on Modal:

```bash
training-gym trackio setup
```

Then open it any time with:

```bash
training-gym trackio open
```

You can also deploy it from the SDK:

```python
from modal_training_gym import TrackioConfig

metrics = TrackioConfig.deploy_to_modal(project="my-rl-project")
```

Just like the [main dashboard](https://gym.modal.dev/guides/dashboard), the Trackio dashboard is unauthenticated unless you set a password:

```bash
training-gym set-password
```

### Deploy on a Hugging Face Space

```python
from modal_training_gym import Qwen3_5_4B, Qwen3_5_4B_Recipe, TrainConfig, TrackioConfig

config = TrainConfig(
    model=Qwen3_5_4B(),
    dataset=my_dataset,
    recipe=Qwen3_5_4B_Recipe(
        # ...
        metrics=TrackioConfig(
            project="my-rl-project",
            space_id="my-org/training-metrics",
            bucket_id="my-org/training-metrics",  # optional
        ),
    ),
)
```

### Locally-hosted


You'll need to create a [Modal Secret](https://modal.com/docs/guide/secrets) with your API key:

```bash
modal secret create trackio-write-token TRACKIO_WRITE_TOKEN=<your-api-key>
```

Then, it's as easy as:

```python
metrics = TrackioConfig(
    project="my-rl-project",
    server_url="https://trackio.example.com",
    modal_secret_name="trackio-write-token",
)
```

See the [reference page](https://gym.modal.dev/reference/trackioconfig) for all parameters.
