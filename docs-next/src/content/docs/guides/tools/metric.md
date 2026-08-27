---
order: 2
---

# Metric logging

The [observability dashboard](https://gym.modal.dev/guides/observability-dashboard) captures the most important plots and metadata you'd care about during training. However, when you need access to everything logged by the [underlying framework](https://miles.radixark.com/docs), you can use our [Weights & Biases](https://wandb.ai) integration.

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
