---
title: TrainConfig
description: API reference for TrainConfig
---

# TrainConfig

```python
from modal_training_gym.common.train import TrainConfig
```

Compose dataset, model, and recipe into one training entrypoint.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `DatasetConfig` |  |  |
| `model` | `ModelConfig` |  |  |
| `recipe` | `modal_training_gym.train_recipes.base.BaseTrainRecipe` |  |  |
| `checkpoint` | `modal_training_gym.common.checkpoint.Checkpoint | None` | `None` |  |

## Methods

### `train(self, run_id: str | None = None) -> modal_training_gym.common.train_result.TrainResult`

Build the app, run training, and return the TrainResult.

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Competitive programming RL with Harbor USACO and sandboxed verification](/tutorials/rl/001_code_golf_sandboxes/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn_number_guessing/)

**Source:** [`modal_training_gym/common/train.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/train.py)
