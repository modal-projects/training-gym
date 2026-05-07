---
title: DeploymentConfig
description: API reference for DeploymentConfig
---

# DeploymentConfig

```python
from modal_training_gym.common.deployment import DeploymentConfig
```

Deploy a model behind a serving engine.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `ModelConfig` |  |  |
| `checkpoint` | `modal_training_gym.common.checkpoint.Checkpoint | None` | `None` |  |
| `recipe` | `VllmRecipe | SglangRecipe | None` | `None` |  |
| `app_name` | `str | None` | `None` |  |
| `served_model_name` | `str | None` | `None` |  |

## Methods

### `serve(self) -> "'ModelDeployment'"`

Build, deploy, and return a ``ModelDeployment`` handle.

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Competitive programming RL with Harbor USACO and sandboxed verification](/tutorials/rl/001_code_golf_sandboxes/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn_number_guessing/)

**Source:** [`modal_training_gym/common/deployment.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/deployment.py)
