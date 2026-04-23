---
title: MilesConfig
description: API reference for MilesConfig
---

# MilesConfig

```python
from modal_training_gym.frameworks.miles.config import MilesConfig
```

Top-level wrapper that composes a Miles RLVR training run.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `'DatasetConfig | None'` | `None` | Dataset configuration with `prepare()` hook. Default `None`. |
| `model` | `'ModelConfiguration | None'` | `None` | Model identity and download hook. Default `None`. |
| `wandb` | `'WandbConfig | None'` | `None` | Weights & Biases logging config. Default `None`. |
| `framework_config` | `MilesFrameworkConfig` | `None` | Miles-specific training and infrastructure settings. Default `MilesFrameworkConfig()`. |

## Methods

### `build_app(self, *, name: 'str | None' = None) -> "'App'"`

**Source:** [`modal_training_gym/frameworks/miles/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/miles/config.py)
