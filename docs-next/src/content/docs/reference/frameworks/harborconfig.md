---
title: HarborConfig
description: API reference for HarborConfig
---

# HarborConfig

```python
from modal_training_gym.frameworks.harbor.config import HarborConfig
```

Top-level wrapper that composes a Harbor + Miles RLVR training run.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `'DatasetConfig | None'` |  | Dataset configuration with `prepare()` hook. Default `None`. |
| `model` | `'ModelConfiguration | None'` |  | Model identity and download hook. Default `None`. |
| `wandb` | `'WandbConfig | None'` |  | Weights & Biases logging config. Default `None`. |
| `framework_config` | `HarborFrameworkConfig` |  | Harbor + Miles training and infrastructure settings. Default `HarborFrameworkConfig()`. |

## Methods

### `build_app(self, *, name: 'str | None' = None) -> "'App'"`

## Related Tutorials

- [Qwen3-4B RL code-golf on MBPP with Harbor sandboxes](/tutorials/rl/harbor_code_golf/)

**Source:** [`modal_training_gym/frameworks/harbor/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/harbor/config.py)
