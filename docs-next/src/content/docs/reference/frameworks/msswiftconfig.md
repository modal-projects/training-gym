---
title: MsSwiftConfig
description: API reference for MsSwiftConfig
---

# MsSwiftConfig

```python
from modal_training_gym.frameworks.ms_swift.config import MsSwiftConfig
```

Top-level wrapper that composes an ms-swift Megatron SFT run.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `'DatasetConfig | None'` |  | Dataset configuration with `prepare()` hook. Default `None`. |
| `model` | `'ModelConfiguration | None'` |  | Model identity and download hook. Default `None`. |
| `wandb` | `'WandbConfig | None'` |  | Weights & Biases logging config. Default `None`. |
| `framework_config` | `MsSwiftFrameworkConfig` |  | ms-swift Megatron training and infrastructure settings. Default `MsSwiftFrameworkConfig()`. |

## Methods

### `build_app(self, *, name: 'str | None' = None) -> "'App'"`

### `cli_args(self, *, output_dir: 'str') -> 'list[str]'`

**Source:** [`modal_training_gym/frameworks/ms_swift/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/ms_swift/config.py)
