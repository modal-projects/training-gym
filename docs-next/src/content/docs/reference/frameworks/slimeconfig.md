---
title: SlimeConfig
description: API reference for SlimeConfig
---

# SlimeConfig

```python
from modal_training_gym.frameworks.slime.config import SlimeConfig
```

Base SLIME training configuration.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `environment` | `dict` | `{'PYTHONPATH': '/root/Megatron-LM/', 'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '1'}` |  |
| `async_mode` | `bool` | `False` |  |
| `slime_model_script` | `str` | `""` |  |
| `dataset` | `DatasetConfig | None` | `None` |  |
| `model` | `ModelConfiguration | None` | `None` |  |
| `wandb` | `WandbConfig | None` | `None` |  |
| `app_tags` | `dict` | `{}` |  |
| `modal` | `ModalConfig | None` | `None` |  |

## Methods

### `build_app(self, *, name: str | None = None, modal: modal_training_gym.frameworks.slime.config.ModalConfig | None = None) -> 'App'`

### `cli_args(self) -> list[str]`

SLIME CLI arguments derived from this config.

### `prepare_data(self) -> None`

Materialize the training data.

### `to_dataset_config(self) -> 'DatasetConfig'`

Extract dataset-related SLIME flags back into a `DatasetConfig`.

### `to_model(self) -> 'ModelConfiguration'`

Extract model-related SLIME flags back into a `ModelConfiguration`.

### `to_wandb_config(self) -> 'WandbConfig'`

Extract wandb-related SLIME flags back into a `WandbConfig`.

### `total_nodes(self) -> int`

Total Modal cluster nodes required by this config.

**Source:** [`modal_training_gym/frameworks/slime/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/slime/config.py)
