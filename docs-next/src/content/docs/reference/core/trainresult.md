---
title: TrainResult
description: API reference for TrainResult
---

```python
from modal_training_gym.common.train_result import TrainResult
```

One completed training run's checkpoint handle.

## Constructor

```python
TrainResult(app_name, framework, training_run_id, checkpoint_dir='', checkpoints_volume_name='', checkpoints_mount_path='', model_config=None, extra=<factory>)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` | *required* |  |
| `framework` | `Framework` | *required* |  |
| `training_run_id` | `str` | *required* |  |
| `checkpoint_dir` | `str` | `""` |  |
| `checkpoints_volume_name` | `str` | `""` |  |
| `checkpoints_mount_path` | `str` | `""` |  |
| `model_config` | `'ModelConfig | None'` | `None` |  |
| `extra` | `dict[str, Any]` | `<factory>` |  |

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` |  |  |
| `framework` | `Framework` |  |  |
| `training_run_id` | `str` |  |  |
| `checkpoint_dir` | `str` | `""` |  |
| `checkpoints_volume_name` | `str` | `""` |  |
| `checkpoints_mount_path` | `str` | `""` |  |
| `model_config` | `'ModelConfig | None'` | `None` |  |
| `extra` | `dict[str, Any]` | `{}` |  |

## Methods

### `dashboard_url(self) -> 'str'`

URL for browsing the checkpoints volume in the Modal dashboard.

### `from_training_run_id(training_run_id: 'str') -> "'TrainResult'"`

Load a completed run's result.

### `list_results() -> "list['TrainResult']"`

### `load(training_run_id: 'str') -> "'TrainResult'"`

### `save(self) -> 'None'`

Persist this result to the shared metadata volume.

### `volume(self) -> "'Volume'"`

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)

**Source:** [`modal_training_gym/common/train_result.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/train_result.py)
