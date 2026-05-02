---
title: TrainResult
description: API reference for TrainResult
---

# TrainResult

```python
from modal_training_gym.common.train_result import TrainResult
```

One completed training run's checkpoint handle.

## Constructor

```python
TrainResult(app_name, framework, training_run_id, checkpoint_dir, model_config=None, checkpoints_volume_name='', checkpoints_mount_path='', iteration_prefix='', wandb_project='', wandb_entity='', wandb_training_run_id='', extra=<factory>)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` | *required* |  |
| `framework` | `Framework` | *required* |  |
| `training_run_id` | `str` | *required* |  |
| `checkpoint_dir` | `str` | *required* |  |
| `model_config` | `'ModelConfig | None'` | `None` |  |
| `checkpoints_volume_name` | `str` | `""` |  |
| `checkpoints_mount_path` | `str` | `""` |  |
| `iteration_prefix` | `str` | `""` |  |
| `wandb_project` | `str` | `""` |  |
| `wandb_entity` | `str` | `""` |  |
| `wandb_training_run_id` | `str` | `""` |  |
| `extra` | `dict[str, Any]` | `<factory>` |  |

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` |  |  |
| `framework` | `Framework` |  |  |
| `training_run_id` | `str` |  |  |
| `checkpoint_dir` | `str` |  |  |
| `model_config` | `'ModelConfig | None'` | `None` |  |
| `checkpoints_volume_name` | `str` | `""` |  |
| `checkpoints_mount_path` | `str` | `""` |  |
| `iteration_prefix` | `str` | `""` |  |
| `wandb_project` | `str` | `""` |  |
| `wandb_entity` | `str` | `""` |  |
| `wandb_training_run_id` | `str` | `""` |  |
| `extra` | `dict[str, Any]` | `{}` |  |

## Methods

### `dashboard_url(self) -> 'str'`

URL for browsing the checkpoints volume in the Modal dashboard.

### `latest_checkpoint_path(self) -> 'str'`

Absolute in-volume path of the latest checkpoint.

### `list_checkpoints(self) -> 'list[str]'`

Return per-iteration checkpoint directory names under

### `load(training_run_id: 'str') -> "'TrainResult'"`

Load a completed run's result from the shared store.

### `save(self) -> 'None'`

Persist this result. Currently saves to a Modal Dict but in the future, we will save to a database.

### `volume(self) -> "'Volume'"`

Return a handle to the checkpoints :class:`modal.Volume`.

### `wandb_metrics(self, keys: 'list[str] | None' = None, samples: 'int' = 500) -> 'list[dict[str, Any]]'`

Fetch training metrics from W&B.

### `wandb_summary(self) -> 'dict[str, Any]'`

Fetch the W&B run summary (final metric values).

### `wandb_url(self) -> 'str | None'`

Return the W&B run URL, or None if W&B info is not set.

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)

**Source:** [`modal_training_gym/common/train_result.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/train_result.py)
