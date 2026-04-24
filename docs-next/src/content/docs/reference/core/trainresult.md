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
TrainResult(app_name, framework, run_id, checkpoint_dir, base_model, checkpoints_volume_name, checkpoints_mount_path, iteration_prefix='', wandb_project='', wandb_entity='', wandb_run_id='', extra=<factory>)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` | *required* |  |
| `framework` | `str` | *required* |  |
| `run_id` | `str` | *required* |  |
| `checkpoint_dir` | `str` | *required* |  |
| `base_model` | `str` | *required* |  |
| `checkpoints_volume_name` | `str` | *required* |  |
| `checkpoints_mount_path` | `str` | *required* |  |
| `iteration_prefix` | `str` | `""` |  |
| `wandb_project` | `str` | `""` |  |
| `wandb_entity` | `str` | `""` |  |
| `wandb_run_id` | `str` | `""` |  |
| `extra` | `dict[str, Any]` | `<factory>` |  |

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `app_name` | `str` |  |  |
| `framework` | `str` |  |  |
| `run_id` | `str` |  |  |
| `checkpoint_dir` | `str` |  |  |
| `base_model` | `str` |  |  |
| `checkpoints_volume_name` | `str` |  |  |
| `checkpoints_mount_path` | `str` |  |  |
| `iteration_prefix` | `str` | `""` |  |
| `wandb_project` | `str` | `""` |  |
| `wandb_entity` | `str` | `""` |  |
| `wandb_run_id` | `str` | `""` |  |
| `extra` | `dict[str, Any]` | `{}` |  |

## Methods

### `build_serve_app(self, *, served_model_name: 'str | None' = None, checkpoint_path: 'str | None' = None, **vllm_kwargs: 'Any') -> "'App'"`

Build a vLLM serving app pointing at a trained checkpoint.

### `dashboard_url(self) -> 'str'`

URL for browsing the checkpoints volume in the Modal dashboard.

### `latest_checkpoint_path(self) -> 'str'`

Absolute in-volume path of the latest checkpoint.

### `list_checkpoints(self) -> 'list[str]'`

Return per-iteration checkpoint directory names under

### `list_runs(app_name: 'str') -> 'list[str]'`

Return all ``run_id``s saved for ``app_name``, sorted oldest

### `load(app_name: 'str', run_id: 'str | None' = None) -> "'TrainResult'"`

Load a completed run's result from the shared store.

### `save(self) -> 'None'`

Persist this result to the shared :class:`modal.Dict`.

### `volume(self) -> "'Volume'"`

Return a handle to the checkpoints :class:`modal.Volume`.

### `wandb_metrics(self, keys: 'list[str] | None' = None, samples: 'int' = 500) -> 'list[dict[str, Any]]'`

Fetch training metrics from W&B.

### `wandb_summary(self) -> 'dict[str, Any]'`

Fetch the W&B run summary (final metric values).

### `wandb_url(self) -> 'str | None'`

Return the W&B run URL, or None if W&B info is not set.

**Source:** [`modal_training_gym/common/train_result.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/modal_training_gym/common/train_result.py)
