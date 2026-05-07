---
title: ModelDeployment
description: API reference for ModelDeployment
---

# ModelDeployment

```python
from modal_training_gym.common.deployment import ModelDeployment
```

A deployed model endpoint.

## Constructor

```python
ModelDeployment(**data)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `deployment_config` | `DeploymentConfig` |  |  |
| `modal_app_id` | `str` | `""` |  |
| `url` | `str` |  |  |

## Methods

### `generate(self, prompt: 'str', **kwargs) -> 'str'`

### `list_deployments() -> "list['ModelDeployment']"`

### `save(self) -> 'None'`

### `wait_until_ready(self, timeout: 'int' = 600) -> 'None'`

**Source:** [`modal_training_gym/common/deployment.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/deployment.py)
