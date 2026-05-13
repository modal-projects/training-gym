---
title: ModelDeployment
description: API reference for ModelDeployment
---

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
| `deployment_id` | `str` |  |  |
| `deployment_config` | `DeploymentConfig` |  |  |
| `modal_app_id` | `str` |  |  |
| `modal_app_url` | `str` |  |  |
| `url` | `str` |  |  |
| `status` | `str` |  |  |

## Methods

### `generate(self, prompt: 'str', ensure_ready: 'bool' = True, **kwargs) -> 'str'`

### `list_deployments() -> "dict[str, 'ModelDeployment']"`

### `save(self) -> 'None'`

### `wait_until_ready(self, timeout: 'int' = 600) -> 'None'`

## Related Tutorials

- [Code RL with Harbor hello-world and sandboxed verification](/tutorials/rl/001_sandboxes/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn/)

**Source:** [`modal_training_gym/common/deployment.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/deployment.py)
