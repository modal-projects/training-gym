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
| `deployment_id` | `str` |  |  |
| `deployment_config` | `DeploymentConfig` |  |  |
| `modal_app_id` | `str` |  |  |
| `url` | `str` |  |  |

## Methods

### `generate(self, prompt: 'str', ensure_ready: 'bool' = True, **kwargs) -> 'str'`

### `list_deployments() -> "dict[str, 'ModelDeployment']"`

### `save(self) -> 'None'`

### `wait_until_ready(self, timeout: 'int' = 600) -> 'None'`

## Related Tutorials

- [Competitive programming RL with Harbor USACO and sandboxed verification](/tutorials/rl/001_code_golf_sandboxes/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn_number_guessing/)

**Source:** [`modal_training_gym/common/deployment.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/deployment.py)
