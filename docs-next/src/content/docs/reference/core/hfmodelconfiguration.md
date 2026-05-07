---
title: HFModelConfiguration
description: API reference for HFModelConfiguration
---

# HFModelConfiguration

```python
from modal_training_gym.common.models.base import HFModelConfiguration
```

ModelConfig for models hosted on HuggingFace.

**Inherits from:** `ModelConfig`

## Constructor

```python
HFModelConfiguration(**kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `""` |  |
| `model_path` | `str | None` | `None` |  |
| `architecture` | `ModelArchitecture | None` | `None` |  |

## Methods

### `download(self) -> 'None'`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/base.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/base.py)
