---
title: ModelConfiguration
description: API reference for ModelConfiguration
---

# ModelConfiguration

```python
from modal_training_gym.common.models.base import ModelConfiguration
```

Base class for model identity and weight-download logic.

## Constructor

```python
ModelConfiguration(**kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `""` | HuggingFace repo ID or other model identifier. Default `""`. |
| `model_path` | `str | None` | `None` | Override local path for model weights. When `None`, frameworks derive the path from `model_name`. Default `None`. |
| `architecture` | `ModelArchitecture | None` | `None` | Transformer architecture spec for Megatron-based frameworks. Not required for HuggingFace-only workflows. Default `None`. |

## Methods

### `download_model(self) -> 'None'`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/base.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/base.py)
