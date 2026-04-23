---
title: Qwen3-32B
description: API reference for Qwen3_32B
---

# Qwen3_32B

```python
from modal_training_gym.common.models.qwen3_32b import Qwen3_32B
```

Qwen3-32B (32 billion parameters) from Alibaba.

**Inherits from:** `HFModelConfiguration`, `ModelConfiguration`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen3-32B"` |  |
| `model_path` | `str | None` | `None` |  |
| `architecture` | `ModelArchitecture | None` | `None` |  |

## Methods

### `download_model(self) -> 'None'`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/qwen3_32b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/qwen3_32b.py)
