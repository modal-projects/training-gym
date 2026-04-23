---
title: Llama2-7B
description: API reference for Llama2_7B
---

# Llama2_7B

```python
from modal_training_gym.common.models.llama2_7b import Llama2_7B
```

Llama 2 7B from Meta.

**Inherits from:** `HFModelConfiguration`, `ModelConfiguration`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"meta-llama/Llama-2-7b-hf"` |  |
| `model_path` | `str | None` | `None` |  |
| `architecture` | `ModelArchitecture | None` | `None` |  |

## Methods

### `download_model(self) -> 'None'`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/llama2_7b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/llama2_7b.py)
