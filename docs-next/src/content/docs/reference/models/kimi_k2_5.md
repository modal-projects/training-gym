---
title: Kimi-K2.5
description: API reference for Kimi_K2_5
---

# Kimi_K2_5

```python
from modal_training_gym.common.models.kimi_k2_5 import Kimi_K2_5
```

Kimi K2.5 from Moonshot AI.

**Inherits from:** `HFModelConfiguration`, `ModelConfiguration`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"moonshotai/Kimi-K2.5"` |  |
| `model_path` | `str | None` | `"/checkpoints/Kimi-K2.5-bf16"` |  |
| `architecture` | `ModelArchitecture | None` | `None` |  |

## Methods

### `download_model(self) -> None`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/kimi_k2_5.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/kimi_k2_5.py)
