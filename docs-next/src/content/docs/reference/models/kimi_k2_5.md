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
| `model_name` | `str` | `"moonshotai/Kimi-K2.5"` | HuggingFace repo ID or other model identifier. Default `""`. |
| `model_path` | `str | None` | `"/checkpoints/Kimi-K2.5-bf16"` | Override local path for model weights. When `None`, frameworks derive the path from `model_name`. Default `None`. |
| `architecture` | `ModelArchitecture | None` | `None` | Transformer architecture spec for Megatron-based frameworks. Not required for HuggingFace-only workflows. Default `None`. |

## Methods

### `download_model(self) -> None`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/kimi_k2_5.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/kimi_k2_5.py)
