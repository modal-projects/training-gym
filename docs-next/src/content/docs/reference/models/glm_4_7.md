---
title: GLM-4.7
description: API reference for GLM_4_7
---

# GLM_4_7

```python
from modal_training_gym.common.models.glm_4_7 import GLM_4_7
```

GLM-4.7 large MoE model from Zhipu AI.

**Inherits from:** `HFModelConfiguration`, `ModelConfiguration`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"zai-org/GLM-4.7"` |  |
| `model_path` | `str | None` | `None` |  |
| `architecture` | `ModelArchitecture | None` | `None` |  |

## Methods

### `download_model(self) -> 'None'`

Download or materialize weights into the model volume.

## Related Tutorials

- [GLM-4.7 LoRA SFT on GSM8K (Megatron)](/tutorials/sft/ms_swift_glm_4_7_gsm8k/)

**Source:** [`modal_training_gym/common/models/glm_4_7.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/glm_4_7.py)
