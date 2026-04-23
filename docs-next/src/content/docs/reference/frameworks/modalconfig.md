---
title: ModalConfig (SLIME)
description: API reference for ModalConfig
---

# ModalConfig

```python
from modal_training_gym.frameworks.slime.config import ModalConfig
```

Modal infrastructure configuration — GPU provisioning and image setup only.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu` | `Literal['H100', 'H200', 'B200', 'B300']` | `"H100"` |  |
| `local_slime` | `str | None` | `None` |  |
| `patch_files` | `list[str]` | `[]` |  |
| `image_run_commands` | `list[str]` | `[]` |  |
| `local_python_sources` | `list[str]` | `[]` |  |

**Source:** [`modal_training_gym/frameworks/slime/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/slime/config.py)
