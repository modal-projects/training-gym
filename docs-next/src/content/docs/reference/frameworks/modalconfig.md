---
title: ModalConfig (slime)
description: API reference for ModalConfig
---

# ModalConfig

```python
from modal_training_gym.frameworks.slime.config import ModalConfig
```

Modal infrastructure configuration for slime — image setup and dev overlays.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local_slime` | `str | None` | `None` | Path to a local slime repo checkout for dev overlay. When set, the launcher mounts it into the image. Default `None`. |
| `patch_files` | `list[str]` |  | Local patch files to inject into the image at `/tmp/<filename>`. Default `[]`. |
| `image_run_commands` | `list[str]` |  | Shell commands run during image build (e.g. `git apply /tmp/my.patch`). Default `[]`. |
| `local_python_sources` | `list[str]` |  | Sibling Python modules (by import name) to ship into the training image via `add_local_python_source`. Use for helper modules like custom reward functions referenced via slime's `custom_rm_path`. Default `[]`. |

## Related Tutorials

- [Qwen3-4B GRPO on haiku poems — structure score + LLM judge](/tutorials/rl/003_slime_with_llm_as_judge/)

**Source:** [`modal_training_gym/frameworks/slime/config.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/modal_training_gym/frameworks/slime/config.py)
