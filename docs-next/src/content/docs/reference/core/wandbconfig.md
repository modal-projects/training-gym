---
title: WandbConfig
description: API reference for WandbConfig
---

# WandbConfig

```python
from modal_training_gym.common.wandb import WandbConfig
```

Weights & Biases logging configuration shared across all frameworks.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | `str` | `""` | W&B project name. Default `""`. |
| `group` | `str` | `""` | W&B group tag for organizing related runs. Default `""`. |
| `exp_name` | `str` | `""` | W&B run display name. Default `""`. |
| `key` | `str` | `""` | W&B API key. Usually injected via `WANDB_API_KEY` at launch time rather than hardcoded. Default `""`. |
| `disable_random_suffix` | `bool` | `True` | When `True`, suppresses the random suffix that W&B appends to run names. Default `True`. |

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/001_quickstart/)
- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfig` subclass, no catalog entry](/tutorials/intro/002_custom_model/)
- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)

**Source:** [`modal_training_gym/common/wandb.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/wandb.py)
