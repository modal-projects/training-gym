---
title: MsSwiftConfig
description: API reference for MsSwiftConfig
---

# MsSwiftConfig

```python
from modal_training_gym.frameworks.ms_swift.config import MsSwiftConfig
```

Top-level wrapper that composes an ms-swift Megatron SFT run.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `'DatasetConfig | None'` | `None` | Dataset configuration with `prepare()` hook. Default `None`. |
| `model` | `'ModelConfiguration | None'` | `None` | Model identity and download hook. Default `None`. |
| `wandb` | `'WandbConfig | None'` | `None` | Weights & Biases logging config. Default `None`. |
| `framework_config` | `MsSwiftFrameworkConfig` | `MsSwiftFrameworkConfig()` | ms-swift Megatron training and infrastructure settings. Default `MsSwiftFrameworkConfig()`. |

## Methods

### `build_app(self, *, name: 'str | None' = None) -> "'App'"`

### `cli_args(self, *, output_dir: 'str') -> 'list[str]'`

### `resolved_framework_config(self) -> 'MsSwiftFrameworkConfig'`

Return framework config with model defaults applied to untouched fields.

## Related Tutorials

- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfiguration` subclass, no catalog entry](/tutorials/intro/002_custom_model/)
- [GLM-4.7 LoRA SFT on GSM8K (Megatron)](/tutorials/sft/001_ms_swift/)

**Source:** [`modal_training_gym/frameworks/ms_swift/config.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/modal_training_gym/frameworks/ms_swift/config.py)
