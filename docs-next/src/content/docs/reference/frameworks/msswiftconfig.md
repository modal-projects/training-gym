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
| `model` | `'ModelConfig | None'` | `None` | Model identity and download hook. Default `None`. |
| `wandb` | `'WandbConfig | None'` | `None` | Weights & Biases logging config. Default `None`. |
| `framework_config` | `MsSwiftFrameworkConfig` | `MsSwiftFrameworkConfig()` | ms-swift Megatron training and infrastructure settings. Default `MsSwiftFrameworkConfig()`. |
| `checkpoint` | `CheckpointConfig` | `None` | Checkpoint directory layout. Default uses `iteration_prefix="iter_"`. |

## Methods

### `build_app(self, *, name: 'str | None' = None) -> "'App'"`

### `cli_args(self, *, output_dir: 'str') -> 'list[str]'`

### `resolved_framework_config(self) -> 'MsSwiftFrameworkConfig'`

Return framework config with model defaults applied to untouched fields.

## Related Tutorials

- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfig` subclass, no catalog entry](/tutorials/intro/002_custom_model/)
- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Qwen3-0.6B SFT on generated arithmetic problems](/tutorials/sft/000_sft_basics/)
- [GLM-4.7 LoRA SFT on GSM8K (Megatron)](/tutorials/sft/001_ms_swift/)

**Source:** [`modal_training_gym/frameworks/ms_swift/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/ms_swift/config.py)
