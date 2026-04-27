---
title: ModelConfiguration
description: API reference for ModelConfiguration
---

# ModelConfiguration

```python
from modal_training_gym.common.models.base import ModelConfiguration
```

Base class for model identity and weight-download logic.

## Constructor

```python
ModelConfiguration(**kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `""` | HuggingFace repo ID or other model identifier. Default `""`. |
| `model_path` | `str | None` | `None` | Override local path for model weights. When `None`, frameworks derive the path from `model_name`. Default `None`. |
| `architecture` | `ModelArchitecture | None` | `None` | Transformer architecture spec for Megatron-based frameworks. Not required for HuggingFace-only workflows. Default `None`. |
| `training` | `ModelTrainingConfig | None` | `None` | Optimal training parameters (parallelism, MoE, LoRA) for this model on a specific GPU type. Frameworks pull defaults from here so users don't need to manually specify model-tuned flags. Default `None`. |

## Methods

### `download_model(self) -> 'None'`

Download or materialize weights into the model volume.

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/001_quickstart/)
- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfiguration` subclass, no catalog entry](/tutorials/intro/002_custom_model/)

**Source:** [`modal_training_gym/common/models/base.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/base.py)
