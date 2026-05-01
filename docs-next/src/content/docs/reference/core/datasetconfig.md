---
title: DatasetConfig
description: API reference for DatasetConfig
---

# DatasetConfig

```python
from modal_training_gym.common.dataset import DatasetConfig
```

Dataset configuration shared across training frameworks.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_data` | `str` | `""` | Path to the training data file (e.g. a `.parquet` file on the data volume). Default `""`. |
| `eval_prompt_data` | `list[str] | str | None` | `None` | Evaluation data path(s). Can be a single path, a list of paths, or `None` to skip evaluation. Default `None`. |
| `input_key` | `str` | `""` | Column/key name for model input in the dataset. Default `""`. |
| `label_key` | `str` | `""` | Column/key name for labels/targets in the dataset. Default `""`. |
| `apply_chat_template` | `bool` | `True` | Whether to apply the model's chat template to inputs. Default `True`. |
| `rollout_shuffle` | `bool` | `True` | Whether to shuffle data during rollout generation. Default `True`. |

## Methods

### `load(self)`

Load raw examples for local inspection or evaluation.

### `prepare(self) -> 'None'`

Download and/or preprocess the dataset into the data volume.

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/001_quickstart/)
- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfig` subclass, no catalog entry](/tutorials/intro/002_custom_model/)

**Source:** [`modal_training_gym/common/dataset.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/dataset.py)
