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
| `rm_type` | `str` | `""` | Reward model type identifier (e.g. `"math"`). Default `""`. |

## Methods

### `prepare(self) -> 'None'`

Download and/or preprocess the dataset into the data volume.

**Source:** [`modal_training_gym/common/dataset.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/dataset.py)
