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
| `input_key` | `str` | `""` |  |
| `label_key` | `str` | `""` |  |
| `apply_chat_template` | `bool` | `True` |  |

## Methods

### `load(self) -> 'Iterable[DatasetRow]'`

Load raw examples for local inspection or evaluation.

### `prepare(self, path: 'str', eval_paths: 'dict[str, str] | None' = None) -> 'None'`

Materialize training data to ``path`` (and eval splits to ``eval_paths``).

**Source:** [`modal_training_gym/common/dataset.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/dataset.py)
