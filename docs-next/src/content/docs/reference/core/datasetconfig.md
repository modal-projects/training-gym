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
| `dataset_id` | `str` | `Field(name=None,type=None,default=<dataclasses._MISSING_TYPE object at 0x7f99a7817170>,default_factory=<function DatasetConfig.<lambda> at 0x7f99a7400e00>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),kw_only=<dataclasses._MISSING_TYPE object at 0x7f99a7817170>,_field_type=None)` |  |
| `input_key` | `str` | `""` |  |
| `label_key` | `str` | `""` |  |
| `apply_chat_template` | `bool` | `True` |  |

## Methods

### `load(self) -> 'Any'`

Load raw examples for local inspection or evaluation.

### `prepare(self, path: 'str', eval_paths: 'dict[str, str] | None' = None) -> 'None'`

Materialize training data to ``path`` (and eval splits to ``eval_paths``).

## Related Tutorials

- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn_number_guessing/)

**Source:** [`modal_training_gym/common/dataset.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/dataset.py)
