---
title: HarborDataset
description: API reference for HarborDataset
---

```python
from modal_training_gym.common.dataset import HarborDataset
```

Dataset configuration shared across training frameworks.

**Inherits from:** `DatasetConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_id` | `str` | `""` |  |
| `input_key` | `str` | `""` |  |
| `label_key` | `str` | `""` |  |
| `apply_chat_template` | `bool` | `True` |  |
| `dataset_name` | `str` | `""` |  |
| `path` | `str \| None` | `None` |  |
| `task_root` | `str` | `""` |  |
| `task_glob` | `str` | `"*"` |  |
| `task_names` | `list[str] \| None` | `None` |  |
| `instruction_path` | `str` | `"instruction.md"` |  |
| `label_metadata_path` | `str \| None` | `None` |  |
| `test_data_dir` | `str \| None` | `None` |  |
| `output_format` | `str` | `"parquet"` |  |
| `prompt_template` | `str` | `"{instruction}"` |  |
| `system_prompt` | `str` | `""` |  |
| `train_size` | `int \| None` | `None` |  |
| `eval_size` | `int \| None` | `None` |  |
| `train_repeats` | `int` | `1` |  |
| `eval_repeats` | `int` | `1` |  |
| `shuffle_tasks` | `bool` | `False` |  |
| `shuffle_seed` | `int` | `0` |  |

## Methods

### `load(self) -> 'Any'`

Load raw examples for local inspection or evaluation.

### `prepare(self, path: 'str', eval_paths: 'dict[str, str] | None' = None) -> 'None'`

Materialize training data to `path` (and eval splits to `eval_paths`).

### `to_pandas(self, *, formatted: 'bool' = False)`

### `validate_prepared(self, path: 'str') -> 'None'`

Sniff what `prepare()` wrote and confirm the columns the framework will index.

## Related Tutorials

- [Code RL with Harbor hello-world and sandboxed verification](/tutorials/rl/001_sandboxes/)

**Source:** [`modal_training_gym/common/dataset.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/dataset.py)
