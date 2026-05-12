---
title: HuggingFaceDataset
description: API reference for HuggingFaceDataset
---

```python
from modal_training_gym.common.dataset import HuggingFaceDataset
```

Dataset backed by a HuggingFace `datasets` repo.

**Inherits from:** `DatasetConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_id` | `str` | `""` |  |
| `input_key` | `str` | `""` |  |
| `label_key` | `str` | `""` |  |
| `apply_chat_template` | `bool` | `True` |  |
| `hf_repo` | `str` | `""` |  |
| `hf_split` | `str` | `"train"` |  |
| `hf_config` | `str \| None` | `None` |  |
| `output_format` | `str` | `"parquet"` |  |
| `input_column` | `str` | `""` |  |
| `output_column` | `str` | `""` |  |
| `system_prompt` | `str` | `""` |  |
| `prompt_template` | `str` | `"{input}"` |  |
| `n_rows` | `int` | `0` |  |

## Methods

### `load(self) -> 'Any'`

Load raw examples for local inspection or evaluation.

### `prepare(self, path: 'str', eval_paths: 'dict[str, str] | None' = None) -> 'None'`

Materialize training data to `path` (and eval splits to `eval_paths`).

### `to_pandas(self, *, formatted: 'bool' = False)`

**Source:** [`modal_training_gym/common/dataset.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/dataset.py)
