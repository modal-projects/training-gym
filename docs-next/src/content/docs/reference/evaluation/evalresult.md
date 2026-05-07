---
title: EvalResult
description: API reference for EvalResult
---

# EvalResult

```python
from modal_training_gym.common.eval import EvalResult
```

Saved results for one evaluation run across a dataset.

## Constructor

```python
EvalResult(**data)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `eval_id` | `str` |  |  |
| `created_at` | `int` |  |  |
| `config` | `dict[str, Any]` |  |  |
| `rows` | `list[EvalRowResult]` |  |  |

## Methods

### `from_id(eval_id: 'str') -> "'EvalResult'"`

### `list_results() -> "list['EvalResult']"`

### `save(self) -> 'None'`

**Source:** [`modal_training_gym/common/eval.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/eval.py)
