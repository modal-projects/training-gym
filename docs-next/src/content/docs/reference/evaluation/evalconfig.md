---
title: EvalConfig
description: API reference for EvalConfig
---

# EvalConfig

```python
from modal_training_gym.common.eval import EvalConfig
```

Evaluate a deployed model on a dataset config.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `'DatasetConfig'` |  |  |
| `eval_fn` | `EvalFn` |  |  |
| `generate_kwargs` | `dict[str, Any]` | `{}` |  |

## Methods

### `build_prompt(self, row: 'DatasetRow') -> 'str'`

### `evaluate(self, deployment: "'ModelDeployment'", debug: 'bool' = False) -> 'EvalResult'`

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)

**Source:** [`modal_training_gym/common/eval.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/eval.py)
