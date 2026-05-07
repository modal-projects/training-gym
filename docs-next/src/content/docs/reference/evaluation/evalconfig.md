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
| `eval_config_id` | `str` | `"eval-config-f4346c92b6d1"` |  |
| `generate_kwargs` | `dict[str, Any]` | `{}` |  |

## Methods

### `build_prompt(self, row: 'DatasetRow') -> 'str'`

### `evaluate(self, deployment: "'ModelDeployment'", debug: 'bool' = False) -> 'EvalResult'`

### `save(self) -> 'EvalConfigDurable'`

### `to_durable(self) -> 'EvalConfigDurable'`

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Code-golf RL with sandboxed verification reward in Modal](/tutorials/rl/001_code_golf_sandboxes/)

**Source:** [`modal_training_gym/common/eval.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/eval.py)
