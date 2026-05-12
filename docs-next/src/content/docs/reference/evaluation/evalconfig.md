---
title: EvalConfig
description: API reference for EvalConfig
---

```python
from modal_training_gym.common.eval import EvalConfig
```

Evaluate a deployed model on a dataset config.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `'DatasetConfig'` |  |  |
| `eval_fn` | `EvalFn \| None` | `None` |  |
| `eval_response_fn` | `EvalResponseFn \| None` | `None` |  |
| `prompt_column` | `str \| None` | `None` |  |
| `eval_config_id` | `str \| None` | `None` |  |
| `generate_kwargs` | `dict[str, Any]` | `{}` |  |

## Methods

### `build_prompt(self, row: 'DatasetRow') -> 'str'`

### `evaluate(self, deployment: "'ModelDeployment'", debug: 'bool' = False, max_concurrency: 'int' = 1) -> 'EvalResult'`

### `save(self) -> 'EvalConfigDurable'`

### `to_durable(self) -> 'EvalConfigDurable'`

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Code RL with Harbor hello-world and sandboxed verification](/tutorials/rl/001_sandboxes/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn/)

**Source:** [`modal_training_gym/common/eval.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/eval.py)
