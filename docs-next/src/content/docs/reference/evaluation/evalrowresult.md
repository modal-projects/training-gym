---
title: EvalRowResult
description: API reference for EvalRowResult
---

```python
from modal_training_gym.common.eval import EvalRowResult
```

One evaluated row: score, response text, and optional metadata.

## Constructor

```python
EvalRowResult(**data)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|

## Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `score` | `float` |  |  |
| `response` | `str` |  |  |
| `metadata` | `dict[str, Any]` |  |  |

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Code RL with Harbor hello-world and sandboxed verification](/tutorials/rl/001_sandboxes/)
- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn/)

**Source:** [`modal_training_gym/common/eval.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/eval.py)
