---
title: MultiTurn
description: API reference for MultiTurn
---

```python
from modal_training_gym.train_recipes.slime_recipe.blocks import MultiTurn
```

MultiTurn(generate_function: 'Callable', reward_function: 'Callable | None' = None, max_turns: 'int | None' = None, log_multi_turn: 'bool | None' = None, config_overrides: 'Mapping[str, Any]' = <factory>)

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generate_function` | `collections.abc.Callable` |  |  |
| `reward_function` | `collections.abc.Callable \| None` | `None` |  |
| `max_turns` | `int \| None` | `None` |  |
| `log_multi_turn` | `bool \| None` | `None` |  |
| `config_overrides` | `collections.abc.Mapping[str, Any]` | `{}` |  |

## Methods

### `apply(self, recipe: "'SlimeRecipe'") -> "'SlimeRecipe'"`

**Source:** [`modal_training_gym/train_recipes/slime_recipe/blocks.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/train_recipes/slime_recipe/blocks.py)
