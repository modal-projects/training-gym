---
title: VllmRecipe
description: API reference for VllmRecipe
---

```python
from modal_training_gym.deploy_recipes.vllm_recipe import VllmRecipe
```

vLLM serving configuration.

**Inherits from:** `BaseDeployRecipe`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recipe_type` | `DeployRecipeType` | `vllm` |  |
| `gpu` | `Optional[Literal['H100', 'H200', 'B200', 'B300']]` | `None` | GPU type for the serving container. Default `None` (inferred). |
| `n_gpu` | `int \| None` | `None` | Number of GPUs (tensor-parallel degree for vLLM). Default `None`. |
| `extra_vllm_args` | `list[str] \| None` | `None` | Additional CLI args passed to `vllm serve`. Default `None`. |
| `environment_name` | `str \| None` | `None` | Modal environment to deploy into. Default `None`. |
| `deploy_strategy` | `str` | `"rolling"` | Modal deployment strategy. Default `"rolling"`. |

**Source:** [`modal_training_gym/deploy_recipes/vllm_recipe.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/deploy_recipes/vllm_recipe.py)
