---
title: SglangRecipe
description: API reference for SglangRecipe
---

```python
from modal_training_gym.deploy_recipes.sglang_recipe import SglangRecipe
```

SGLang serving configuration.

**Inherits from:** `BaseDeployRecipe`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recipe_type` | `<enum 'DeployRecipeType` | `sglang` |  |
| `gpu` | `Literal['H100', 'H200', 'B200', 'B300']` | `"H100"` | GPU type for the serving container. Default `"H100"`. |
| `tp` | `int | None` | `None` | Tensor parallelism degree. Default `None` (SGLang infers from GPU count). |
| `dp` | `int | None` | `None` | Data parallelism degree. Enables `--enable-dp-attention` when set. Default `None`. |
| `context_length` | `int | None` | `None` | Maximum context length. Default `None` (model default). |
| `mem_fraction_static` | `float | None` | `None` | Fraction of GPU memory for KV cache. Default `None` (SGLang default). |
| `chunked_prefill_size` | `int | None` | `None` | Chunked prefill token budget. Default `None`. |
| `max_running_requests` | `int | None` | `None` | Max concurrent requests per worker. Default `None`. |
| `sglang_image` | `str` | `"lmsysorg/sglang:nightly-dev-cu13-20260311-dc4380e3"` | Docker image tag for the SGLang container. Default is a recent nightly. |
| `extra_server_args` | `dict[str, str] | None` | `None` | Additional `--flag value` pairs passed to `sglang.launch_server`. Use an empty string value for boolean flags (e.g. `{"--trust-remote-code": ""}`). Default `None`. |
| `environment_name` | `str | None` | `None` | Modal environment to deploy into. Default `None`. |
| `deploy_strategy` | `str` | `"rolling"` | Modal deployment strategy. Default `"rolling"`. |

## Methods

### `server_args(self, *, served_model_name: 'str') -> 'dict[str, str]'`

Build the ``--flag value`` dict for ``SGLangEndpoint``.

**Source:** [`modal_training_gym/deploy_recipes/sglang_recipe.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/deploy_recipes/sglang_recipe.py)
