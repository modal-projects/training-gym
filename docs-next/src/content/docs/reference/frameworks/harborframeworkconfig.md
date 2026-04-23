---
title: HarborFrameworkConfig
description: API reference for HarborFrameworkConfig
---

# HarborFrameworkConfig

```python
from modal_training_gym.frameworks.harbor.config import HarborFrameworkConfig
```

Harbor + Miles configuration.

**Inherits from:** `MilesFrameworkConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu` | `Literal['H100', 'H200', 'B200', 'B300']` | `"H100"` |  |
| `miles_image` | `str` | `"radixark/miles:dev-202604201238"` |  |
| `image_run_commands` | `list[str]` | `<dataclasses._MISSING_TYPE object at 0x7f6b4318eb70>` |  |
| `n_nodes` | `int` | `1` |  |
| `recipe_args` | `str` | `""` |  |
| `extra_args` | `str` | `""` |  |
| `custom_config_yaml` | `str` | `""` |  |
| `colocate` | `bool` | `True` |  |
| `actor_nodes` | `int | None` | `None` |  |
| `rollout_num_gpus` | `int | None` | `None` |  |
| `app_tags` | `dict` | `<dataclasses._MISSING_TYPE object at 0x7f6b4318eb70>` |  |
| `advantage_estimator` | `str` | `"grpo"` |  |
| `eps_clip` | `float` | `0.2` |  |
| `clip_grad` | `float` | `1.0` |  |
| `kl_coef` | `float` | `0.0` |  |
| `normalize_advantages` | `bool` | `False` |  |
| `seed` | `int` | `1234` |  |
| `lr` | `float` | `1e-06` |  |
| `lr_decay_style` | `str` | `"constant"` |  |
| `weight_decay` | `float` | `0.0` |  |
| `adam_beta1` | `float` | `0.9` |  |
| `adam_beta2` | `float` | `0.95` |  |
| `micro_batch_size` | `int` | `1` |  |
| `n_samples_per_prompt` | `int` | `8` |  |
| `bf16` | `bool` | `True` |  |
| `attention_softmax_in_fp32` | `bool` | `True` |  |
| `attention_dropout` | `float` | `0.0` |  |
| `hidden_dropout` | `float` | `0.0` |  |
| `rollout_temperature` | `float` | `1.0` |  |
| `no_save_optim` | `bool` | `True` |  |
| `agent_import_path` | `str` | `""` |  |
| `agent_model_name` | `str` | `"model"` |  |
| `agent_kwargs` | `dict[str, Any]` | `<dataclasses._MISSING_TYPE object at 0x7f6b4318eb70>` |  |
| `environment_import_path` | `str | None` | `None` |  |
| `sandbox_timeout_secs` | `int` | `1800` |  |
| `sandbox_idle_timeout_secs` | `int` | `300` |  |
| `task_root` | `str` | `"/data/tasks"` |  |
| `task_glob` | `str` | `"*"` |  |
| `instruction_path` | `str` | `"instruction.md"` |  |
| `miles_src_commit` | `str` | `"9a003644739f4e6dd509e2e8337e8ae7e571941c"` |  |
| `harbor_install_command` | `str` | `"uv pip install --system git+https://github.com/laude-institute/harbor.git"` |  |

## Methods

### `cli_args(self) -> 'list[str]'`

Emit Miles CLI flags, skipping Harbor-specific fields.

### `parsed_recipe_args(self) -> 'list[str]'`

Shlex-parse `recipe_args` + `extra_args` into a flat argv list.

### `resolved_actor_nodes(self) -> 'int'`

### `resolved_rollout_num_gpus(self) -> 'int | None'`

**Source:** [`modal_training_gym/frameworks/harbor/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/harbor/config.py)
