---
title: HarborFrameworkConfig
description: API reference for HarborFrameworkConfig
---

# HarborFrameworkConfig

```python
from modal_training_gym.frameworks.harbor.config import HarborFrameworkConfig
```

Harbor + Miles configuration for sandbox-based RL training.

**Inherits from:** `MilesFrameworkConfig`

## Harbor Agent

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent_import_path` | `str` | `""` | Python import path for the Harbor agent class. Default `""`. |
| `agent_model_name` | `str` | `"model"` | Model name passed to the agent. Default `"model"`. |
| `agent_kwargs` | `dict[str, Any]` | `{}` | Extra keyword arguments for agent construction. Default `{}`. |

## Harbor Environment

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `environment_import_path` | `str | None` | `None` | Import path for custom Harbor environment. Default `None`. |
| `sandbox_timeout_secs` | `int` | `1800` | Maximum sandbox execution time in seconds. Default `1800` (30 min). |
| `sandbox_idle_timeout_secs` | `int` | `300` | Sandbox idle timeout in seconds. Default `300` (5 min). |

## Harbor Tasks

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_root` | `str` | `"/data/tasks"` | Root directory for Harbor task directories on the data volume. Default `"/data/tasks"`. |
| `task_glob` | `str` | `"*"` | Glob pattern for discovering task directories. Default `"*"`. |
| `instruction_path` | `str` | `"instruction.md"` | Relative path to the instruction file within each task dir. Default `"instruction.md"`. |

## Image

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `miles_image` | `str` | `"radixark/miles:dev-202604201238"` | Docker image with Miles trainer. Default `"radixark/miles:dev-202604201238"`. |
| `miles_src_commit` | `str` | `"9a003644739f4e6dd509e2e8337e8ae7e571941c"` | Miles source commit with `--custom-agent-function-path` support. |
| `harbor_install_command` | `str` | `"uv pip install --system git+https://github.com/laude-institute/harbor.git"` | Command to install Harbor in the image. |

## Other Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu` | `Literal['H100', 'H200', 'B200', 'B300']` | `"H100"` |  |
| `image_run_commands` | `list[str]` | `[]` |  |
| `n_nodes` | `int` | `1` |  |
| `recipe_args` | `str` | `""` |  |
| `extra_args` | `str` | `""` |  |
| `custom_config_yaml` | `str` | `""` |  |
| `colocate` | `bool` | `True` |  |
| `actor_nodes` | `int | None` | `None` |  |
| `rollout_num_gpus` | `int | None` | `None` |  |
| `app_tags` | `dict` | `{}` |  |
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

## Methods

### `cli_args(self) -> 'list[str]'`

Emit Miles CLI flags, skipping Harbor-specific fields.

### `parsed_recipe_args(self) -> 'list[str]'`

Shlex-parse `recipe_args` + `extra_args` into a flat argv list.

### `resolved_actor_nodes(self) -> 'int'`

### `resolved_rollout_num_gpus(self) -> 'int | None'`

## Related Tutorials

- [Qwen3-4B RL code-golf on MBPP with Harbor sandboxes](/tutorials/rl/harbor_code_golf/)

**Source:** [`modal_training_gym/frameworks/harbor/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/harbor/config.py)
