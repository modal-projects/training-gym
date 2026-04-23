---
title: MilesFrameworkConfig
description: API reference for MilesFrameworkConfig
---

# MilesFrameworkConfig

```python
from modal_training_gym.frameworks.miles.config import MilesFrameworkConfig
```

Miles RLVR configuration, including Modal infrastructure.

## Modal Infrastructure

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `miles_image` | `str` | `"radixark/miles:dev-202603231227"` | Docker image with patched Megatron-LM + Miles trainer. |
| `image_run_commands` | `list[str]` | `[]` | Extra commands appended to the image build. Default `[]`. |
| `n_nodes` | `int` | `1` | Number of cluster nodes. Default `1`. |
| `app_tags` | `dict` | `{}` | Extra Modal app tags. Default `{}`. |

## Recipe and Overrides

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recipe_args` | `str` | `""` | Raw Miles CLI flag block. Model architecture flags and per-run overrides go here. Values override typed defaults. Default `""`. |
| `extra_args` | `str` | `""` | Extra flags appended after `recipe_args`. Default `""`. |
| `custom_config_yaml` | `str` | `""` | Inline YAML overrides passed via `--custom-config-path`. Default `""`. |

## Actor and Rollout Topology

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `colocate` | `bool` | `True` | Reuse a single compute pool for actor + rollout. Default `True`. |
| `actor_nodes` | `int | None` | `None` | Actor node count. Defaults to `n_nodes` (colocated) or `n_nodes - 1` (non-colocated). Default `None`. |
| `rollout_num_gpus` | `int | None` | `None` | Rollout GPU count for non-colocated mode. Default `None`. |

## RL Algorithm

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `advantage_estimator` | `str` | `"grpo"` | Advantage estimation method. Default `"grpo"`. |
| `eps_clip` | `float` | `0.2` | PPO clipping epsilon. Default `0.2`. |
| `clip_grad` | `float` | `1.0` | Gradient clipping norm. Default `1.0`. |
| `kl_coef` | `float` | `0.0` | KL divergence penalty coefficient. Default `0.0`. |
| `normalize_advantages` | `bool` | `False` | Normalize advantages. Default `False`. |
| `seed` | `int` | `1234` | Random seed. Default `1234`. |

## Optimizer

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lr` | `float` | `1e-06` | Learning rate. Default `1e-6`. |
| `lr_decay_style` | `str` | `"constant"` | LR decay schedule. Default `"constant"`. |
| `weight_decay` | `float` | `0.0` | Weight decay. Default `0.0`. |
| `adam_beta1` | `float` | `0.9` | Adam beta1. Default `0.9`. |
| `adam_beta2` | `float` | `0.95` | Adam beta2. Default `0.95`. |

## Batch

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `micro_batch_size` | `int` | `1` | Micro batch size per GPU. Default `1`. |
| `n_samples_per_prompt` | `int` | `8` | Rollout samples per prompt. Default `8`. |

## Precision and Memory

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bf16` | `bool` | `True` | Enable BF16 training. Default `True`. |
| `attention_softmax_in_fp32` | `bool` | `True` | Compute attention softmax in FP32. Default `True`. |

## Regularization

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `attention_dropout` | `float` | `0.0` | Attention dropout rate. Default `0.0`. |
| `hidden_dropout` | `float` | `0.0` | Hidden layer dropout rate. Default `0.0`. |

## Rollout

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rollout_temperature` | `float` | `1.0` | Sampling temperature for rollouts. Default `1.0`. |

## Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `no_save_optim` | `bool` | `True` | Skip saving optimizer state. Default `True`. |

## Methods

### `cli_args(self) -> 'list[str]'`

Convert typed training defaults to Miles CLI flags.

### `parsed_recipe_args(self) -> 'list[str]'`

Shlex-parse `recipe_args` + `extra_args` into a flat argv list.

### `resolved_actor_nodes(self) -> 'int'`

### `resolved_rollout_num_gpus(self) -> 'int | None'`

**Source:** [`modal_training_gym/frameworks/miles/config.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/modal_training_gym/frameworks/miles/config.py)
