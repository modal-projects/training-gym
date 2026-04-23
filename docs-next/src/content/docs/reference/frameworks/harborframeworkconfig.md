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

## Harbor RL Defaults

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input_key` | `str` | `"prompt"` | Dataset column key for model input prompts. Default `"prompt"`. |
| `label_key` | `str` | `"metadata"` | Dataset column key for metadata/labels. Default `"metadata"`. |
| `apply_chat_template` | `bool` | `True` | Apply the model's chat template to inputs. Default `True`. |
| `enable_thinking` | `bool` | `True` | Enable thinking/reasoning mode in the model. Default `True`. |
| `rollout_shuffle` | `bool` | `True` | Shuffle data during rollout generation. Default `True`. |

## Parallelism (Harbor Overrides)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tensor_model_parallel_size` | `int` | `2` | Tensor parallelism degree. Default `2`. |
| `sequence_parallel` | `bool` | `True` | Enable sequence parallelism. Default `True`. |

## Memory (Harbor Overrides)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recompute_granularity` | `str` | `"selective"` | Activation recomputation granularity. Default `"selective"`. |

## Model Overrides

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_position_embeddings` | `int` | `32768` | Maximum sequence position embeddings. Default `32768`. |
| `untie_embeddings_and_output_weights` | `bool` | `True` | Untie input embeddings from output projection. Default `True`. |
| `no_masked_softmax_fusion` | `bool` | `True` | Disable masked softmax fusion. Default `True`. |

## Rollout (Harbor)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_rollout` | `int` | `200` | Number of rollout episodes per training step. Default `200`. |
| `rollout_batch_size` | `int` | `64` | Batch size for rollout generation. Default `64`. |
| `rollout_max_response_len` | `int` | `1024` | Maximum response length during rollout. Default `1024`. |
| `sglang_mem_fraction_static` | `float` | `0.7` | SGLang static memory fraction. Default `0.7`. |

## Training (Harbor Overrides)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_iters` | `int` | `50` | Total training iterations. Default `50`. |
| `global_batch_size` | `int` | `512` | Global batch size across all ranks. Default `512`. |

## Eval and Checkpointing (Harbor)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eval_interval` | `int` | `10` | Evaluation interval in iterations. Default `10`. |
| `save_interval` | `int` | `10` | Checkpoint save interval in iterations. Default `10`. |

## Image

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `miles_image` | `str` | `"radixark/miles:dev-202604201238"` | Docker image with Miles trainer. Default `"radixark/miles:dev-202604201238"`. |
| `miles_src_commit` | `str` | `"9a003644739f4e6dd509e2e8337e8ae7e571941c"` | Miles source commit with `--custom-agent-function-path` support. |
| `harbor_install_command` | `str` | `"uv pip install --system git+https://github.com/laude-institute/harbor.git"` | Command to install Harbor in the image. |

## Modal Infrastructure

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu` | `Literal['H100', 'H200', 'B200', 'B300']` | `"H100"` | Modal GPU type. Default `"H100"`. |
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

Emit Miles CLI flags, skipping Harbor-specific fields.

### `parsed_recipe_args(self) -> 'list[str]'`

Shlex-parse `recipe_args` + `extra_args` into a flat argv list.

### `resolved_actor_nodes(self) -> 'int'`

### `resolved_rollout_num_gpus(self) -> 'int | None'`

## Related Tutorials

- [Qwen3-4B RL code-golf on MBPP with Harbor sandboxes](/tutorials/rl/harbor_code_golf/)

**Source:** [`modal_training_gym/frameworks/harbor/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/harbor/config.py)
