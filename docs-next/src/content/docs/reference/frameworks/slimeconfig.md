---
title: SlimeConfig
description: API reference for SlimeConfig
---

# SlimeConfig

```python
from modal_training_gym.frameworks.slime.config import SlimeConfig
```

slime GRPO training configuration.

## Launcher Instructions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `environment` | `dict` | `{'PYTHONPATH': '/root/Megatron-LM/', 'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '1'}` | Injected into the Ray job runtime env. |
| `async_mode` | `bool` | `False` | When `True`, uses `train_async.py`. Default `False`. |
| `slime_model_script` | `str` | `""` | Shell script path relative to `/root/slime`. Default `""`. |
| `model` | `ModelConfiguration | None` | `None` | Model identity. Requires `ModelArchitecture`. Default `None`. |

## Composed Configs

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | `DatasetConfig | None` | `None` | Dataset configuration. Default `None`. |
| `wandb` | `WandbConfig | None` | `None` | W&B logging config. Default `None`. |
| `modal` | `ModalConfig | None` | `None` | Modal infrastructure config. Default `None`. |
| `app_tags` | `dict` | `{}` | Extra Modal app tags. Default `{}`. |

## Cluster and Parallelism

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `actor_num_nodes` | `int | None` | `None` | Number of actor training nodes. Default `1`. |
| `actor_num_gpus_per_node` | `int | None` | `None` | GPUs per actor node. Default `8`. |
| `colocate` | `bool | None` | `None` | Colocate actor and rollout on the same GPUs. Default `False`. |
| `rollout_num_gpus` | `int | None` | `None` | GPU count for non-colocated rollout. Default `None`. |
| `rollout_num_gpus_per_engine` | `int` | `1` | GPUs per SGLang rollout engine. Default `1`. |
| `tensor_model_parallel_size` | `int | None` | `None` | Tensor parallelism degree. Default `1`. |
| `use_critic` | `bool` | `False` | Use a separate critic network. Default `False`. |
| `critic_num_nodes` | `int | None` | `None` | Critic node count. Default `None`. |
| `critic_num_gpus_per_node` | `int | None` | `None` | GPUs per critic node. Default `None`. |

## RL Algorithm

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `advantage_estimator` | `str` | `"grpo"` | Advantage estimation method. Default `"grpo"`. |
| `n_samples_per_prompt` | `int` | `2` | Rollout samples per prompt. Default `2`. |
| `eps_clip` | `float` | `0.2` | PPO clipping epsilon. Default `0.2`. |
| `eps_clip_high` | `float` | `0.28` | Asymmetric high-side PPO clip. Default `0.28`. |
| `use_kl_loss` | `bool` | `True` | Enable KL divergence loss. Default `False`. |
| `kl_loss_type` | `str` | `"low_var_kl"` | KL loss variant. Default `"low_var_kl"`. |
| `kl_loss_coef` | `float` | `0.0` | KL loss coefficient. Default `0.0`. |
| `entropy_coef` | `float` | `0.0` | Entropy bonus coefficient. Default `0.0`. |
| `ref_load` | `str` | `""` | Reference model checkpoint path. Default `""`. |

## Rollout

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_rollout` | `int` | `1` | Number of rollout episodes per step. Default `1`. |
| `rollout_batch_size` | `int` | `8` | Batch size for rollout generation. Default `8`. |
| `rollout_max_response_len` | `int` | `8192` | Maximum response length during rollout. Default `8192`. |
| `rollout_temperature` | `float` | `1.0` | Sampling temperature for rollouts. Default `1.0`. |
| `sglang_mem_fraction_static` | `float` | `0.7` | SGLang static memory fraction. Default `0.7`. |

## Training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `global_batch_size` | `int` | `16` | Global batch size. Default `16`. |
| `lr` | `float` | `1e-06` | Learning rate. Default `1e-6`. |
| `lr_decay_style` | `str` | `"constant"` | LR decay schedule. Default `"constant"`. |
| `weight_decay` | `float` | `0.1` | Weight decay. Default `0.0`. |
| `adam_beta1` | `float` | `0.9` | Adam beta1. Default `0.9`. |
| `adam_beta2` | `float` | `0.98` | Adam beta2. Default `0.95`. |
| `optimizer` | `str` | `"adam"` | Optimizer name. Default `"adam"`. |
| `name` | `str` | `""` |  |

## Memory and Precision

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `attention_dropout` | `float` | `0.0` | Attention dropout. Default `0.0`. |
| `hidden_dropout` | `float` | `0.0` | Hidden layer dropout. Default `0.0`. |
| `attention_softmax_in_fp32` | `bool` | `True` | Compute softmax in FP32. Default `True`. |
| `accumulate_allreduce_grads_in_fp32` | `bool` | `True` | Accumulate allreduce grads in FP32. Default `True`. |
| `recompute_granularity` | `str` | `"full"` | Activation recomputation granularity. Default `""`. |
| `recompute_method` | `str` | `"uniform"` | Activation recomputation method. Default `""`. |
| `recompute_num_layers` | `int` | `1` | Number of layers to recompute. Default `None`. |

## Dynamic Batching

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_dynamic_batch_size` | `bool` | `True` | Use dynamic batch sizing. Default `False`. |
| `max_tokens_per_gpu` | `int` | `9216` | Max tokens per GPU for dynamic batching. Default `None`. |

## Eval

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eval_interval` | `int` | `20` | Evaluation interval in training steps. Default `20`. |
| `n_samples_per_eval_prompt` | `int` | `4` | Eval samples per prompt. Default `4`. |
| `eval_max_response_len` | `int` | `16384` | Max response length for eval. Default `16384`. |
| `eval_top_p` | `float` | `1.0` | Top-p sampling for eval. Default `1.0`. |
| `eval_config` | `dict | None` | `None` | YAML eval configuration. Default `None`. |

## Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save` | `str` | `"/checkpoints"` | Enable checkpoint saving. Default `True`. |
| `save_interval` | `int` | `1000` | Checkpoint save interval. Default `1000`. |
| `megatron_to_hf_mode` | `str` | `"bridge"` | Checkpoint conversion mode (`"bridge"` or `"raw"`). Default `"bridge"`. |
| `use_fault_tolerance` | `bool` | `True` | Enable fault tolerance. Default `False`. |

## Custom Reward

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `custom_rm_path` | `str` | `""` | Python import path for custom reward function. Default `""`. |

## SGLang

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sglang_config` | `dict | None` | `None` | YAML SGLang server configuration. Default `None`. |
| `custom_config_path` | `dict | None` | `None` | YAML custom config overrides. Default `None`. |
| `apply_chat_template_kwargs` | `str` | `""` | Extra kwargs for chat template. Default `None`. |

## Other Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_run_commands` | `list[str]` | `[]` |  |
| `local_python_sources` | `list[str]` | `[]` |  |
| `gpu_type` | `str | None` | `None` |  |
| `sequence_parallel` | `bool | None` | `None` |  |
| `rm_type` | `str` | `"math"` |  |

## Methods

### `build_app(self, *, name: str | None = None, modal: modal_training_gym.frameworks.slime.config.ModalConfig | None = None) -> 'App'`

### `cli_args(self) -> list[str]`

slime CLI arguments derived from this config.

### `prepare_data(self) -> None`

Materialize the training data.

### `to_dataset_config(self) -> 'DatasetConfig'`

Extract dataset-related slime flags back into a `DatasetConfig`.

### `to_model(self) -> 'ModelConfiguration'`

Extract model-related slime flags back into a `ModelConfiguration`.

### `to_wandb_config(self) -> 'WandbConfig'`

Extract wandb-related slime flags back into a `WandbConfig`.

### `total_nodes(self) -> int`

Total Modal cluster nodes required by this config.

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/001_quickstart/)
- [Qwen3-0.6B GRPO on GSM8K (colocated)](/tutorials/rl/001_slime_intro/)
- [Customizing your slime run — scaling nodes, parallelism, and throughput](/tutorials/rl/002_customizing_your_slime_run/)
- [Qwen3-0.6B GRPO on haiku poems — structure score + LLM judge](/tutorials/rl/003_slime_with_llm_as_judge/)

**Source:** [`modal_training_gym/frameworks/slime/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/slime/config.py)
