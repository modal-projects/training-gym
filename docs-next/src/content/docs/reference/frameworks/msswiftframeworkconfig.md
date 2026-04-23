---
title: MsSwiftFrameworkConfig
description: API reference for MsSwiftFrameworkConfig
---

# MsSwiftFrameworkConfig

```python
from modal_training_gym.frameworks.ms_swift.config import MsSwiftFrameworkConfig
```

ms-swift Megatron SFT configuration, including Modal infrastructure.

## Modal Infrastructure

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu` | `Literal['H100', 'H200', 'B200', 'B300']` | `"H100"` | Modal GPU type. Default `"H100"`. |
| `image` | `str` | `"modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.3"` | Docker image for the training container. |
| `transformers_version` | `str` | `"4.57.3"` | Transformers version to reinstall for compatibility. Default `"4.57.3"`. |
| `app_tags` | `dict` | `{}` | Extra Modal app tags. Default `{}`. |
| `environment` | `dict[str, str]` | `{'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'CUDA_DEVICE_MAX_CONNECTIONS': '1'}` | Environment variables for the training container. |
| `n_nodes` | `int` | `4` | Number of Modal cluster nodes. Default `4`. |
| `gpus_per_node` | `int` | `8` | GPUs per node. Default `8`. |

## Tuner and Data

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tuner_type` | `str` | `"lora"` | Fine-tuning method (e.g. `"lora"`). Default `"lora"`. |
| `split_dataset_ratio` | `float` | `0.01` | Train/eval split ratio. Default `0.01`. |
| `perform_initialization` | `bool` | `True` | Emit `--perform_initialization` flag. Default `True`. |

## Parallelism

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tensor_model_parallel_size` | `int` | `2` | Tensor parallelism degree. Default `2`. |
| `expert_model_parallel_size` | `int` | `4` | Expert parallelism for MoE models. Default `4`. |
| `pipeline_model_parallel_size` | `int` | `4` | Pipeline parallelism degree. Default `4`. |
| `context_parallel_size` | `int` | `1` | Context parallelism degree. Default `1`. |
| `sequence_parallel` | `bool` | `True` | Enable sequence parallelism. Default `True`. |
| `use_distributed_optimizer` | `bool` | `True` | Enable distributed optimizer. Default `True`. |

## MoE

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `moe_permute_fusion` | `bool` | `True` | Enable MoE permute fusion. Default `True`. |
| `moe_grouped_gemm` | `bool` | `True` | Enable grouped GEMM for MoE. Default `True`. |
| `moe_shared_expert_overlap` | `bool` | `True` | Enable shared expert overlap. Default `True`. |
| `moe_aux_loss_coeff` | `float` | `0.001` | Auxiliary loss coefficient for MoE load balancing. Default `1e-3`. |

## Batch

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `global_batch_size` | `int` | `8` | Global batch size across all ranks. Default `8`. |
| `packing` | `bool` | `True` | Enable sequence packing. Default `True`. |
| `padding_free` | `bool` | `True` | Enable padding-free attention. Default `True`. |
| `use_precision_aware_optimizer` | `bool` | `True` | Enable precision-aware optimizer. Default `True`. |

## Training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_iters` | `int | None` | `None` | Max training iterations. Overrides `num_train_epochs`. Default `None`. |
| `num_train_epochs` | `int` | `4` | Number of training epochs. Default `4`. |
| `lr` | `float` | `0.0001` | Learning rate. Default `1e-4`. |
| `lr_warmup_fraction` | `float` | `0.05` | Fraction of training for LR warmup. Default `0.05`. |
| `lr_decay_style` | `str` | `"cosine"` | LR decay schedule. Default `"cosine"`. |
| `min_lr` | `float` | `1e-05` | Minimum learning rate. Default `1e-5`. |
| `weight_decay` | `float` | `0.1` | Weight decay coefficient. Default `0.1`. |
| `clip_grad` | `float` | `1.0` | Gradient clipping norm. Default `1.0`. |
| `adam_beta1` | `float` | `0.9` | Adam beta1. Default `0.9`. |
| `adam_beta2` | `float` | `0.95` | Adam beta2. Default `0.95`. |
| `seed` | `int` | `42` | Random seed. Default `42`. |

## Precision and Memory

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bf16` | `bool` | `True` | Enable BF16 training. Default `True`. |
| `recompute_granularity` | `str` | `"selective"` | Activation recomputation granularity. Default `"selective"`. |
| `recompute_modules` | `str` | `"core_attn"` | Modules to recompute. Default `"core_attn"`. |
| `attention_softmax_in_fp32` | `bool` | `True` | Compute attention softmax in FP32. Default `True`. |

## Context and Attention

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_length` | `int` | `2048` | Maximum sequence length. Default `2048`. |
| `attention_backend` | `str` | `"flash"` | Attention implementation. Default `"flash"`. |

## IO and Checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_num_proc` | `int` | `8` | Dataset preprocessing workers. Default `8`. |
| `dataloader_num_workers` | `int` | `4` | DataLoader workers. Default `4`. |
| `save_interval` | `int` | `50` | Checkpoint save interval (iterations). Default `50`. |
| `no_save_optim` | `bool` | `True` | Skip saving optimizer state. Default `True`. |
| `no_save_rng` | `bool` | `True` | Skip saving RNG state. Default `True`. |
| `save_safetensors` | `bool` | `True` | Save in safetensors format. Default `True`. |
| `use_hf` | `int` | `1` | Use HuggingFace checkpoint format. Default `1`. |
| `add_version` | `bool` | `False` | Add version sub-directory to save path. Default `False`. |

## Logging and Eval

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_interval` | `int` | `1` | Logging interval (iterations). Default `1`. |
| `eval_iters` | `int` | `10` | Evaluation iterations. Default `10`. |
| `eval_interval` | `int` | `50` | Evaluation interval (iterations). Default `50`. |

## LoRA

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_modules` | `str` | `"all-linear"` | LoRA target modules. Default `"all-linear"`. |
| `lora_rank` | `int` | `128` | LoRA rank. Default `128`. |
| `lora_alpha` | `int` | `32` | LoRA alpha scaling. Default `32`. |
| `lora_dropout` | `float` | `0.05` | LoRA dropout. Default `0.05`. |
| `merge_lora` | `bool` | `False` | Merge LoRA weights after training. Default `False`. |

## Methods

### `from_toml(path: 'str | Path') -> "'MsSwiftFrameworkConfig'"`

## Related Tutorials

- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfiguration` subclass, no catalog entry](/tutorials/sft/ms_swift_custom_hf/)
- [GLM-4.7 LoRA SFT on GSM8K (Megatron)](/tutorials/sft/ms_swift_glm_4_7_gsm8k/)

**Source:** [`modal_training_gym/frameworks/ms_swift/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/ms_swift/config.py)
