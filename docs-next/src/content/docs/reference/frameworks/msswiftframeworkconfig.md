---
title: MsSwiftFrameworkConfig
description: API reference for MsSwiftFrameworkConfig
---

# MsSwiftFrameworkConfig

```python
from modal_training_gym.frameworks.ms_swift.config import MsSwiftFrameworkConfig
```

ms-swift Megatron configuration, including Modal infrastructure.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu` | `Literal['H100', 'H200', 'B200', 'B300']` | `"H100"` |  |
| `image` | `str` | `"modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.10.3"` |  |
| `transformers_version` | `str` | `"4.57.3"` |  |
| `app_tags` | `dict` | `<dataclasses._MISSING_TYPE object at 0x7f6b4318eb70>` |  |
| `environment` | `dict[str, str]` | `<dataclasses._MISSING_TYPE object at 0x7f6b4318eb70>` |  |
| `n_nodes` | `int` | `4` |  |
| `gpus_per_node` | `int` | `8` |  |
| `tuner_type` | `str` | `"lora"` |  |
| `split_dataset_ratio` | `float` | `0.01` |  |
| `perform_initialization` | `bool` | `True` |  |
| `tensor_model_parallel_size` | `int` | `2` |  |
| `expert_model_parallel_size` | `int` | `4` |  |
| `pipeline_model_parallel_size` | `int` | `4` |  |
| `context_parallel_size` | `int` | `1` |  |
| `sequence_parallel` | `bool` | `True` |  |
| `use_distributed_optimizer` | `bool` | `True` |  |
| `moe_permute_fusion` | `bool` | `True` |  |
| `moe_grouped_gemm` | `bool` | `True` |  |
| `moe_shared_expert_overlap` | `bool` | `True` |  |
| `moe_aux_loss_coeff` | `float` | `0.001` |  |
| `global_batch_size` | `int` | `8` |  |
| `packing` | `bool` | `True` |  |
| `padding_free` | `bool` | `True` |  |
| `use_precision_aware_optimizer` | `bool` | `True` |  |
| `train_iters` | `int | None` | `None` |  |
| `num_train_epochs` | `int` | `4` |  |
| `lr` | `float` | `0.0001` |  |
| `lr_warmup_fraction` | `float` | `0.05` |  |
| `lr_decay_style` | `str` | `"cosine"` |  |
| `min_lr` | `float` | `1e-05` |  |
| `weight_decay` | `float` | `0.1` |  |
| `clip_grad` | `float` | `1.0` |  |
| `adam_beta1` | `float` | `0.9` |  |
| `adam_beta2` | `float` | `0.95` |  |
| `seed` | `int` | `42` |  |
| `bf16` | `bool` | `True` |  |
| `recompute_granularity` | `str` | `"selective"` |  |
| `recompute_modules` | `str` | `"core_attn"` |  |
| `attention_softmax_in_fp32` | `bool` | `True` |  |
| `max_length` | `int` | `2048` |  |
| `attention_backend` | `str` | `"flash"` |  |
| `dataset_num_proc` | `int` | `8` |  |
| `dataloader_num_workers` | `int` | `4` |  |
| `save_interval` | `int` | `50` |  |
| `no_save_optim` | `bool` | `True` |  |
| `no_save_rng` | `bool` | `True` |  |
| `save_safetensors` | `bool` | `True` |  |
| `use_hf` | `int` | `1` |  |
| `add_version` | `bool` | `False` |  |
| `log_interval` | `int` | `1` |  |
| `eval_iters` | `int` | `10` |  |
| `eval_interval` | `int` | `50` |  |
| `target_modules` | `str` | `"all-linear"` |  |
| `lora_rank` | `int` | `128` |  |
| `lora_alpha` | `int` | `32` |  |
| `lora_dropout` | `float` | `0.05` |  |
| `merge_lora` | `bool` | `False` |  |

## Methods

### `from_toml(path: 'str | Path') -> "'MsSwiftFrameworkConfig'"`

**Source:** [`modal_training_gym/frameworks/ms_swift/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/ms_swift/config.py)
