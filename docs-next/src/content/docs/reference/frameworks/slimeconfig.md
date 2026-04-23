---
title: SlimeConfig
description: API reference for SlimeConfig
---

# SlimeConfig

```python
from modal_training_gym.frameworks.slime.config import SlimeConfig
```

Base SLIME GRPO training configuration.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `environment` | `dict` | `{'PYTHONPATH': '/root/Megatron-LM/', 'CUDA_DEVICE_MAX_CONNECTIONS': '1', 'NCCL_NVLS_ENABLE': '1'}` | Injected into the Ray job runtime env. Default includes `PYTHONPATH`, `CUDA_DEVICE_MAX_CONNECTIONS`, `NCCL_NVLS_ENABLE`. |
| `async_mode` | `bool` | `False` | When `True`, uses `train_async.py` instead of `train.py`. Default `False`. |
| `slime_model_script` | `str` | `""` | Shell script path relative to `/root/slime` that defines `MODEL_ARGS`. Default `""`. |
| `dataset` | `DatasetConfig | None` | `None` | Dataset configuration. Fields expanded to SLIME flags via `_dataset_to_fields()`. Default `None`. |
| `model` | `ModelConfiguration | None` | `None` | Model identity. Fields expanded to SLIME flags via `_model_to_fields()`. Requires `ModelArchitecture`. Default `None`. |
| `wandb` | `WandbConfig | None` | `None` | W&B logging config. Fields expanded to SLIME flags via `_wandb_to_fields()`. Default `None`. |
| `app_tags` | `dict` | `{}` | Extra Modal app tags merged at build time. Default `{}`. |
| `modal` | `ModalConfig | None` | `None` | Modal infrastructure config. Default `None`. |

## Methods

### `build_app(self, *, name: str | None = None, modal: modal_training_gym.frameworks.slime.config.ModalConfig | None = None) -> 'App'`

### `cli_args(self) -> list[str]`

SLIME CLI arguments derived from this config.

### `prepare_data(self) -> None`

Materialize the training data.

### `to_dataset_config(self) -> 'DatasetConfig'`

Extract dataset-related SLIME flags back into a `DatasetConfig`.

### `to_model(self) -> 'ModelConfiguration'`

Extract model-related SLIME flags back into a `ModelConfiguration`.

### `to_wandb_config(self) -> 'WandbConfig'`

Extract wandb-related SLIME flags back into a `WandbConfig`.

### `total_nodes(self) -> int`

Total Modal cluster nodes required by this config.

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/quickstart/)
- [Qwen3-4B GRPO on GSM8K (colocated)](/tutorials/rl/slime_gsm8k/)
- [Qwen3-4B GRPO on haiku poems — structure score + LLM judge](/tutorials/rl/slime_haiku/)

**Source:** [`modal_training_gym/frameworks/slime/config.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/frameworks/slime/config.py)
