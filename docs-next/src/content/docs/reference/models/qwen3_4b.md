---
title: Qwen3-4B
description: API reference for Qwen3_4B
---

# Qwen3_4B

```python
from modal_training_gym.common.models.qwen3_4b import Qwen3_4B
```

Qwen3-4B (4 billion parameters) from Alibaba.

**Inherits from:** `HFModelConfiguration`, `ModelConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen3-4B"` | HuggingFace repo ID or other model identifier. Default `""`. |
| `model_path` | `str | None` | `None` | Override local path for model weights. When `None`, frameworks derive the path from `model_name`. Default `None`. |
| `architecture` | `ModelArchitecture | None` | `ModelArchitecture(num_layers=36, hidden_size=2560, ffn_hidden_size=9728, num_attention_heads=32, group_query_attention=True, num_query_groups=8, kv_channels=128, vocab_size=151936, normalization='RMSNorm', norm_epsilon=1e-06, swiglu=True, disable_bias_linear=True, qk_layernorm=True, use_rotary_position_embeddings=True, rotary_base=1000000)` | Transformer architecture spec for Megatron-based frameworks. Not required for HuggingFace-only workflows. Default `None`. |
| `training` | `ModelTrainingConfig | None` | `None` | Optimal training parameters (parallelism, MoE, LoRA) for this model on a specific GPU type. Frameworks pull defaults from here so users don't need to manually specify model-tuned flags. Default `None`. |
| `checkpoints_volume_name` | `str` | `""` | Optional Modal volume name to mount when `model_path` points to a local checkpoint path. Typically injected by `TrainResult.model`. |
| `checkpoints_mount_path` | `str` | `"/checkpoints"` | Optional mount path for `checkpoints_volume_name`. Defaults to `"/checkpoints"`. |
| `gpu` | `'GPUType | None'` | `None` | Optional vLLM serving GPU type override. When `None`, inferred from model presets. |
| `n_gpu` | `int | None` | `None` | Optional vLLM serving GPU count override. When `None`, inferred from model tensor-parallel presets. |
| `extra_vllm_args` | `list[str] | None` | `None` | Optional additional vLLM CLI args for serving. |
| `environment_name` | `str | None` | `None` | Optional Modal environment name to deploy into. |
| `deploy_strategy` | `str` | `"rolling"` | Modal deployment strategy. Default `"rolling"`. |

## Methods

### `download(self) -> 'None'`

Download or materialize weights into the model volume.

### `serve(self, *, app_name: 'str | None' = None, served_model_name: 'str | None' = None, checkpoint_path: 'str | None' = None, checkpoints_volume: 'str | None' = None, checkpoints_mount_path: 'str | None' = None) -> "'ModelDeployment'"`

Build + deploy a vLLM serving app and return endpoint metadata.

## Related Tutorials

- [Qwen3-4B haiku evaluation with verifiable rewards — serve, evaluate, train, compare](/tutorials/rl/000_rl_basics/)
- [Customizing your slime run — scaling nodes, parallelism, and throughput](/tutorials/rl/002_customizing_your_slime_run/)

**Source:** [`modal_training_gym/common/models/qwen3_4b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/qwen3_4b.py)
