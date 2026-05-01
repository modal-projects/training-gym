---
title: Llama2-7B
description: API reference for Llama2_7B
---

# Llama2_7B

```python
from modal_training_gym.common.models.llama2_7b import Llama2_7B
```

Llama 2 7B from Meta.

**Inherits from:** `HFModelConfiguration`, `ModelConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"meta-llama/Llama-2-7b-hf"` | HuggingFace repo ID or other model identifier. Default `""`. |
| `model_path` | `str | None` | `None` | Override local path for model weights. When `None`, frameworks derive the path from `model_name`. Default `None`. |
| `architecture` | `ModelArchitecture | None` | `None` | Transformer architecture spec for Megatron-based frameworks. Not required for HuggingFace-only workflows. Default `None`. |
| `training` | `ModelTrainingConfig | None` | `ModelTrainingConfig(gpu_type='H100', n_nodes=1, gpus_per_node=8, tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1, sequence_parallel=False, expert_model_parallel_size=1, moe_permute_fusion=False, moe_grouped_gemm=False, moe_shared_expert_overlap=False, moe_aux_loss_coeff=0.0, lora_rank=8, lora_alpha=16, target_modules='all-linear', merge_lora=False)` | Optimal training parameters (parallelism, MoE, LoRA) for this model on a specific GPU type. Frameworks pull defaults from here so users don't need to manually specify model-tuned flags. Default `None`. |
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

**Source:** [`modal_training_gym/common/models/llama2_7b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/llama2_7b.py)
