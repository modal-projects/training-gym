---
title: GLM-4.7
description: API reference for GLM_4_7
---

# GLM_4_7

```python
from modal_training_gym.common.models.glm_4_7 import GLM_4_7
```

GLM-4.7 large MoE model from Zhipu AI.

**Inherits from:** `HFModelConfiguration`, `ModelConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"zai-org/GLM-4.7"` | HuggingFace repo ID or other model identifier. Default `""`. |
| `model_path` | `str | None` | `None` | Override local path for model weights. When `None`, frameworks derive the path from `model_name`. Default `None`. |
| `architecture` | `ModelArchitecture | None` | `None` | Transformer architecture spec for Megatron-based frameworks. Not required for HuggingFace-only workflows. Default `None`. |
| `training` | `ModelTrainingConfig | None` | `ModelTrainingConfig(gpu_type='H100', n_nodes=4, gpus_per_node=8, tensor_model_parallel_size=2, pipeline_model_parallel_size=4, context_parallel_size=1, sequence_parallel=True, expert_model_parallel_size=4, moe_permute_fusion=True, moe_grouped_gemm=True, moe_shared_expert_overlap=True, moe_aux_loss_coeff=0.001, lora_rank=128, lora_alpha=32, target_modules='all-linear', merge_lora=False)` | Optimal training parameters (parallelism, MoE, LoRA) for this model on a specific GPU type. Frameworks pull defaults from here so users don't need to manually specify model-tuned flags. Default `None`. |
| `deploy` | `modal_training_gym.common.deploy.DeployConfig | None` | `None` | vLLM serving configuration (GPU, extra args, deploy strategy). When `None`, `serve()` infers settings from the model's training presets. Default `None`. |

## Methods

### `download(self) -> 'None'`

Download or materialize weights into the model volume.

### `serve(self, *, app_name: 'str | None' = None, served_model_name: 'str | None' = None, checkpoint_path: 'str | None' = None, checkpoints_volume: 'str | None' = None, checkpoints_mount_path: 'str | None' = None) -> "'ModelDeployment'"`

Build + deploy a vLLM serving app and return endpoint metadata.

**Source:** [`modal_training_gym/common/models/glm_4_7.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/glm_4_7.py)
