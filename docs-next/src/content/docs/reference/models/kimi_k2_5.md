---
title: Kimi-K2.5
description: API reference for Kimi_K2_5
---

# Kimi_K2_5

```python
from modal_training_gym.common.models.kimi_k2_5 import Kimi_K2_5
```

Kimi K2.5 from Moonshot AI.

**Inherits from:** `HFModelConfiguration`, `ModelConfiguration`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"moonshotai/Kimi-K2.5"` | HuggingFace repo ID or other model identifier. Default `""`. |
| `model_path` | `str | None` | `"/checkpoints/Kimi-K2.5-bf16"` | Override local path for model weights. When `None`, frameworks derive the path from `model_name`. Default `None`. |
| `architecture` | `ModelArchitecture | None` | `None` | Transformer architecture spec for Megatron-based frameworks. Not required for HuggingFace-only workflows. Default `None`. |
| `training` | `ModelTrainingConfig | None` | `ModelTrainingConfig(gpu_type='H100', n_nodes=4, gpus_per_node=8, tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1, sequence_parallel=False, expert_model_parallel_size=1, moe_permute_fusion=False, moe_grouped_gemm=False, moe_shared_expert_overlap=False, moe_aux_loss_coeff=0.0, lora_rank=8, lora_alpha=16, target_modules='all-linear', merge_lora=False)` | Optimal training parameters (parallelism, MoE, LoRA) for this model on a specific GPU type. Frameworks pull defaults from here so users don't need to manually specify model-tuned flags. Default `None`. |

## Methods

### `download_model(self) -> None`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/kimi_k2_5.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/modal_training_gym/common/models/kimi_k2_5.py)
