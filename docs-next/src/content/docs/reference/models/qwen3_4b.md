---
title: Qwen3-4B
description: API reference for Qwen3_4B
---

# Qwen3_4B

```python
from modal_training_gym.common.models.qwen3_4b import Qwen3_4B
```

Qwen3-4B (4 billion parameters) from Alibaba.

**Inherits from:** `HFModelConfiguration`, `ModelConfiguration`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen3-4B"` | HuggingFace repo ID or other model identifier. Default `""`. |
| `model_path` | `str | None` | `None` | Override local path for model weights. When `None`, frameworks derive the path from `model_name`. Default `None`. |
| `architecture` | `ModelArchitecture | None` | `ModelArchitecture(num_layers=36, hidden_size=2560, ffn_hidden_size=9728, num_attention_heads=32, group_query_attention=True, num_query_groups=8, kv_channels=128, vocab_size=151936, normalization='RMSNorm', norm_epsilon=1e-06, swiglu=True, disable_bias_linear=True, qk_layernorm=True, use_rotary_position_embeddings=True, rotary_base=1000000)` | Transformer architecture spec for Megatron-based frameworks. Not required for HuggingFace-only workflows. Default `None`. |
| `training` | `ModelTrainingConfig | None` | `None` | Optimal training parameters (parallelism, MoE, LoRA) for this model on a specific GPU type. Frameworks pull defaults from here so users don't need to manually specify model-tuned flags. Default `None`. |

## Methods

### `download_model(self) -> 'None'`

Download or materialize weights into the model volume.

## Related Tutorials

- [Customizing your slime run — scaling nodes, parallelism, and throughput](/tutorials/rl/002_customizing_your_slime_run/)
- [Qwen3-4B RL code-golf on MBPP with Harbor sandboxes](/tutorials/rl/004_harbor_codegolf/)

**Source:** [`modal_training_gym/common/models/qwen3_4b.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/modal_training_gym/common/models/qwen3_4b.py)
