---
title: Qwen3-30B-A3B
description: API reference for Qwen3_30B
---

# Qwen3_30B

```python
from modal_training_gym.common.models.qwen3_30b import Qwen3_30B
```

Qwen3-30B-A3B (30B total, ~3B active) MoE model from Alibaba.

**Inherits from:** `HFModelConfiguration`, `ModelConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen3-30B-A3B"` |  |
| `model_path` | `str | None` | `None` |  |
| `architecture` | `ModelArchitecture | None` | `ModelArchitecture(num_layers=48, hidden_size=2048, ffn_hidden_size=6144, num_attention_heads=32, group_query_attention=True, num_query_groups=4, kv_channels=128, vocab_size=151936, normalization='RMSNorm', norm_epsilon=1e-06, swiglu=True, disable_bias_linear=True, qk_layernorm=True, use_rotary_position_embeddings=True, rotary_base=1000000)` |  |

## Methods

### `download(self) -> 'None'`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/qwen3_30b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/qwen3_30b.py)
