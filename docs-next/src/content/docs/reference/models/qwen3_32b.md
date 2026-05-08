---
title: Qwen3-32B
description: API reference for Qwen3_32B
---

```python
from modal_training_gym.common.models.qwen3_32b import Qwen3_32B
```

Qwen3-32B (32 billion parameters) from Alibaba.

**Inherits from:** `HFModelConfiguration`, `ModelConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen3-32B"` |  |
| `model_path` | `str | None` | `None` |  |
| `architecture` | `ModelArchitecture | None` | `ModelArchitecture(num_layers=64, hidden_size=5120, ffn_hidden_size=25600, num_attention_heads=64, group_query_attention=True, num_query_groups=8, kv_channels=128, vocab_size=151936, normalization='RMSNorm', norm_epsilon=1e-06, swiglu=True, disable_bias_linear=True, qk_layernorm=True, untie_embeddings_and_output_weights=True, use_rotary_position_embeddings=True, rotary_base=1000000)` |  |

## Methods

### `download(self) -> 'None'`

Download or materialize weights into the model volume.

**Source:** [`modal_training_gym/common/models/qwen3_32b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/qwen3_32b.py)
