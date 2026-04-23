---
title: ModelArchitecture
description: API reference for ModelArchitecture
---

# ModelArchitecture

```python
from modal_training_gym.common.models.base import ModelArchitecture
```

Transformer architecture parameters for a specific model.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_layers` | `int` | `0` | Number of transformer layers. Default `0`. |
| `hidden_size` | `int` | `0` | Hidden dimension size. Default `0`. |
| `ffn_hidden_size` | `int` | `0` | Feed-forward network intermediate size. Default `0`. |
| `num_attention_heads` | `int` | `0` | Number of attention heads. Default `0`. |
| `group_query_attention` | `bool` | `True` | Enable grouped-query attention (GQA). Default `True`. |
| `num_query_groups` | `int` | `0` | Number of KV head groups for GQA. Default `0`. |
| `kv_channels` | `int` | `0` | Per-head key/value channel dimension. Default `0`. |
| `vocab_size` | `int` | `0` | Vocabulary size. Default `0`. |
| `normalization` | `str` | `"RMSNorm"` | Layer normalization type. Default `"RMSNorm"`. |
| `norm_epsilon` | `float` | `1e-06` | Normalization epsilon. Default `1e-6`. |
| `swiglu` | `bool` | `True` | Use SwiGLU activation in FFN. Default `True`. |
| `disable_bias_linear` | `bool` | `True` | Disable bias in linear layers. Default `True`. |
| `qk_layernorm` | `bool` | `True` | Apply layer norm to query and key projections. Default `True`. |
| `use_rotary_position_embeddings` | `bool` | `True` | Use RoPE positional encoding. Default `True`. |
| `rotary_base` | `int` | `10000` | Base frequency for RoPE. Default `10000`. |

**Source:** [`modal_training_gym/common/models/base.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/base.py)
