---
title: Qwen3-0.6B
description: API reference for Qwen3_0_6B
---

# Qwen3_0_6B

```python
from modal_training_gym.common.models.qwen3_0_6b import Qwen3_0_6B
```

Qwen3-0.6B (0.6 billion parameters) from Alibaba.

**Inherits from:** `HFModelConfiguration`, `ModelConfig`

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_name` | `str` | `"Qwen/Qwen3-0.6B"` |  |
| `model_path` | `str | None` | `None` |  |
| `architecture` | `ModelArchitecture | None` | `ModelArchitecture(num_layers=28, hidden_size=1024, ffn_hidden_size=3072, num_attention_heads=16, group_query_attention=True, num_query_groups=8, kv_channels=128, vocab_size=151936, normalization='RMSNorm', norm_epsilon=1e-06, swiglu=True, disable_bias_linear=True, qk_layernorm=True, use_rotary_position_embeddings=True, rotary_base=1000000)` |  |

## Methods

### `download(self) -> 'None'`

Download or materialize weights into the model volume.

## Related Tutorials

- [Multi-turn number-guessing RL with custom generate and reward functions](/tutorials/rl/002_multiturn_number_guessing/)

**Source:** [`modal_training_gym/common/models/qwen3_0_6b.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/models/qwen3_0_6b.py)
