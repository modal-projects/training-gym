---
title: WandbConfig
description: API reference for WandbConfig
---

# WandbConfig

```python
from modal_training_gym.common.wandb import WandbConfig
```

Weights & Biases logging configuration shared across all frameworks.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `project` | `str` | `""` | W&B project name. Default `""`. |
| `group` | `str` | `""` | W&B group tag for organizing related runs. Default `""`. |
| `exp_name` | `str` | `""` | W&B run display name. Default `""`. |
| `key` | `str` | `""` | W&B API key. Usually injected via `WANDB_API_KEY` at launch time rather than hardcoded. Default `""`. |
| `disable_random_suffix` | `bool` | `True` | When `True`, suppresses the random suffix that W&B appends to run names. Default `True`. |

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/quickstart/)
- [Qwen3-4B RL code-golf on MBPP with Harbor sandboxes](/tutorials/rl/harbor_code_golf/)
- [Qwen3-4B GRPO on GSM8K (colocated)](/tutorials/rl/slime_gsm8k/)
- [Qwen3-4B GRPO on haiku poems — structure score + LLM judge](/tutorials/rl/slime_haiku/)
- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfiguration` subclass, no catalog entry](/tutorials/sft/ms_swift_custom_hf/)
- [GLM-4.7 LoRA SFT on GSM8K (Megatron)](/tutorials/sft/ms_swift_glm_4_7_gsm8k/)

**Source:** [`modal_training_gym/common/wandb.py`](https://github.com/modal-projects/training-gym/blob/joy/initial-setup/modal_training_gym/common/wandb.py)
