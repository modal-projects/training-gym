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

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/001_quickstart/)
- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfiguration` subclass, no catalog entry](/tutorials/intro/002_custom_model/)
- [Qwen3-0.6B GRPO on GSM8K (colocated)](/tutorials/rl/001_slime_intro/)
- [Customizing your slime run — scaling nodes, parallelism, and throughput](/tutorials/rl/002_customizing_your_slime_run/)
- [Qwen3-0.6B GRPO on haiku poems — structure score + LLM judge](/tutorials/rl/003_slime_with_llm_as_judge/)
- [Qwen3-4B RL code-golf on MBPP with Harbor sandboxes](/tutorials/rl/004_harbor_codegolf/)
- [GLM-4.7 LoRA SFT on GSM8K (Megatron)](/tutorials/sft/001_ms_swift/)

**Source:** [`modal_training_gym/common/wandb.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/wandb.py)
