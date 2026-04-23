---
title: DatasetConfig
description: API reference for DatasetConfig
---

# DatasetConfig

```python
from modal_training_gym.common.dataset import DatasetConfig
```

Dataset configuration shared across training frameworks.

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_data` | `str` | `""` | Path to the training data file (e.g. a `.parquet` file on the data volume). Default `""`. |
| `eval_prompt_data` | `list[str] | str | None` | `None` | Evaluation data path(s). Can be a single path, a list of paths, or `None` to skip evaluation. Default `None`. |
| `input_key` | `str` | `""` | Column/key name for model input in the dataset. Default `""`. |
| `label_key` | `str` | `""` | Column/key name for labels/targets in the dataset. Default `""`. |
| `apply_chat_template` | `bool` | `True` | Whether to apply the model's chat template to inputs. Default `True`. |
| `rollout_shuffle` | `bool` | `True` | Whether to shuffle data during rollout generation. Default `True`. |
| `rm_type` | `str` | `""` | Reward model type identifier (e.g. `"math"`). Default `""`. |

## Methods

### `prepare(self) -> 'None'`

Download and/or preprocess the dataset into the data volume.

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/quickstart/)
- [Qwen3-4B RL code-golf on MBPP with Harbor sandboxes](/tutorials/rl/harbor_code_golf/)
- [Qwen3-4B GRPO on GSM8K (colocated)](/tutorials/rl/slime_gsm8k/)
- [Qwen3-4B GRPO on haiku poems — structure score + LLM judge](/tutorials/rl/slime_haiku/)
- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfiguration` subclass, no catalog entry](/tutorials/sft/ms_swift_custom_hf/)
- [GLM-4.7 LoRA SFT on GSM8K (Megatron)](/tutorials/sft/ms_swift_glm_4_7_gsm8k/)

**Source:** [`modal_training_gym/common/dataset.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/dataset.py)
