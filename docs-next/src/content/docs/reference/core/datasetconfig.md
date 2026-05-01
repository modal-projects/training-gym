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

## Methods

### `load(self)`

Load raw examples for local inspection or evaluation.

### `prepare(self) -> 'None'`

Download and/or preprocess the dataset into the data volume.

## Related Tutorials

- [Shared concepts: config containers, framework factories, volume layout, running the pipeline](/tutorials/intro/001_quickstart/)
- [Custom HuggingFace model (SmolLM2-135M) LoRA SFT — inline `ModelConfig` subclass, no catalog entry](/tutorials/intro/002_custom_model/)
- [Qwen3-0.6B GRPO on GSM8K (colocated)](/tutorials/rl/001_slime_intro/)
- [Customizing your slime run — scaling nodes, parallelism, and throughput](/tutorials/rl/002_customizing_your_slime_run/)
- [Qwen3-0.6B GRPO on haiku poems — structure score + LLM judge](/tutorials/rl/003_slime_with_llm_as_judge/)
- [GLM-4.7 LoRA SFT on GSM8K (Megatron)](/tutorials/sft/001_ms_swift/)

**Source:** [`modal_training_gym/common/dataset.py`](https://github.com/modal-projects/training-gym/blob/main/modal_training_gym/common/dataset.py)
